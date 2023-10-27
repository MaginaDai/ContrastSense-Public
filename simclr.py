import logging
import math
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from judge import AverageMeter
from utils import save_config_file, accuracy, save_checkpoint, evaluate, split_last, merge_last


class MyNet(nn.Module):

    def __init__(self, transfer=True, if_bn=True, if_g=True, if_lstm=True, out_dim=16, classes=6, hidden_dim=32):
        super(MyNet, self).__init__()
        self.transfer = transfer
        self.if_bn = if_bn
        self.if_g = if_g
        self.if_lstm = if_lstm

        self.dropout = torch.nn.Dropout(p=0.1)
        self.relu = torch.nn.ReLU()

        self.BN_acc = torch.nn.BatchNorm2d(num_features=32, momentum=0.1, affine=False)
        self.BN_gyro = torch.nn.BatchNorm2d(num_features=32, momentum=0.1, affine=False)

        self.conv1_acc = torch.nn.Conv2d(1, 32, kernel_size=(25, 3), stride=1)
        self.conv1_gyro = torch.nn.Conv2d(1, 32, kernel_size=(25, 3), stride=1)

        self.conv2_acc_1 = torch.nn.Conv2d(32, 32, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv2_acc_2 = torch.nn.Conv2d(32, 32, kernel_size=(5, 1), stride=1, padding=(2, 0))

        self.conv2_gyro_1 = torch.nn.Conv1d(32, 32, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv2_gyro_2 = torch.nn.Conv1d(32, 32, kernel_size=(5, 1), stride=1, padding=(2, 0))

        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=(5, 2), stride=1, padding=(2, 0))

        self.conv4_1 = torch.nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv4_2 = torch.nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)

        self.attn = MultiHeadedSelfAttention(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.pwff = PositionWiseFeedForward(hidden_dim=hidden_dim, hidden_ff=hidden_dim*2)
        self.norm2 = LayerNorm(hidden_dim)

        # self.lstm1 = torch.nn.LSTM(32, 16, num_layers=2, batch_first=True, bidirectional=True)
        # self.lstm2 = torch.nn.LSTM(16*2, 4, num_layers=1, batch_first=True, bidirectional=True)

        self.gru1 = torch.nn.GRU(32, 16, num_layers=2, batch_first=True, bidirectional=True)
        self.gru2 = torch.nn.GRU(16 * 2, 4, num_layers=1, batch_first=True, bidirectional=True)

        if self.if_lstm:
            self.linear = torch.nn.Linear(in_features=384*2, out_features=out_dim)
        else:
            self.linear = torch.nn.Linear(in_features=3072, out_features=out_dim)

        self.linear3 = torch.nn.Linear(in_features=out_dim, out_features=out_dim)
        self.linear4 = torch.nn.Linear(in_features=out_dim, out_features=out_dim)

        if self.transfer:
            self.linear2 = torch.nn.Linear(in_features=out_dim, out_features=classes)

    def forward(self, x):
        # extract in-sensor information
        x_acc = self.relu(self.dropout(self.conv1_acc(x[:, :, :, 0:3])))
        x_gyro = self.relu(self.dropout(self.conv1_gyro(x[:, :, :, 3:])))

        # ResNet Arch for high-level information
        x1 = x_acc
        x_acc = self.relu(self.dropout(self.conv2_acc_1(x_acc)))
        x_acc = self.dropout(self.conv2_acc_2(x_acc))
        x_acc = self.relu(x_acc + x1)

        x1 = x_gyro
        x_gyro = self.relu(self.dropout(self.conv2_gyro_1(x_gyro)))
        x_gyro = self.dropout(self.conv2_gyro_2(x_gyro))
        x_gyro = self.relu(x_gyro + x1)

        # we need to normalize the data to make the features comparable
        if self.if_bn:
            x_acc = self.BN_acc(x_acc)  # add visualization?
            x_gyro = self.BN_gyro(x_gyro)

        x = torch.cat((x_acc, x_gyro), dim=3)
        # extract intra-sensor information
        x = self.relu(self.dropout(self.conv3(x)))
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        # ResNet Arch for high-level information
        x1 = x
        x = self.relu(self.dropout(self.conv4_1(x)))
        x = self.dropout(self.conv4_2(x))
        x = self.relu(x + x1)

        x = x.view(x.shape[0], x.shape[2], -1)

        h = self.attn(x)
        h = self.norm1(h + self.proj(h))
        x = self.norm2(h + self.pwff(h))

        if self.if_lstm:
            # x, _ = self.lstm1(x)
            # x, _ = self.lstm2(x)
            x, _ = self.gru1(x)
            x, _ = self.gru2(x)

        x = x.reshape(x.shape[0], -1)

        # keep the output layer constant with the SimCLR output
        x = self.relu(self.linear(x))

        if self.transfer:  # this is the head for HAR
            x = self.linear2(x)
        else:
            if self.if_g:
                x = self.relu(self.linear3(x))  # add nonlinear / add linear
                x = self.linear4(x)
        return x


# one baseline
class SeqActivityClassifier(nn.Module):
    def __init__(self, out_dim, classes):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)

        self.lstm1 = nn.LSTM(6, 32, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(32, 16, num_layers=1, batch_first=True)

        self.linear1 = nn.Linear(16, 6)
        self.linear2 = torch.nn.Linear(in_features=600, out_features=out_dim)
        self.linear3 = torch.nn.Linear(in_features=out_dim, out_features=classes)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[2], -1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, hidden_dim=72, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        # return x


class Embeddings(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        # factorized embedding
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden)  # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], -1)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)

        # factorized embedding
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)


class MultiProjection(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        return q, k, v


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, hidden_dim=72, n_heads=4):
        super().__init__()
        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = n_heads

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        # scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = q @ k.transpose(-2, -1)

        scores = F.softmax(scores, dim=-1)

        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, hidden_dim=72, hidden_ff=144):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_ff)
        self.fc2 = nn.Linear(hidden_ff, hidden_dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)

    def forward(self, x):
        h = self.embed(x)

        for i in range(self.n_layers):
            # h = block(h, mask)
            h = self.attn(h, i)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))

        return h


class LIMU_encoder(nn.Module):

    def __init__(self, cfg, out_dim=72, classes=6, transfer=False):
        super().__init__()
        self.transfer = transfer
        self.transformer = Transformer(cfg) # encoder
        # self.conv = torch.nn.Conv1d(120, 64, kernel_size=1, stride=1)
        # self.conv2 = torch.nn.Conv1d(64, 32, kernel_size=1, stride=1)
        # self.conv3 = torch.nn.Conv1d(32, 1, kernel_size=1, stride=1)

        self.conv = torch.nn.Conv1d(72, 36, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv1d(36, 1, kernel_size=1, stride=1)
        if self.transfer:
            self.linear2 = torch.nn.Linear(in_features=120, out_features=classes)

    def forward(self, input_seqs, masked_pos=None):
        h = self.transformer(input_seqs)
        h = h.permute(0, 2, 1)
        h = self.conv(h)
        h = self.conv2(h)
        # h = self.conv3(h)
        h = h.view(h.shape[0], -1)
        if self.transfer:
            h = self.linear2(h)
        return h


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        writer_pos = './runs/' + self.args.store + '/' + self.args.name
        if self.args.transfer is True:
            if self.args.if_fine_tune:
                writer_pos += '_ft'
            else:
                writer_pos += '_le'
            if self.args.shot:
                writer_pos += f'_shot_{self.args.shot}'
            else:
                writer_pos += f'_percent_{self.args.percent}'
        else:
            writer_pos += '/'
        self.writer = SummaryWriter(writer_pos)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)



    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        acc = 0
        best_epoch = 0
        best_acc = 0
        not_best_counter = 0
        best_loss = 1e6

        if self.args.resume:
            best_acc = self.args.best_acc
        else:
            best_acc = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')
            loss_batch = AverageMeter('loss_batch', ':6.5f')
            mscr_batch = AverageMeter('mslr_batch', ':6.3f')
            msdr_batch = AverageMeter('msdr_batch', ':6.3f')
            mscsdr_batch = AverageMeter('mscsdr_batch', ':6.3f')

            for sensor, _ in train_loader:
                sensor = [t.to(self.args.device) for t in sensor]
                domain_label = None

                with autocast(enabled=self.args.fp16_precision):
                    logits, labels = self.model(sensor[0], sensor[1], domain_label=domain_label)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                acc = accuracy(logits, labels, topk=(1,))

                acc_batch.update(acc, sensor[0].size(0))
                loss_batch.update(loss, sensor[0].size(0))

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc', acc, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                    self.writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=n_iter)
                    
                n_iter += 1

            is_best = loss_batch.avg <= best_loss
            if epoch_counter >= 10:  # only after the first 10 epochs, the best_acc is updated.
                best_loss = min(loss_batch.avg, best_loss)
                best_acc = max(acc_batch.avg, best_acc)
            if is_best:
                best_epoch = epoch_counter
                checkpoint_name = 'model_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)
                not_best_counter = 0
            else:
                not_best_counter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            
            if epoch_counter == 0:
                continue  # the first epoch would not have record.
            log_str = f"Epoch: {epoch_counter} Loss: {loss_batch.avg} accuracy: {acc_batch.avg} "
            logging.debug(log_str)

            if not_best_counter >= 200:
                print(f"early stop at {epoch_counter}")
                break

        logging.info("Training has finished.")
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def transfer_train(self, tune_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        logging.info(f"Start SimCLR fine-tuning head for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        if not self.args.if_fine_tune:
            """
            Switch to eval mode:
            Under the protocol of linear classification on frozen features/models,
            it is not legitimate to change any part of the pre-trained model.
            BatchNorm in train mode may revise running mean/std (even if it receives
            no gradient), which are part of the model parameters too.
            """
            self.model.eval()

        acc = 0

        if self.args.resume:
            best_acc = self.args.best_acc
        else:
            best_acc = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc1 = AverageMeter('acc1', ':6.2f')
            for sensor, target in tune_loader:

                sensor = sensor.to(self.args.device)
                target = target.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    logits = self.model(sensor)
                    loss = self.criterion(logits, target)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                acc = accuracy(logits, target, topk=(1,))
                acc1.update(acc, sensor.size(0))
                if n_iter_train % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter_train)
                    self.writer.add_scalar('acc', acc, global_step=n_iter_train)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter_train)

                n_iter_train += 1

            if self.args.if_val:
                val_acc = evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)
                if self.args.if_fine_tune:
                    self.model.train()
            else:
                val_acc = 0

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            if is_best:
                # checkpoint_name = 'checkpoint_at_{:04d}.pth.tar'.format(epoch_counter)
                checkpoint_name = 'model_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)

            self.writer.add_scalar('evaluation acc', val_acc, global_step=epoch_counter)
            # self.writer.add_scalar('evaluation recall', val_recall, global_step=epoch_counter)
            # self.writer.add_scalar('evaluation f1', val_f1, global_step=epoch_counter)

            self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter} \t Loss: {loss} \t train acc: {acc1.avg} \t val acc: {val_acc}")

        if_best = bool(1-self.args.if_val)
        logging.info("Fine-tuning has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'acc': acc
        }, is_best=if_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        print('eval acc is {}'.format(best_acc))

    def test_performance(self, best_model_dir, test_loader):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        test_acc = evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=test_loader)
        print('test acc is {}'.format(test_acc))
        logging.info(f"test acc is {test_acc}.")


class BaseLine(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        lr_str = str(self.args.lr)
        if '.' in lr_str:
            writer_pos = './runs/' + self.args.store + '_' + lr_str.replace('.', '_')
        else:
            writer_pos = './runs/' + self.args.store + '_' + lr_str
        if self.args.transfer is True:
            writer_pos += '_fine_tune'
        self.writer = SummaryWriter(writer_pos)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def train(self, train_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        logging.info(f"Start baseline training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        if not self.args.if_fine_tune:
            self.model.eval()
            """
            Switch to eval mode:
            Under the protocol of linear classification on frozen features/models,
            it is not legitimate to change any part of the pre-trained model.
            BatchNorm in train mode may revise running mean/std (even if it receives
            no gradient), which are part of the model parameters too.
            """

        acc = 0

        if self.args.resume:
            best_acc = self.args.best_acc
        else:
            best_acc = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc1 = AverageMeter('acc1', ':6.2f')
            for sensor, target in train_loader:
                self.model.train()
                sensor = sensor.to(self.args.device)
                target = target.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    logits = self.model(sensor)
                    loss = self.criterion(logits, target)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                acc = accuracy(logits, target, topk=(1,))
                acc1.update(acc, sensor.size(0))
                if n_iter_train % self.args.log_every_n_steps == 0:

                    self.writer.add_scalar('loss', loss, global_step=n_iter_train)
                    self.writer.add_scalar('acc', acc, global_step=n_iter_train)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter_train)

                n_iter_train += 1

            if self.args.if_val:
                val_acc = evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)
            else:
                val_acc = 0
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            if is_best:
                checkpoint_name = 'checkpoint_at_{:04d}.pth.tar'.format(epoch_counter)
                save_checkpoint({
                    'epoch': epoch_counter + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)

            self.writer.add_scalar('evaluation acc', val_acc, global_step=epoch_counter)

            # warmup for the first 10 epochs
            if self.args.transfer or epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter} \t Loss: {loss} \t train acc: {acc1.avg} \t val accuracy: {val_acc}")

        if_best = bool(1-self.args.if_val)
        logging.info("Fine-tuning has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'acc': acc
        }, is_best=if_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def test_performance(self, best_model_dir, test_loader):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        test_acc = evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=test_loader)
        print('test acc is {}'.format(test_acc))
        logging.info(f"test acc is {test_acc}.")


