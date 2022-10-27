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
from simclr import MultiHeadedSelfAttention, LayerNorm, PositionWiseFeedForward
from utils import save_config_file, accuracy, save_checkpoint, evaluate, split_last, merge_last, CPC_evaluate

SMALL_NUM = np.log(1e-45)


class CPCEncoder(nn.Module):
    def __init__(self, dims=32):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.relu = torch.nn.ReLU()

        self.BN_acc = torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)
        self.BN_gyro = torch.nn.BatchNorm2d(num_features=dims, momentum=0.1, affine=False)

        self.conv1_acc = torch.nn.Conv2d(1, dims, kernel_size=(25, 3), stride=1)
        self.conv1_gyro = torch.nn.Conv2d(1, dims, kernel_size=(25, 3), stride=1)

        self.conv2_acc_1 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv2_acc_2 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))

        self.conv2_gyro_1 = torch.nn.Conv1d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv2_gyro_2 = torch.nn.Conv1d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))

        self.conv3 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 2), stride=1, padding=(2, 0))

        self.conv4_1 = torch.nn.Conv1d(dims, dims, kernel_size=5, stride=1, padding=2)
        self.conv4_2 = torch.nn.Conv1d(dims, dims, kernel_size=5, stride=1, padding=2)


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

        # x = x.view(x.shape[0], x.shape[2], -1)
        #
        # h = self.attn(x)
        # h = self.norm1(h + self.proj(h))
        # x = self.norm2(h + self.pwff(h))
        #
        # x = x.view(x.shape[0], x.shape[2], -1)
        return x


class Classifier(nn.Module):
    def __init__(self, out_dim=512, classes=6, dims=32):
        super().__init__()
        self.gru1 = nn.GRU(int(dims/2), int(dims/2), num_layers=1, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(dims, 4, num_layers=1, bidirectional=True, batch_first=True)
        self.linear1 = torch.nn.Linear(in_features=384*2, out_features=out_dim)
        self.linear2 = torch.nn.Linear(in_features=out_dim, out_features=classes)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class CPCV1(nn.Module):
    def __init__(self, timestep, batch_size, seq_len, transfer=False, classes=6, dims=32, temperature=1):

        super(CPCV1, self).__init__()
        self.transfer = transfer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.dims = dims
        self.temperature = temperature
        self.l = DCL(temperature=self.temperature)

        self.encoder = CPCEncoder(dims=self.dims)
        self.gru = nn.GRU(self.dims, int(self.dims/2), num_layers=1, bidirectional=False, batch_first=True)
        if self.transfer:
            self.classifier = Classifier(classes=classes, dims=self.dims)

        self.Wk = nn.ModuleList([nn.Linear(int(self.dims/2), self.dims) for i in range(timestep)])
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru "why this step?"
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu:
            return torch.zeros(1, batch_size, 256).cuda()
        else:
            return torch.zeros(1, batch_size, 256)

    def forward(self, x):
        batch = x.size()[0]
        # input sequence is N*C*L, e.g. 8*1*20480
        t_samples = torch.randint(self.seq_len - self.timestep, size=(1,)).long()  # randomly pick time stamps
        # encoded sequence is N*C*L, e.g. 8*512*128
        z = self.encoder(x)
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1, 2)
        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.dims)).float()  # e.g. size 12*8*512
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, self.dims)  # z_tk e.g. size 8*512
        forward_seq = z[:, :t_samples + 1, :]  # e.g. size 8*100*512
        output, _ = self.gru(forward_seq)  # output size e.g. 8*100*256
        c_t = output[:, t_samples, :].view(batch, int(self.dims/2))  # c_t e.g. size 8*256
        pred = torch.empty((self.timestep, batch, self.dims)).float()  # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # Wk*c_t e.g. size 8*512
        correct = torch.zeros(1)
        for i in np.arange(0, self.timestep):
            # total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            total = torch.mm(F.normalize(encode_samples[i], dim=1), torch.transpose(F.normalize(pred[i], dim=1), 0, 1))/self.temperature  # e.g. size 8*8
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
            # nce += self.l(encode_samples[i], pred[i])  # use DCL as the loss calculator
        nce /= -1. * batch * self.timestep
        accuracy = 1. * correct.item() / batch
        return accuracy, nce

    def predict(self, x):
        # input sequence is N*C*L, e.g. 8*1*20480
        batch = x.size()[0]
        # encoded sequence is N*C*L, e.g. 8*512*128
        z = self.encoder(x)
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1, 2)
        z, _ = self.gru(z)  # output size e.g. 8*128*256
        output = self.classifier(z)

        return output  # return every frame


class DCL(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: real embedding vector
        :param z2: predicted embedding vector
        :return: one-way loss
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        positive_loss = -torch.diag(sim_matrix)
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_mask = torch.eye(z1.size()[0], device=z1.device)
        negative_loss = torch.logsumexp(sim_matrix + neg_mask * SMALL_NUM, dim=0, keepdim=False)  # dim = 0 since the prediction = z2
        return (positive_loss + negative_loss).mean()


class CPC(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        writer_pos = './runs/' + self.args.store + '/'
        if self.args.transfer is True:
            writer_pos += '_ft'
        self.writer = SummaryWriter(writer_pos)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start CPC training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        acc = 0

        if self.args.resume:
            best_acc = self.args.best_acc
        else:
            best_acc = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            for sensor, _ in train_loader:
                sensor = sensor.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    acc, loss = self.model(sensor)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc', acc, global_step=n_iter)

                n_iter += 1

            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
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

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\t accuracy: {acc}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_end_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'acc': acc
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def transfer_train(self, tune_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        logging.info(f"Start CPC fine-tuning head for {self.args.epochs} epochs.")
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
                    logits = self.model.predict(sensor)
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
                val_acc = CPC_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)
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
        print('best eval acc is {}'.format(best_acc))

    def test_performance(self, best_model_dir, test_loader):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        test_acc = CPC_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=test_loader)
        print('test acc is {}'.format(test_acc))
        logging.info(f"test acc is {test_acc}.")
