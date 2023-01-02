import os
import sys
import logging
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))

from MoCo import GradientReversal
from baseline.MMD.mmd_loss import MMD_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from judge import AverageMeter
from utils import f1_cal, save_config_file, accuracy, save_checkpoint, evaluate, split_last, merge_last, CPC_evaluate


class FM_model(nn.Module):
    def __init__(self, method, classes, domains=None):
        super(FM_model, self).__init__()
        self.encoder = FM_Encoder()
        self.head = FM_Classifier(classes=classes)
        self.method = method
        if self.method == 'CM':
            self.discriminator = FM_discriminator(domains=domains)
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.head(h)
        return z

    def predict_domain(self, x):
        h = self.encoder(x)
        d = self.discriminator(h)
        return d


class FM_Encoder(nn.Module):
    def __init__(self):
        super(FM_Encoder, self).__init__()

        self.leakyRelu = nn.LeakyReLU(0.3)

        self.dropout = nn.Dropout()
        
        self.conv1 = nn.Conv1d(6, 16, kernel_size=3, stride=1)
        self.norm1 = nn.InstanceNorm1d(16)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, stride=1)
        self.norm2 = nn.InstanceNorm1d(16)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=5, stride=4)
        self.norm3 = nn.InstanceNorm1d(32)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, stride=1)
        self.norm4 = nn.InstanceNorm1d(32)
        self.conv5 = nn.Conv1d(32, 64, kernel_size=5, stride=4)
        self.norm5 = nn.InstanceNorm1d(64)
        self.conv6 = nn.Conv1d(64, 100, kernel_size=5, stride=1)
        self.norm6 = nn.InstanceNorm1d(100)

        
        
    
    def forward(self, x):
        x = x.permute(0, 2, 1)

        h = self.dropout(self.norm1(self.leakyRelu(self.conv1(x))))
        h = self.dropout(self.norm2(self.leakyRelu(self.conv2(h))))
        h = self.dropout(self.norm3(self.leakyRelu(self.conv3(h))))
        h = self.dropout(self.norm4(self.leakyRelu(self.conv4(h))))
        h = self.dropout(self.norm5(self.leakyRelu(self.conv5(h))))
        h = self.dropout(self.norm6(self.leakyRelu(self.conv6(h))))

        h = h.reshape(h.shape[0], h.shape[1], -1)
        h = F.max_pool1d(h, kernel_size=h.size()[-1]) 
        h = h.reshape(h.shape[0], -1)

        return h


class FM_Classifier(nn.Module):
    def __init__(self, classes):
        super(FM_Classifier, self).__init__()
        self.linear1 = nn.Linear(100, classes)
        
    
    def forward(self, h):
        y = self.linear1(h)
        return y
    
class FM_discriminator(nn.Module):
    def __init__(self, domains):
        super(FM_discriminator, self).__init__()
        self.reverse_layer = GradientReversal()
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, domains)
    
    def forward(self, h):
        h = self.reverse_layer(h)
        h = self.linear1(h)
        y = self.linear2(h)
        return y


class FMUDA(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        writer_pos = './runs/' + self.args.store + '/'
        writer_pos += self.args.name
        if self.args.shot:
            writer_pos += f'_shot_{self.args.shot}'
        else:
            writer_pos += f'_percent_{self.args.percent}'
        self.writer = SummaryWriter(writer_pos)

        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        
        self.clf_loss = torch.nn.CrossEntropyLoss().to(self.args.device)
        if self.args.method == 'FM':
            self.mmd_loss = MMD_loss()
        else:
            self.domain_loss = torch.nn.CrossEntropyLoss().to(self.args.device)

    def train(self, train_loader, tune_loader, val_loader, test_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        logging.info(f"Start {self.args.method}UDA training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        acc = 0
        f1 = 0
        best_epoch = 0
        best_acc = 0
        best_f1 = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')
            
            pred_batch = torch.empty(0).to(self.args.device)
            label_batch = torch.empty(0).to(self.args.device)

            loss_clf_batch = AverageMeter('loss_clf_batch', ':6.5f')
            loss_uda_batch = AverageMeter('loss_uda_batch', ':6.5f')

            data_loader = zip(tune_loader, train_loader)

            for (sensor, target), (sensor_domain, target_domain) in data_loader:
                sensor = sensor.to(self.args.device)
                target = target[:, 0].to(self.args.device)
                
                sensor_domain = sensor_domain.to(self.args.device)
                target_domain = target_domain[:, 1].to(self.args.device)
                
                with autocast(enabled=self.args.fp16_precision):
                    logits = self.model(sensor)
                    loss_clf = self.clf_loss(logits, target)
                
                self.optimizer.zero_grad()
                scaler.scale(loss_clf).backward()
                scaler.step(self.optimizer)
                scaler.update()

                
                if self.args.method == 'FM':
                    with autocast(enabled=self.args.fp16_precision):
                        feature = self.model.encoder(sensor_domain)
                        loss_uda = self.mmd_loss(feature, target_domain)
                else:
                    with autocast(enabled=self.args.fp16_precision):
                        logits_domain = self.model.predict_domain(sensor_domain)
                        loss_uda = self.domain_loss(logits_domain, target_domain)

                self.optimizer.zero_grad()
                scaler.scale(loss_uda).backward()
                scaler.step(self.optimizer)
                scaler.update()

                acc = accuracy(logits, target, topk=(1,))
                acc_batch.update(acc, sensor.size(0))

                label_batch = torch.cat((label_batch, target))
                _, pred = logits.topk(1, 1, True, True)
                pred_batch = torch.cat((pred_batch, pred.reshape(-1)))

                loss_clf_batch.update(loss_clf, sensor.size(0))
                loss_uda_batch.update(loss_uda, sensor.size(0))

            self.writer.add_scalar('loss_clf', loss_clf_batch.avg, global_step=epoch_counter)
            self.writer.add_scalar('loss_uda', loss_uda_batch.avg, global_step=epoch_counter)
            self.writer.add_scalar('acc', acc_batch.avg, global_step=epoch_counter)
            self.writer.add_scalar('f1', f1_batch, global_step=epoch_counter)

            f1_batch = f1_score(label_batch.cpu().numpy(), pred_batch.cpu().numpy(), average='macro') * 100
            val_acc, val_f1 = evaluate(model=self.model, criterion=self.clf_loss, args=self.args, data_loader=val_loader)

            is_best = val_f1 > best_f1
        
            best_f1 = max(val_f1, best_f1)
            best_acc = max(val_acc, best_acc)

            if is_best:
                best_epoch = epoch_counter
                checkpoint_name = 'model_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter,
                    'state_dict': self.model.state_dict(),
                    'best_f1': best_f1,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)

            self.writer.add_scalar('eval acc', val_acc, global_step=epoch_counter)
            self.writer.add_scalar('eval f1', val_f1, global_step=epoch_counter)
            logging.debug(f"Epoch: {epoch_counter} Loss_clf: {loss_clf_batch.avg} Loss_uda: {loss_uda_batch.avg} acc: {acc_batch.avg: .3f}/{val_acc: .3f} f1: {f1_batch.avg: .3f}/{val_f1: .3f}")
            
        logging.info("Fine-tuning has finished.")
        logging.info(f"best eval f1 is {best_f1} at {best_epoch}.")

        print('best eval f1 is {} for {}'.format(best_f1, self.args.name))
    
    def test_performance(self, best_model_dir, test_loader):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        test_acc, test_f1 = evaluate(model=self.model, criterion=self.clf_loss, args=self.args, data_loader=test_loader)
        logging.info(f"test f1 is {test_f1}.")
        logging.info(f"test acc is {test_acc}.")
        
        print('test f1 is {} for {}'.format(test_f1, self.args.name))
        print('test acc is {} for {}'.format(test_acc, self.args.name))