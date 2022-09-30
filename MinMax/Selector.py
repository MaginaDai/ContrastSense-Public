import os
import logging
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from MinMax import IMU_Trans_MinMax
from judge import AverageMeter
from utils import save_config_file, save_checkpoint, accuracy, MoCo_evaluate


class DeepConvLSTM(nn.Module):

    def __init__(self, transfer=False, dims=32, aug_dim=9):
        super(DeepConvLSTM, self).__init__()

        self.transfer = transfer

        self.dropout = torch.nn.Dropout(p=0.1)
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()

        self.conv1 = torch.nn.Conv2d(1, dims, kernel_size=(25, 6), stride=1)
        self.conv2 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1)
        self.conv3 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1)

        self.gru1 = torch.nn.GRU(dims, int(dims / 2), num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = torch.nn.GRU(dims, 4, num_layers=1, batch_first=True, bidirectional=True)

        self.linear1 = torch.nn.Linear(in_features=352 * 2, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=aug_dim)

    def forward(self, x):
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        x = x.reshape(x.shape[0], -1, x.shape[1], x.shape[2])
        x = self.relu(self.dropout(self.conv1(x)))
        x = self.relu(self.dropout(self.conv2(x)))
        x = self.relu(self.dropout(self.conv3(x)))

        x = x.reshape(x.shape[0], x.shape[2], -1)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        x = x.reshape(x.shape[0], -1)

        x = self.relu(self.linear1(x))
        x = self.sig(self.linear2(x))
        return x


class Selector(nn.Module):

    def __init__(self, transfer=False):
        """

        """
        super(Selector, self).__init__()


class MoCoMinMax(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.selector = kwargs['selector'].to(self.args.device)

        self.optimizer1 = kwargs['optimizer1']
        self.scheduler1 = kwargs['scheduler1']

        self.optimizer2 = kwargs['optimizer2']
        self.scheduler2 = kwargs['scheduler2']

        self.cl_steps = 1
        self.selector_steps = 1

        writer_pos = './runs/' + self.args.store + '/'
        if self.args.transfer is True:
            writer_pos += '_ft'
        self.writer = SummaryWriter(writer_pos)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        self.imu_noise = IMU_Trans_MinMax.IMUNoise(var=0.05, p=0.8)
        self.imu_scale = IMU_Trans_MinMax.IMUScale(scale=[0.9, 1.1], p=0.8)
        self.imu_rotate = IMU_Trans_MinMax.IMURotate(p=0.8)
        self.imu_negate = IMU_Trans_MinMax.IMUNegated(p=0.4)
        self.imu_flip = IMU_Trans_MinMax.IMUHorizontalFlip(p=0.1)
        self.imu_warp = IMU_Trans_MinMax.IMUTimeWarp(p=0.4)
        self.imu_error_model = IMU_Trans_MinMax.IMUErrorModel(p=0.8, scale=[0.9, 1.1], error_magn=0.02, bias_magn=0.05)
        self.imu_multi_person = IMU_Trans_MinMax.IMUMultiPerson(p=0.4, scale=[0.8, 1.2])
        self.imu_malfunction = IMU_Trans_MinMax.IMUMalFunction(p=0.1, mal_length=25)
        self.imu_toTensor = IMU_Trans_MinMax.ToTensor()

    def get_pipeline_transform(self, data, choice):
        data = self.imu_scale(data, choice[0])
        data = self.imu_error_model(data, choice[1])
        data = self.imu_rotate(data, choice[2])
        data = self.imu_negate(data, choice[3])
        data = self.imu_flip(data, choice[4])
        data = self.imu_warp(data, choice[5])
        data = self.imu_multi_person(data, choice[6])
        data = self.imu_malfunction(data, choice[7])
        data = self.imu_noise(data, choice[8])
        data = self.imu_toTensor(data)
        return data

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start MoCo MinMax training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        acc = 0

        if self.args.resume:
            best_acc = self.args.best_acc
        else:
            best_acc = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            for sensor, _ in train_loader:
                sensor_a = torch.clone(sensor).to(self.args.device)
                for cl_index in range(self.cl_steps):
                    self.optimizer1.zero_grad()

                    sensor_b = list()
                    with autocast(enabled=self.args.fp16_precision):
                        choice = self.selector(sensor_a).detach()
                        choice = F.normalize(choice, dim=1)

                    sensor = sensor.numpy()
                    for idx in range(choice.shape[0]):
                        sensor_trans = self.get_pipeline_transform(sensor[idx], choice[idx])
                        sensor_b.append(sensor_trans)
                    sensor_b = torch.cat(sensor_b, dim=0).reshape(sensor.shape[0], sensor.shape[1], -1)
                    sensor_b = sensor_b.to(self.args.device)

                    with autocast(enabled=self.args.fp16_precision):
                        output1, target1 = self.model(sensor_a, sensor_b)
                        loss1 = self.criterion(output1, target1)

                    scaler.scale(loss1).backward()

                    scaler.step(self.optimizer1)
                    scaler.update()

                for selector_index in range(self.selector_steps):
                    self.optimizer2.zero_grad()

                    sensor_b = list()
                    with autocast(enabled=self.args.fp16_precision):
                        choice = self.selector(sensor_a)
                        choice = F.normalize(choice, dim=1)

                        for idx in range(choice.shape[0]):
                            sensor_trans = self.get_pipeline_transform(sensor[idx], choice[idx])
                            sensor_b.append(sensor_trans)
                        sensor_b = torch.cat(sensor_b, dim=0).reshape(sensor.shape[0], sensor.shape[1], -1)
                        sensor_b = sensor_b.to(self.args.device)

                        output2, target2 = self.model(sensor_a, sensor_b)
                        loss2 = -self.criterion(output2, target2)

                    scaler.scale(loss2).backward()

                    scaler.step(self.optimizer2)
                    scaler.update()

            if n_iter % self.args.log_every_n_steps == 0:
                acc = accuracy(output1, target1, topk=(1,))
                self.writer.add_scalar('loss1', loss1, global_step=n_iter)
                self.writer.add_scalar('loss2', loss2, global_step=n_iter)
                self.writer.add_scalar('acc', acc, global_step=n_iter)

            n_iter += 1

            is_best = acc > best_acc
            if epoch_counter >= 10:  # only after the first 10 epochs, the best_acc is updated.
                best_acc = max(acc, best_acc)
            if is_best:
                checkpoint_name = 'model_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'state_dict_selector': self.selector.state_dict(),
                    'best_acc': best_acc,
                    'optimizer1': self.optimizer1.state_dict(),
                    'optimizer2': self.optimizer2.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler1.step()
                self.scheduler2.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss1: {loss1}\tLoss2: {loss2}\t accuracy: {acc}")

        logging.info("Training has finished.")
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def transfer_train(self, tune_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        logging.info(f"Start MoCo fine-tuning head for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

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

                self.optimizer1.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer1)
                scaler.update()

                acc = accuracy(logits, target, topk=(1,))
                acc1.update(acc, sensor.size(0))
                if n_iter_train % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter_train)
                    self.writer.add_scalar('acc', acc, global_step=n_iter_train)
                    self.writer.add_scalar('learning_rate', self.scheduler1.get_last_lr()[0], global_step=n_iter_train)

                n_iter_train += 1

            if self.args.if_val:
                val_acc = MoCo_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)
                if self.args.if_fine_tune:
                    self.model.train()
            else:
                val_acc = 0

            is_best = val_acc > best_acc
            if epoch_counter >= 10:  # only after the first 10 epochs, the best_acc is updated.
                best_acc = max(val_acc, best_acc)
            if is_best:
                checkpoint_name = 'model_best.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer1': self.optimizer1.state_dict(),
                }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name), path_dir=self.writer.log_dir)

            self.writer.add_scalar('evaluation acc', val_acc, global_step=epoch_counter)
            self.scheduler1.step()
            logging.debug(f"Epoch: {epoch_counter} \t Loss: {loss} \t train acc: {acc1.avg} \t val acc: {val_acc}")

        logging.info("Fine-tuning has finished.")
        print('best eval acc is {}'.format(best_acc))

    def test_performance(self, best_model_dir, test_loader):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        test_acc = MoCo_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=test_loader)
        print('test acc is {}'.format(test_acc))
        logging.info(f"test acc is {test_acc}.")
