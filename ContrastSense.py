import os, time
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from SupContrast import SupConLoss

from attention.attention import MultiHeadedSelfAttention, LayerNorm, PositionWiseFeedForward
from utils import f1_cal, save_config_file, save_checkpoint, accuracy, ContrastSense_evaluate, AverageMeter
from sklearn.manifold import TSNE
# from kmeans_pytorch import kmeans
from sklearn.metrics import f1_score

class ContrastSense_model(nn.Module):
    def __init__(self, transfer=False, out_dim=256, classes=6, dims=32, classifier_dim=1024, final_dim=8, momentum=0.9, drop=0.1, modal='imu'):
        super(ContrastSense_model, self).__init__()

        self.modal = modal
        self.transfer = transfer

        if self.modal == 'imu':
            self.encoder = ContrastSense_encoder(dims=dims, momentum=momentum, drop=drop)
        elif self.modal == 'emg':
            self.encoder = ContrastSense_encoder_for_emg_v2(dims=dims, momentum=momentum, drop=drop)
        else:
            NotADirectoryError
        
        if transfer:
            self.classifier = ContrastSense_classifier(classes=classes, dims=dims, classifier_dim=classifier_dim, final_dim=final_dim, drop=drop, modal=self.modal)
        else:
            self.projector = ContrastSense_projector(out_dim=out_dim, modal=self.modal)

    def forward(self, x):
        h = self.encoder(x)
        if self.transfer:
            z = self.classifier(h)
        else:
            z = self.projector(h)
    
        return z
    

class ContrastSense_model_for_mobile(nn.Module):
    def __init__(self, transfer=False, out_dim=256, classes=6, dims=32, classifier_dim=64, final_dim=8, momentum=0.9, drop=0.1, modal='imu'):
        super(ContrastSense_model_for_mobile, self).__init__()
        self.transfer = transfer
        self.modal = modal
        if self.modal == 'imu':
            self.encoder = ContrastSense_encoder(dims=dims, momentum=momentum, drop=drop)
        elif self.modal == 'emg':
            self.encoder = ContrastSense_encoder_for_emg_v2(dims=dims, momentum=momentum, drop=drop)
        else:
            NotADirectoryError
        
        if transfer:
            self.classifier = ContrastSense_classifier(classes=classes, dims=dims, classifier_dim=classifier_dim, final_dim=final_dim, drop=drop, modal=self.modal)
        else:
            self.projector = ContrastSense_projector(out_dim=out_dim, modal=self.modal)
        

    def forward(self, x):
        h = self.encoder(x)
        if self.transfer:
            z = self.classifier(h)
        else:
            z = self.projector(h)
        return z


class ContrastSense_classifier(nn.Module):
    def __init__(self, classes=6, dims=32, classifier_dim=1024, final_dim=8, drop=0.1, modal='imu'):
        super(ContrastSense_classifier, self).__init__()

        if modal == 'imu':
            feature_num = 100 * dims
            self.gru = torch.nn.GRU(dims, final_dim, num_layers=1, batch_first=True, bidirectional=True)
        elif modal == 'emg':
            feature_num = dims * 32
            # feature_num = 832
            self.gru = torch.nn.GRU(16, final_dim, num_layers=1, batch_first=True, bidirectional=True)
        else:
            NotADirectoryError
    
        
        self.MLP = nn.Sequential(nn.Linear(in_features=feature_num, out_features=classifier_dim), # 1920 for 120
                                nn.ReLU(),
                                nn.Linear(in_features=classifier_dim, out_features=classes))
        self.dropout = torch.nn.Dropout(p=drop)
    
    def forward(self, h):
        self.gru.flatten_parameters()
        # h, _ = self.gru1(h)
        # h = self.dropout(h)
        h, _ = self.gru(h)
        h = self.dropout(h)
        h = h.reshape(h.shape[0], -1)
        h = self.MLP(h)
        return h


class ContrastSense_projector(nn.Module):
    def __init__(self, out_dim=512, modal='imu'):
        super(ContrastSense_projector, self).__init__()
        if modal == 'imu':
            feature_num = 6400
        elif modal == 'emg':
            feature_num = 1024
        else:
            NotADirectoryError

        self.linear1= torch.nn.Linear(in_features=feature_num, out_features=out_dim*4)  # 3840 for 120
        self.linear2 = torch.nn.Linear(in_features=out_dim*4, out_features=out_dim*2)
        self.linear3 = torch.nn.Linear(in_features=out_dim*2, out_features=out_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, h):
        # keep the output layer constant with the SimCLR output
        h = h.reshape(h.shape[0], -1)
        h = self.relu(self.linear1(h))
        h = self.relu(self.linear2(h))
        z = self.linear3(h)
        return z


class ContrastSense_encoder(nn.Module):
    # dims=32
    def __init__(self, dims=16, momentum=0.9, drop=0.1):
        super(ContrastSense_encoder, self).__init__()

        self.dropout = torch.nn.Dropout(p=drop)
        self.relu = torch.nn.ReLU()

        self.conv1_acc = torch.nn.Conv2d(1, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv1_gyro = torch.nn.Conv2d(1, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
    
        self.conv2_acc_1 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv2_acc_2 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        
        self.conv2_gyro_1 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv2_gyro_2 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        
        self.BN_acc = torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)
        self.BN_gyro = torch.nn.BatchNorm2d(num_features=dims, momentum=momentum, affine=False)

        self.conv3 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 3), stride=(1, 3), padding=(2, 0))
         
        self.conv4_1 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv4_2 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 1), stride=1, padding=(2, 0))
         
        self.conv5_1 = torch.nn.Conv2d(dims, dims, kernel_size=(5, 2), stride=1, padding=(2, 0))
         
        self.attn = MultiHeadedSelfAttention(dims)
        self.proj = nn.Linear(dims, dims)
        self.norm1 = LayerNorm(dims)
        self.pwff = PositionWiseFeedForward(hidden_dim=dims, hidden_ff=dims*2)
        self.norm2 = LayerNorm(dims)


    def forward(self, x):

        # extract in-sensor information
        x_acc = self.conv1_acc(x[:, :, :, 0:3])
        x_acc = self.dropout(self.relu(x_acc))

        x_gyro = self.conv1_gyro(x[:, :, :, 3:])
        x_gyro = self.dropout(self.relu(x_gyro))

        # ResNet Arch for high-level information
        x1 = self.conv2_acc_1(x_acc)
        x1 = self.dropout(self.relu(x1))
        x1 = self.conv2_acc_2(x1)
        x_acc = self.dropout(self.relu(x_acc + x1))
 

        x1 = self.conv2_gyro_1(x_gyro)
        x1 = self.dropout(self.relu(x1))
        x1 = self.conv2_gyro_2(x1) 
        x_gyro = self.dropout(self.relu(x_gyro + x1))
 
        
        # # we need to normalize the data to make the features comparable
        x_acc = self.BN_acc(x_acc)
        x_gyro = self.BN_gyro(x_gyro)

        h = torch.cat((x_acc, x_gyro), dim=3)
        # extract intra-sensor information
        h = self.conv3(h)
        h = self.dropout(self.relu(h))

        # ResNet Arch for high-level information
        x1 = self.conv4_1(h)
        x1 = self.dropout(self.relu(x1))
        x1 = self.conv4_2(x1)
        h = self.dropout(self.relu(h + x1))

        h = self.conv5_1(h)
        h = self.dropout(self.relu(h))
        
        h = h.view(h.shape[0], h.shape[1], -1)
        h = h.permute(0, 2, 1)
        
        h = self.attn(h)
        h = self.norm1(h + self.proj(h))
        h = self.norm2(h + self.pwff(h))
        h = self.dropout(h)

        return h
    
    

class ContrastSense_encoder_for_emg_v2(nn.Module):  
    ## keep align with lysseCoteAllard/MyoArmbandDataset
    ## at 62b886fc7014aeb81af65d77affedadf40de684c (github.com)
    def __init__(self, dims=32, momentum=0.9, drop=0.1):
        super(ContrastSense_encoder_for_emg_v2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, dims, kernel_size=(5, 3)),
            nn.BatchNorm2d(dims),
            nn.PReLU(dims),
            nn.Dropout2d(.5),
            nn.MaxPool2d(kernel_size=(3, 1)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dims, dims*2, kernel_size=(5, 3)),
            nn.BatchNorm2d(dims*2),
            nn.PReLU(dims*2),
            nn.Dropout2d(.5),
            nn.MaxPool2d(kernel_size=(3, 1)),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x.view(x.shape[0], x.shape[1], -1)  # to keep align with IMU encoder

    
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class ContrastSense_v1(nn.Module):
    """
    Build a model with: a query encoder, a key encoder, and a queue, 
    """
    def __init__(self, args, transfer=False, out_dim=256):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(ContrastSense_v1, self).__init__()

        device=args.device

        self.K = args.ContrastSense_K 
        self.m = args.ContrastSense_m
        self.T = args.temperature
        self.T_labels = args.tem_labels
        self.label_type = args.label_type

        self.modal = args.modal

        self.hard_sample = args.hard
        self.sample_ratio = args.sample_ratio
        self.last_ratio = args.last_ratio
        self.hard_record = args.hard_record
        self.time_window = args.time_window
        self.scale_ratio = args.scale_ratio 
        self.cross = args.cross
        

        # create the encoders
        # num_classes is the output fc dimension

        self.encoder_q = ContrastSense_model(transfer=transfer, modal=self.modal)
        self.encoder_k = ContrastSense_model(transfer=transfer, modal=self.modal)

        self.sup_loss = SupConLoss(device=device)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(out_dim, self.K))
        self.register_buffer("queue_dis", torch.randn(out_dim, self.K))
        self.register_buffer("queue_labels", torch.randint(0, 10, [self.K, 1]))  # store label for SupCon
        self.register_buffer("queue_activity_label", torch.randint(0, 10, [self.K, 1])) # store label for visualization
        self.register_buffer("queue_time_labels", torch.randint(0, 10, [self.K, 1]))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_dis_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_labels_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_activity_label_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_time_ptr", torch.zeros(1, dtype=torch.long))

        self.queue = F.normalize(self.queue, dim=0)
        self.queue_dis = F.normalize(self.queue_dis, dim=0)

        self.tsne = TSNE(n_components=2, learning_rate='auto', init='random')

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels=None, activity_label=None, time_label=None):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        assert self.K % batch_size == 0  # for simplicity

        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T  # modified
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

        if labels is not None:
            for i, l in enumerate(labels):
                ptr_label = int(self.queue_labels_ptr[i])
                self.queue_labels[ptr_label:ptr_label + batch_size, i] = l
                ptr_label = (ptr_label + batch_size) % self.K
                self.queue_labels_ptr[i] = ptr_label
        
        if activity_label is not None:
            ptr_activity_label = int(self.queue_activity_label_ptr)
            self.queue_activity_label[ptr_activity_label:ptr_activity_label + batch_size, 0] = activity_label
            ptr_activity_label = (ptr_activity_label + batch_size) % self.K
            self.queue_activity_label_ptr[0] = ptr_activity_label
        
        if time_label is not None:
            ptr_time = int(self.queue_time_ptr)
            self.queue_time_labels[ptr_time:ptr_time + batch_size, 0] = time_label[0]
            ptr_time = (ptr_time + batch_size) % self.K
            self.queue_time_ptr[0] = ptr_time

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    

    def forward(self, sen_q, sen_k, domain_label, activity_label, time_label):
        """
        Input:
            sen_q: a batch of query sensors data
            sen_k: a batch of key sensors data
            domain_label: the domain label 
            activity_label: the ground truth label for visualization
            time_label: the time label for each data sample
        Output:
            logits, targets
        """
        device = (torch.device('cuda')
            if sen_q.is_cuda
            else torch.device('cpu'))
        similarity_across_domains = 0
        
        hardest_related_info = [0, 0, 0, 0, 0]
        mean_same_class_ratio = 0
        mean_same_domain_ratio = 0
        mean_same_class_same_domain = 0

        start_time = time.time()
        q = self.encoder_q(sen_q)  # queries: NxC
        q = F.normalize(q, dim=1)
        end_time = time.time()
        hardest_related_info[4] = end_time - start_time
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(sen_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  ## we further improve this step

        if self.hard_record:
            sim_wih_other_domain = l_neg.clone().detach()
            _, indices = torch.sort(sim_wih_other_domain, dim=1, descending=True)
            num_eliminate = int(l_neg.shape[1] * self.sample_ratio)

            rows = torch.arange(l_neg.shape[0]).unsqueeze(-1)
            que_labels = self.queue_activity_label.T.expand(l_neg.shape[0], l_neg.shape[1])
            labels_of_eliminated = que_labels[rows, indices[:, :num_eliminate]]
            mask_same_class = torch.eq(activity_label[0].contiguous().view(-1, 1), labels_of_eliminated)
            same_class_ratio = mask_same_class.sum(dim=1)/num_eliminate * 100

            que_domain_labels = self.queue_labels.T.expand(l_neg.shape[0], l_neg.shape[1])
            domain_labels_of_eliminated = que_domain_labels[rows, indices[:, :num_eliminate]]
            mask_same_domain = torch.eq(domain_label[0].contiguous().view(-1, 1), domain_labels_of_eliminated)
            same_domain_ratio = mask_same_domain.sum(dim=1)/num_eliminate * 100
            same_class_same_domain = torch.logical_and(mask_same_class, mask_same_domain).sum(dim=1) / num_eliminate * 100
            
            mean_same_class_ratio = same_class_ratio.mean()
            mean_same_domain_ratio = same_domain_ratio.mean()
            mean_same_class_same_domain = same_class_same_domain.mean()
            hardest_related_info = [mean_same_class_ratio, mean_same_domain_ratio, mean_same_class_same_domain]
        
        # negative logits: NxK
        if self.hard_sample:
            #### domain-wise sorting + time window
            if self.last_ratio < 1.0:  ## if last_ratio >= 1, we don't apply this simplist elimination. 
                start_time = time.time()
                domains_in_queues = torch.unique(self.queue_labels.clone().detach()).contiguous().view(-1, 1)
                domain_queues_mask = torch.eq(domains_in_queues, self.queue_labels.T).bool().to(device)
                neg_for_sampling = l_neg.clone().detach()

                for j, domain_for_compare in enumerate(domains_in_queues):
                    key_in_domain_j = domain_queues_mask[j].repeat(neg_for_sampling.shape[0], 1)
                    domain_queue_j = neg_for_sampling[key_in_domain_j].view(neg_for_sampling.shape[0], -1)
                    _, indices = torch.sort(domain_queue_j, dim=1, descending=True)
                    idx_to_eliminate = indices[:, int(domain_queue_j.shape[1] * self.last_ratio):]
                    position = torch.where(domain_queues_mask[j] == True)[0].repeat(neg_for_sampling.shape[0], 1)

                    rows = torch.arange(neg_for_sampling.shape[0]).unsqueeze(-1)
                    masks_for_domain_j = torch.zeros(neg_for_sampling.shape).bool().to(device)
                    masks_for_domain_j[rows, position[rows, idx_to_eliminate]] = True
                    l_neg[masks_for_domain_j] = -torch.inf
                    hardest_related_info[1] += idx_to_eliminate.shape[1]
                
                hardest_related_info[1]  /= len(domains_in_queues)  ## record the avg number of eliminated samples
            
                end_time = time.time()
                hardest_related_info[2] = end_time - start_time

            if self.time_window != 0:
                start_time = time.time()
                low_boundary = time_label[0].contiguous().view(-1, 1) - self.time_window
                high_boundary = time_label[0].contiguous().view(-1, 1) + self.time_window
                queue_time = self.queue_time_labels.T.expand(l_neg.shape[0], l_neg.shape[1])
                mask_low = low_boundary < queue_time # (NxQ) = label of sen_q (Nx1) x labels of queue (Qx1).T
                mask_high = queue_time < high_boundary
                mask = torch.logical_and(mask_low, mask_high)
                    
                l_neg[mask] = -torch.inf    
                hardest_related_info[0] = mask.sum(1).float().mean()
                
                end_time = time.time()
                hardest_related_info[3] = end_time - start_time
        
        feature = torch.concat([q, self.queue.clone().detach().T], dim=0)  # the overall features rather than the dot product of features.  

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # targets: positive key indicators
        targets = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, domain_label, activity_label, time_label)
        logits_labels = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        return logits, targets, logits_labels, hardest_related_info, similarity_across_domains, feature
    
    
    def supervised_CL(self, logits_labels=None, labels=None):
        if labels and self.label_type:
            loss = torch.zeros(self.label_type)
            for i in range(self.label_type):
                loss[i] = self.sup_loss(logits_labels/self.T_labels[i], labels=labels[i], queue_labels=self.queue_labels[:, i].view(-1, 1))
            return loss
        else:
            return None


class ContrastSense(object):

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
        logging.info(f"Start ContrastSense training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        acc = 0
        best_epoch = 0
        best_acc = 0
        not_best_counter = 0
        best_loss = 1e6

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')
            loss_batch = AverageMeter('loss_batch', ':6.5f')
            mscr_batch = AverageMeter('mslr_batch', ':6.3f')
            msdr_batch = AverageMeter('msdr_batch', ':6.3f')
            mscsdr_batch = AverageMeter('mscsdr_batch', ':6.3f')

            start_time = time.time()
            for sensor, labels in train_loader:
                
                sensor = [t.to(self.args.device) for t in sensor]
                class_label = labels[:, 0].to(self.args.device) # the first dim is motion labels

                if self.args.label_type or self.args.hard:
                    time_label = [labels[:, -1].to(self.args.device)] # the last dim is time labels
                    if self.args.cross == 'users': # use domain labels
                        domain_label = [labels[:, 1].to(self.args.device)] 
                        
                    elif self.args.cross == 'positions' or self.args.cross == 'devices' :
                        domain_label = [labels[:, 2].to(self.args.device)] 
                    
                    elif self.args.cross == 'multiple':
                        domain_label = [labels[:, 3].to(self.args.device)]

                    elif self.args.cross == 'datasets':
                        domain_label = [labels[:, 4].to(self.args.device)]
                        
                    else:
                        NotADirectoryError
                else:
                    domain_label = None
                    time_label = None
                with autocast(enabled=self.args.fp16_precision):
                    output, target, logits_labels, hardest_related_info, similarity_across_domains, feature = self.model(sensor[0], sensor[1], 
                                                                                                                        domain_label=domain_label, 
                                                                                                                        activity_label=class_label, 
                                                                                                                        time_label=time_label)
                    sup_loss = self.model.supervised_CL(logits_labels=logits_labels, labels=domain_label)

                    loss = self.criterion(output, target)
                    ori_loss = loss.detach().clone()
                    if sup_loss is not None:
                        for i in range(len(sup_loss)):
                            loss -= self.args.slr[i] * sup_loss[i]

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                acc = accuracy(output, target, topk=(1,))

                acc_batch.update(acc, sensor[0].size(0))
                loss_batch.update(loss, sensor[0].size(0))

                mscr_batch.update(hardest_related_info[0], sensor[0].size(0))
                msdr_batch.update(hardest_related_info[1], sensor[0].size(0))
                mscsdr_batch.update(hardest_related_info[2], sensor[0].size(0))
                
                if n_iter % self.args.log_every_n_steps == 0 and n_iter != 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc', acc, global_step=n_iter)
                    self.writer.add_scalar('mslr', hardest_related_info[0], global_step=n_iter)
                    self.writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=n_iter)
                    if sup_loss is not None:
                        self.writer.add_scalar('ori_loss_{}'.format(i), ori_loss, global_step=n_iter)
                        for i in range(len(sup_loss)):
                            self.writer.add_scalar('sup_loss_{}'.format(i), sup_loss[i], global_step=n_iter)

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
            if self.args.hard:
                log_str += f"sim: {similarity_across_domains} mscr: {mscr_batch.avg} nes: {msdr_batch.avg}"
            if self.args.hard_record:
                log_str += f"mscr: {mscr_batch.avg} msdr: {msdr_batch.avg} mscsdr: {mscsdr_batch.avg}"
            if sup_loss is not None:
                log_str += f"ori_Loss :{ori_loss} "
                for i in range(len(sup_loss)):
                    log_str += f"sup_loss_{i}: {sup_loss[i]} "
            # if cluster_eval is not None:
            #     log_str += f"chs: {chs.avg} center_shift: {center_shift}"
            logging.debug(log_str)

            # if best_acc > 99 and epoch_counter >= 50:
            #     print(f"early stop at {epoch_counter}")
            #     break  # early stop

            if not_best_counter >= 200:
                print(f"early stop at {epoch_counter}")
                break
            
        
        logging.info("Training has finished.")
        logging.info(f"Model of Epoch {best_epoch} checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def test_performance(self, best_model_dir, test_loader):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict)
        test_acc, test_f1 = ContrastSense_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=test_loader, test=True)
        logging.info(f"test f1 is {test_f1}.")
        logging.info(f"test acc is {test_acc}.")

        print('test f1 is {} for {}'.format(test_f1, self.args.name))
        print('test acc is {} for {}'.format(test_acc, self.args.name))
    
    def test_performance_cross_dataset(self, best_model_dir, test_dataloader_for_all_datasets, datasets):
        checkpoint = torch.load(best_model_dir, map_location="cpu")
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        for i, test_loader in enumerate(test_dataloader_for_all_datasets):
            test_acc, test_f1 = ContrastSense_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=test_loader)

            logging.info(f"test f1 is {test_f1} for {datasets[i]}.")
            logging.info(f"test acc is {test_acc} for {datasets[i]}.")

            print('test f1 is {} for {}'.format(test_f1, datasets[i]))
            print('test acc is {} for {}'.format(test_acc, datasets[i]))

    
    def transfer_train_penalty(self, tune_loader, val_loader, fisher):
        assert self.args.if_fine_tune is True

        self.fisher = fisher
        self.mean = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter_train = 0
        logging.info(f"Start ContrastSense fine-tuning head for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        acc = 0
        f1 = 0
        best_epoch = 0

        if self.args.resume:
            best_f1 = self.args.best_f1
            best_acc = self.args.best_acc
        else:
            best_f1 = 0
            best_acc = 0

        for epoch_counter in tqdm(range(self.args.epochs)):
            acc_batch = AverageMeter('acc_batch', ':6.2f')

            pred_batch = torch.empty(0).to(self.args.device)
            label_batch = torch.empty(0).to(self.args.device)

            # f1_batch = AverageMeter('f1_batch', ':6.2f')
            if self.args.if_fine_tune:
                self.model.train()
            else:  
                self.model.eval()
                self.model.classifier.train()
                
            for sensor, target in tune_loader:
                sensor = sensor.to(self.args.device)
                target = target[:, 0].to(self.args.device)
                label_batch = torch.cat((label_batch, target))

                with autocast(enabled=self.args.fp16_precision):
                    logits = self.model(sensor)
                    loss_clf = self.criterion(logits, target)

                    loss_penalty = self.compute_penalty()

                    loss = loss_clf + self.args.penalty_lambda * loss_penalty

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                    
                acc = accuracy(logits, target, topk=(1,))
                _, pred = logits.topk(1, 1, True, True)
                pred_batch = torch.cat((pred_batch, pred.reshape(-1)))
                
                f1 = f1_cal(logits, target, topk=(1,))
                acc_batch.update(acc, sensor.size(0))
                if n_iter_train % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter_train)
                    self.writer.add_scalar('loss_clf', loss_clf, global_step=n_iter_train)
                    self.writer.add_scalar('loss_penalty', loss_penalty, global_step=n_iter_train)
                    self.writer.add_scalar('acc', acc, global_step=n_iter_train)
                    self.writer.add_scalar('f1', f1, global_step=n_iter_train)
                    self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step=n_iter_train)

                n_iter_train += 1


            val_acc, val_f1 = ContrastSense_evaluate(model=self.model, criterion=self.criterion, args=self.args, data_loader=val_loader)

            is_best = val_f1 > best_f1

            if epoch_counter >= 10:  # only after the first 10 epochs, the best_f1/acc is updated.
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
            self.scheduler.step()

            f1_batch = f1_score(label_batch.cpu().numpy(), pred_batch.cpu().numpy(), average='macro') * 100
            logging.debug(f"Epoch: {epoch_counter} Loss: {loss} Loss_penalty: {loss_penalty} acc: {acc_batch.avg: .3f}/{val_acc: .3f} f1: {f1_batch: .3f}/{val_f1: .3f}")


        logging.info("Fine-tuning has finished.")
        logging.info(f"best eval f1 is {best_f1} at {best_epoch}.")

        print('best eval f1 is {} for {}'.format(best_f1, self.args.name))

    def compute_penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher.keys():
                loss += (
                    torch.sum(
                        (self.fisher[n])
                        * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                    )
                    / 2
                )
        return loss