import torch
from ContrastSense import ContrastSense_v1
import numpy as np
import torch.multiprocessing
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy('file_system')


def getPenalty(args, train_loader, save_dir):
    model = ContrastSense_v1(args)
        
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)

    fisher, fisher_infoNCE = calculateFisher(args, model, optimizer, train_loader, save_dir)
    
    for n, p in fisher.items():
        fisher[n] = torch.min(fisher[n], torch.tensor(args.fishermax)).to('cpu')

    for n, p in fisher_infoNCE.items():
        fisher_infoNCE[n] = torch.min(fisher_infoNCE[n], torch.tensor(args.fishermax)).to('cpu')
    
    fisher_dir = save_dir + 'fisher.npz'
    np.savez(fisher_dir, fisher=fisher, fisher_infoNCE=fisher_infoNCE)
    return

def replenish_queue(model, train_loader, args):
    # as we use the queue to enlarge the negative samples size, 
    # we need to re-fresh the queue with real samples, rather than using randomly generated samples.

    model.eval()
    with torch.no_grad():
        for sensor, labels in train_loader:
            sensor = [t.to(args.device) for t in sensor]
            if args.label_type:
                if args.cross == 'users': # use domain labels
                    sup_label = [labels[:, 1].to(args.device)] 
                elif args.cross == 'positions' or args.cross == 'devices' :
                    sup_label = [labels[:, 2].to(args.device)] 
                elif args.cross == 'multiple':
                    sup_label = [labels[:, 3].to(args.device)]
                elif args.cross == 'datasets':
                    sup_label = [labels[:, 4].to(args.device)]
                else:
                    NotADirectoryError
            
            sen_q = sensor[0]
            sen_k = sensor[1]
            
            q = model.encoder_q(sen_q)  # queries: NxC
            q = F.normalize(q, dim=1)

            k = model.encoder_k(sen_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            model._dequeue_and_enqueue(k, sup_label)

    return model

def calculateFisher(args, model, optimizer, train_loader, save_dir):

    model_dir = save_dir + '/model_best.pth.tar'
    checkpoint = torch.load(model_dir, map_location="cpu")
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])

    model.to(args.device)

    model = replenish_queue(model, train_loader, args)

    model.train()

    fisher_cdl = {
        n[len("encoder_q."):]: torch.zeros(p.shape).to(args.device)
        for n, p in model.named_parameters()
            if p.requires_grad and n.startswith('encoder_q.encoder')
    }
    fisher_infoNCE = {
        n[len("encoder_q."):]: torch.zeros(p.shape).to(args.device)
        for n, p in model.named_parameters()
            if p.requires_grad and n.startswith('encoder_q.encoder')
    }
    
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    with torch.cuda.device(args.gpu_index):
        optimizer.zero_grad()
        for sensor, labels in train_loader:
            sensor = [t.to(args.device) for t in sensor]
            activity_label = labels[:, 0].to(args.device) # the first dim is motion labels
            if args.label_type:
                if args.cross == 'users': # use domain labels
                    domain_label = [labels[:, 1].to(args.device)] 
                elif args.cross == 'positions' or args.cross == 'devices' :
                    domain_label = [labels[:, 2].to(args.device)] 
                elif args.cross == 'multiple':
                    domain_label = [labels[:, 3].to(args.device)]
                elif args.cross == 'datasets':
                    domain_label = [labels[:, 4].to(args.device)]
                else:
                    NotADirectoryError
                    
            time_label = labels[:, -1].to(args.device) # the last dim is time labels
            _, _, logits_labels, _, _, _ = model(sensor[0], sensor[1], domain_label=domain_label, activity_label=activity_label, time_label=time_label)
            sup_loss = model.supervised_CL(logits_labels=logits_labels, labels=domain_label)
            loss = - args.slr[0] * sup_loss
            loss /= len(train_loader)
            loss.backward() 

        for n, p in model.named_parameters():
            if p.grad is not None and n.startswith('encoder_q.encoder'):
                fisher_cdl[n[len("encoder_q."):]] += p.grad.pow(2).clone()
            
        optimizer.zero_grad()
        for sensor, labels in train_loader:
            sensor = [t.to(args.device) for t in sensor]
            activity_label = labels[:, 0].to(args.device) # the first dim is motion labels
            if args.label_type or args.hard:
                time_label = [labels[:, -1].to(args.device)] # the last dim is time labels
                if args.cross == 'users': # use domain labels
                    domain_label = [labels[:, 1].to(args.device)] 
                elif args.cross == 'positions' or args.cross == 'devices' :
                    domain_label = [labels[:, 2].to(args.device)] 
                elif args.cross == 'multiple':
                    domain_label = [labels[:, 3].to(args.device)]
                elif args.cross == 'datasets':
                    domain_label = [labels[:, 4].to(args.device)]

                else:
                    NotADirectoryError
            
            output, target,  _, _, _, _ = model(sensor[0], sensor[1], domain_label=domain_label, activity_label=activity_label, time_label=time_label)
            loss = criterion(output, target)
            loss /= len(train_loader)
            loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None and n.startswith('encoder_q.encoder'):
                fisher_infoNCE[n[len("encoder_q."):]] += p.grad.pow(2).clone()
        
    model.to('cpu')

    return fisher_cdl, fisher_infoNCE


def load_fisher_matrix(pretrain_dir, device):
    fisher_dir = './runs/' + pretrain_dir + '/fisher.npz'
    fisher = np.load(fisher_dir, allow_pickle=True)
    fisher_cdl = fisher['fisher'].tolist()
    fisher_infoNCE = fisher['fisher_infoNCE'].tolist()
    for n, _ in fisher_cdl.items():
        fisher_cdl[n] = fisher_cdl[n].to(device)
        fisher_infoNCE[n] = fisher_infoNCE[n].to(device)
    return fisher_cdl, fisher_infoNCE
