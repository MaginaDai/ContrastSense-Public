import torch
import torch.nn as nn
import os
import copy
import sys
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from SACL import SAAdversarialLoss, SAContrastiveAdversarialLoss, momentum_model_parameter_update
import pickle as pkl

class SACLResBlock(torch.nn.Module):
    """
    see appendix Figure A.1 in arxiv.org/pdf/2007.04871.pdf for diagram
    """
    def __init__(self, num_channels_in, num_channels_out, kernel_size, dropout_rate=0.5):
        super(SACLResBlock, self).__init__()
        self.batch_norm_1 = torch.nn.BatchNorm1d(num_channels_in, track_running_stats=False)
        self.elu_1 = torch.nn.ELU()
        self.conv1d_residual = torch.nn.Conv1d(num_channels_in, num_channels_out, 1)
        self.conv1d_1 = torch.nn.Conv1d(num_channels_in, num_channels_out, kernel_size, padding=kernel_size-1)
        self.batch_norm_2 = torch.nn.BatchNorm1d(num_channels_out, track_running_stats=False)
        self.elu_2 = torch.nn.ELU()
        self.conv1d_2 = torch.nn.Conv1d(num_channels_out, num_channels_out, kernel_size)
        pass
    
    def forward(self, x):
        # print("\nSACLResBlock: x.shape == ", x.shape)
        x = self.batch_norm_1(x)
        # print("SACLResBlock: x.shape == ", x.shape)
        x = self.elu_1(x)
        # print("SACLResBlock: x.shape == ", x.shape)
        x_resid = self.conv1d_residual(x)
        # print("SACLResBlock: x_resid.shape == ", x_resid.shape)
        x = self.conv1d_1(x)
        # print("SACLResBlock: x.shape == ", x.shape)
        x = self.batch_norm_2(x)
        # print("SACLResBlock: x.shape == ", x.shape)
        x = self.elu_2(x)
        # print("SACLResBlock: x.shape == ", x.shape)
        x = self.conv1d_2(x)
        # print("SACLResBlock: x.shape == ", x.shape)
        out = x + x_resid
        # print("SACLResBlock: out.shape == ", out.shape)
        return out

class SACLFlatten(torch.nn.Module):
    """
    see https://stackoverflow.com/questions/53953460/how-to-flatten-input-in-nn-sequential-in-pytorch
    """
    def __init__(self):
        super(SACLFlatten, self).__init__()
        pass
    
    def forward(self, x):
        return x.view(x.size(0), -1)

class SACLEncoder(torch.nn.Module):
    """
    NOTE: IF YOU ARE GETTING SHAPE ERRORS, CHANGE THE SHAPE OF THE FINAL LINEAR LAYER IN THE EMBEDDER FIRST (BY CHANGING THE INTEGER-DIVISION DENOMINATOR)
    see  appendix Figure A.2 in arxiv.org/pdf/2007.04871.pdf for diagram
    """
    def __init__(self, num_channels, temporal_len, dropout_rate=0.5, embed_dim=256):
        super(SACLEncoder, self).__init__()
        self.sequential_process = torch.nn.Sequential(torch.nn.Conv1d(num_channels, num_channels//2, temporal_len//32), 
                                               SACLResBlock(num_channels//2, num_channels//2, temporal_len//16), 
                                               torch.nn.MaxPool1d(4),  
                                               SACLResBlock(num_channels//2, num_channels, temporal_len//16), 
                                               torch.nn.MaxPool1d(4),  
                                               SACLResBlock(num_channels, num_channels*2, temporal_len//32), 
                                               torch.nn.ELU(), 
                                               SACLFlatten(), # see https://stackoverflow.com/questions/53953460/how-to-flatten-input-in-nn-sequential-in-pytorch
                                               torch.nn.Linear(num_channels*(2**1)*(int(temporal_len/16.5)), embed_dim) # added to make it easier for different data sets with different shapes to be run through model
        )

        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        pass
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.sequential_process(x)
        return out

class SACLNet(torch.nn.Module):
    """
    NOTE: IF YOU ARE GETTING SHAPE ERRORS, CHANGE THE SHAPE OF THE FINAL LINEAR LAYER IN THE EMBEDDER FIRST (BY CHANGING THE INTEGER-DIVISION DENOMINATOR)
    see  appendix Figure A.2 in arxiv.org/pdf/2007.04871.pdf for diagram
    """
    def __init__(self, num_channels, temporal_len, dropout_rate=0.5, embed_dim=256, num_upstream_decode_features=64):
        super(SACLNet, self).__init__()
        self.embed_model = SACLEncoder(num_channels, temporal_len, dropout_rate=0.5, embed_dim=embed_dim)

        self.decode_model = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim//2), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(embed_dim//2, embed_dim//2), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(embed_dim//2, embed_dim//2), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(embed_dim//2, num_upstream_decode_features)
        )

        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        pass
    
    def forward(self, x):
        # print("\nSACLNet: x.shape == ", x.shape, "\n")
        # x = x.permute(0, 2, 1)
        x_embedded = self.embed_model(x)
        # print("\nSACLNet: x_embedded.shape == ", x_embedded.shape, "\n")
        out = self.decode_model(x_embedded)
        # print("\nSACLNet: out.shape == ", out.shape, "\n")
        return out


class SACLAdversary(torch.nn.Module):
    """
    see  Figure 1 in arxiv.org/pdf/2007.04871.pdf for diagram
    """
    def __init__(self, embed_dim, num_subjects, dropout_rate=0.5):
        super(SACLAdversary, self).__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim//2), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(embed_dim//2, embed_dim//2), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(embed_dim//2, embed_dim//2), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(embed_dim//2, num_subjects), 
                                                torch.nn.Sigmoid() # ADDED BY ZAC TO ADDRESS NANs IN ADVERSARIAL LOSS
        )
        pass
    
    def forward(self, x):
        return self.model(x)
    

class SACL_model(nn.Module):
    def __init__(self, device, num_subjects=15, channels=62, temporal_len=200, dropout_rate=0.5, embed_dim=256, num_upstream_decode_features=64):
        super(SACL_model, self).__init__()
        
        self.model = SACLNet(channels, temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim, num_upstream_decode_features=num_upstream_decode_features)
        self.momentum_model = copy.deepcopy(self.model) # see https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492 and https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/
    
        self.adversary = SACLAdversary(embed_dim, num_subjects, dropout_rate=dropout_rate).to(device)

    def forward(self, x_q, x_k):
        x1_rep = self.model(x_q)
        x2_rep = self.momentum_model(x_k)
        x1_embeds = self.model.embed_model(x_q)
        x1_subject_preds = self.adversary(x1_embeds)
        return x1_rep, x2_rep, x1_embeds, x1_subject_preds
    
    def adversarial_forward(self, x):
        x_t1_initial_reps = self.model.embed_model(x)
        x_t1_initial_subject_preds = self.adversary(x_t1_initial_reps)
        return x_t1_initial_subject_preds
    

class SACL_ft_model(nn.Module):
    def __init__(self, transfer=True, num_class=7):
        super(SACL_ft_model, self).__init__()
        # self.BN_0 = nn.BatchNorm1d(62)
        self.embed_model = SACLEncoder(62, 200, dropout_rate=0.5, embed_dim=256)
        self.BN = nn.BatchNorm1d(62)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_class))

    def forward(self, x):
        # x = self.BN_0(x)
        x = x.permute(0, 2, 1)
        x = self.BN(x)
        x = x.permute(0, 2, 1)
        x = self.embed_model(x)
        
        h = self.classifier(x)
        return h


def train_SA_model(save_dir_for_model, model_file_name="final_SA_model.bin", batch_size=256, shuffle=True, # hyper parameters for training loop
                    max_epochs=100, learning_rate=5e-4, beta_vals=(0.9, 0.999), weight_decay=0.001, #num_workers=4, 
                    max_evals_after_saving=6, save_freq=20, former_state_dict_file=None, ct_dim=None, h_dim=None, 
                    channels=11, temporal_len=3000, dropout_rate=0.5, embed_dim=100, encoder_type=None, bw=5, # hyper parameters for SA Model
                    randomized_augmentation=False, num_upstream_decode_features=32, temperature=0.05, NUM_AUGMENTATIONS=2, perturb_orig_signal=True, former_adversary_state_dict_file=None, adversarial_weighting_factor=1., momentum=0.999, # hyper parameters for SA Model
                    cached_datasets_list_dir=None, total_points_val=2000, tpos_val=None, tneg_val=None, window_size=3, #hyper parameters for data loaders
                    sfreq=1000, Nc=None, Np=None, Nb=None, max_Nb_iters=None, total_points_factor=None, 
                    windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                    windowed_start_time_name="_Windowed_StartTime.npy", data_folder_name="Mouse_Training_Data", 
                    data_root_name="Windowed_Data", file_names_list="training_names.txt", train_portion=0.7, 
                    val_portion=0.2, test_portion=0.1, random_seed=0):
    
    # First, load the training, validation, and test sets
    train_set, val_set, test_set = load_SSL_Dataset_Based_On_Subjects('SA', 
                                                    cached_datasets_list_dir=cached_datasets_list_dir, 
                                                    total_points_val=total_points_val, 
                                                    tpos_val=tpos_val, 
                                                    tneg_val=tneg_val, 
                                                    window_size=window_size, 
                                                    sfreq=sfreq, 
                                                    Nc=Nc, 
                                                    Np=Np, 
                                                    Nb=Nb, # this used to be 2 not 4, but 4 would work better
                                                    max_Nb_iters=max_Nb_iters, 
                                                    total_points_factor=total_points_factor, 
                                                    bw=bw,                                              # items for SA data loading
                                                    randomized_augmentation=randomized_augmentation,    # items for SA data loading
                                                    num_channels=channels,                              # items for SA data loading
                                                    temporal_len=temporal_len,                          # items for SA data loading
                                                    NUM_AUGMENTATIONS=NUM_AUGMENTATIONS,                # items for SA data loading
                                                    perturb_orig_signal=perturb_orig_signal,            # items for SA data loading
                                                    windowed_data_name=windowed_data_name,
                                                    windowed_start_time_name=windowed_start_time_name,
                                                    data_folder_name=data_folder_name, 
                                                    data_root_name=data_root_name, 
                                                    file_names_list=file_names_list, 
                                                    train_portion=train_portion, 
                                                    val_portion=val_portion, 
                                                    test_portion=test_portion, 
                                                    random_seed=random_seed
    )

    # initialize data loaders for training
    train_loader = torch.utils.data.DataLoader(train_set, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle#, num_workers=num_workers # see https://www.programmersought.com/article/93393550792/
    )
    val_loader = torch.utils.data.DataLoader(val_set, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle#, num_workers=num_workers
    )

    # print("train_SA_model: len of the train_loader is ", len(train_loader))

    # cuda setup if allowed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # initialize models - see Figure 1 of arxiv.org/pdf/2007.04871.pdf
    model = SACLNet(channels, temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim, num_upstream_decode_features=num_upstream_decode_features)
    momentum_model = copy.deepcopy(model) # see https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492 and https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/
    model = model.to(device)
    momentum_model = momentum_model.to(device)

    _, _, y0 = next(iter(train_loader))
    assert len(y0.shape) == 2 
    num_subjects = y0.shape[1]
    adversary = SACLAdversary(embed_dim, num_subjects, dropout_rate=dropout_rate).to(device)
    if former_adversary_state_dict_file is not None:
        adversary.load_state_dict(torch.load(former_adversary_state_dict_file))

    print("train_SA_model: START OF TRAINING")
    # initialize training state
    min_val_inaccuracy = float("inf")
    min_state = None
    num_evaluations_since_model_saved = 0
    saved_model = None
    saved_momentum_model = None
    loss_fn = SAContrastiveAdversarialLoss(temperature, adversarial_weighting_factor=adversarial_weighting_factor)
    # learning_rate = learning_rate
    # beta_vals = beta_vals
    optimizer = torch.optim.Adam(model.parameters(), betas=beta_vals, lr=learning_rate, weight_decay=weight_decay)

    saved_adversary = None
    adversarial_loss_fn = SAAdversarialLoss()
    adversarial_optimizer = torch.optim.Adam(adversary.parameters(), betas=beta_vals, lr=learning_rate, weight_decay=weight_decay)

    # Iterate over epochs
    avg_train_losses = []
    avg_train_accs = []
    avg_val_accs = []
    avg_adversary_train_losses = []
    avg_adversary_train_accs = []
    avg_adversary_val_accs = []
    for epoch in range(max_epochs):
        # print("train_SA_model: epoch ", epoch, " of ", max_epochs)

        model.train()
        momentum_model.train()
        adversary.train()

        running_train_loss = 0
        num_correct_train_preds = 0
        total_num_train_preds = 0
        running_adversary_train_loss = 0
        num_adversary_correct_train_preds = 0
        total_num_adversary_train_preds = 0
        
        # iterate over training batches
        # print("train_SA_model: \tNow performing training updates")
        counter = 0
        for x_t1, x_t2, y in train_loader:
            x_t1, x_t2, y = x_t1.to(device), x_t2.to(device), y.to(device)
            for p in model.parameters():
                p.requires_grad = False
            for p in momentum_model.parameters():
                p.requires_grad = False
            for p in adversary.parameters():
                p.requires_grad = True
            
            adversarial_optimizer.zero_grad()
            
            x_t1_initial_reps = model.embed_model(x_t1)
            x_t1_initial_subject_preds = adversary(x_t1_initial_reps)

            adversarial_loss = adversarial_loss_fn(x_t1_initial_subject_preds, y)
            num_adversary_correct_train_preds += adversarial_loss_fn.get_number_of_correct_preds(x_t1_initial_subject_preds, y)
            total_num_adversary_train_preds += len(x_t1_initial_subject_preds)

            adversarial_loss.backward()
            adversarial_optimizer.step()

            running_adversary_train_loss += adversarial_loss.item()

            # UPDATE MODEL - references Algorithm 1 of arxiv.org/pdf/1911.05722.pdf and Figure 1 of arxiv.org/pdf/2007.04871.pdf
            for p in model.parameters():
                p.requires_grad = True
            for p in momentum_model.parameters():
                p.requires_grad = False
            for p in adversary.parameters():
                p.requires_grad = False

            # zero out any pre-existing gradients
            optimizer.zero_grad()
            x1_rep = model(x_t1)
            x2_rep = momentum_model(x_t2)
            x1_embeds = model.embed_model(x_t1)
            x1_subject_preds = adversary(x1_embeds)
            loss = loss_fn(x1_rep, x2_rep, x1_subject_preds, y)
            # print("loss == ", loss)

            num_correct_train_preds += loss_fn.get_number_of_correct_reps(x1_rep, x2_rep, x1_subject_preds, y)
            total_num_train_preds += len(x1_rep)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

            momentum_model = momentum_model_parameter_update(momentum, momentum_model, model)
            counter += 1
        
        # iterate over validation batches
        # print("train_SA_model: \tNow performing validation")
        num_correct_val_preds = 0
        total_num_val_preds = 0
        num_correct_adversarial_val_preds = 0
        total_num_adversarial_val_preds = 0
        with torch.no_grad():
            model.eval()
            momentum_model.eval()
            adversary.eval()

            for x_t1, x_t2, y in val_loader:
                x_t1, x_t2, y = x_t1.to(device), x_t2.to(device), y.to(device)

                # evaluate model and adversary
                x1_rep = model(x_t1)
                x2_rep = momentum_model(x_t2)
                x1_embeds = model.embed_model(x_t1)
                x1_subject_preds = adversary(x1_embeds)
                # x1_subject_preds = adversary(x1_rep)

                num_correct_val_preds += loss_fn.get_number_of_correct_reps(x1_rep, x2_rep, x1_subject_preds, y)
                total_num_val_preds += len(x1_rep)

                num_correct_adversarial_val_preds += adversarial_loss_fn.get_number_of_correct_preds(x1_subject_preds, y)
                total_num_adversarial_val_preds += len(x1_subject_preds)
        
        # record averages
        avg_train_accs.append(num_correct_train_preds / total_num_train_preds)
        avg_val_accs.append(num_correct_val_preds / total_num_val_preds)
        avg_train_losses.append(running_train_loss / len(train_loader))
        
        avg_adversary_train_accs.append(num_adversary_correct_train_preds / total_num_adversary_train_preds)
        avg_adversary_val_accs.append(num_correct_adversarial_val_preds / total_num_adversarial_val_preds)
        avg_adversary_train_losses.append(running_adversary_train_loss / len(train_loader))
        
        # check stopping criterion / save model
        incorrect_val_percentage = 1. - (num_correct_val_preds / total_num_val_preds)
        if incorrect_val_percentage < min_val_inaccuracy:
            num_evaluations_since_model_saved = 0
            min_val_inaccuracy = incorrect_val_percentage
            saved_model = model.state_dict()
            saved_momentum_model = momentum_model.state_dict()
            saved_adversary = adversary.state_dict()
        else:
            num_evaluations_since_model_saved += 1
            if num_evaluations_since_model_saved >= max_evals_after_saving:
                print("train_SA_model: EARLY STOPPING on epoch ", epoch)
                break

    print("train_SA_model: END OF TRAINING - now saving final model / other info")

    # save final model(s)
    model.load_state_dict(saved_model)
    model_save_path = os.path.join(save_dir_for_model, model_file_name)
    torch.save(model.state_dict(), model_save_path)

    momentum_model.load_state_dict(saved_momentum_model)
    model_save_path = os.path.join(save_dir_for_model, "momentum_model_"+model_file_name)
    torch.save(model.state_dict(), model_save_path)

    adversary.load_state_dict(saved_adversary)
    model_save_path = os.path.join(save_dir_for_model, "adversary_"+model_file_name)
    torch.save(model.state_dict(), model_save_path)

    embedder_save_path = os.path.join(save_dir_for_model, "embedder_"+model_file_name)
    torch.save(model.embed_model.state_dict(), embedder_save_path)

    meta_data_save_path = os.path.join(save_dir_for_model, "meta_data_and_hyper_parameters.pkl")
    with open(meta_data_save_path, 'wb') as outfile:
        pkl.dump({
            "avg_train_losses": avg_train_losses, 
            "avg_train_accs": avg_train_accs, 
            "avg_val_accs": avg_val_accs, 
            "avg_adversary_train_losses": avg_adversary_train_losses, 
            "avg_adversary_train_accs": avg_adversary_train_accs, 
            "avg_adversary_val_accs": avg_adversary_val_accs, 
            "save_dir_for_model": save_dir_for_model, 
            "model_file_name": model_file_name, 
            "batch_size": batch_size, 
            "shuffle": shuffle, #"num_workers": num_workers, 
            "max_epochs": max_epochs, 
            "learning_rate": learning_rate, 
            "beta_vals": beta_vals, 
            "weight_decay": weight_decay, 
            "max_evals_after_saving": max_evals_after_saving, 
            "save_freq": save_freq, 
            "former_state_dict_file": former_state_dict_file, 
            "ct_dim": ct_dim, 
            "h_dim": h_dim, 
            "channels": channels, 
            "temporal_len": temporal_len, 
            "dropout_rate": dropout_rate, 
            "embed_dim": embed_dim,
            "encoder_type": encoder_type, 
            "bw": bw, 
            "randomized_augmentation": randomized_augmentation, 
            "num_upstream_decode_features": num_upstream_decode_features, 
            "temperature": temperature, 
            "NUM_AUGMENTATIONS": NUM_AUGMENTATIONS, 
            "perturb_orig_signal": perturb_orig_signal, 
            "former_adversary_state_dict_file": former_adversary_state_dict_file, 
            "adversarial_weighting_factor": adversarial_weighting_factor, 
            "momentum": momentum, 
            "cached_datasets_list_dir": cached_datasets_list_dir, 
            "total_points_val": total_points_val, 
            "tpos_val": tpos_val, 
            "tneg_val": tneg_val, 
            "window_size": window_size,
            "sfreq": sfreq, 
            "Nc": Nc, 
            "Np": Np, 
            "Nb": Nb,
            "max_Nb_iters": max_Nb_iters, 
            "total_points_factor": total_points_factor, 
            "windowed_data_name": windowed_data_name,
            "windowed_start_time_name": windowed_start_time_name,
            "data_folder_name": data_folder_name, 
            "data_root_name": data_root_name, 
            "file_names_list": file_names_list, 
            "train_portion": train_portion, 
            "val_portion": val_portion, 
            "test_portion": test_portion, 
            "random_seed": random_seed, 
        }, outfile)

    print("train_SA_model: DONE!")
    pass
