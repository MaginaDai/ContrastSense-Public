import argparse
import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from baseline.CDA.model import STCN
from baseline.CALDA.model import CALDA_encoder
from baseline.Mixup.ConvNet import ConvNet
from baseline.LIMU_BERT.models import BERTClassifier, fetch_classifier
from baseline.ClusterHAR.TPN_model import Encoder
from baseline.MMD.FMUDA import FM_model
from MoCo import MoCo_model, MoCo_model_for_mobile
from data_aug.preprocessing import ClassesNum, PositionNum, UsersNum
from baseline.CPCHAR.CPC import Transfer_Coder
from baseline.GILE.model_GILE import GILE_for_mobile
from utils import handle_argv, load_bert_classifier_data_config

def generate_mobile_model(modal, method):
    if modal == 'emg':
        name='NinaPro'
        input_size = (1, 52, 8)
    elif modal == 'imu': 
        name='HHAR'
        input_size = (1, 200, 6)
    else:
        NotADirectoryError

    if modal == 'emg':
        name='NinaPro'
        if method in ["FM", "CM"]:
            input_tensor = torch.randn(1, 52, 8)
        else:
            input_tensor = torch.randn(1, 1, 52, 8)
    elif modal == 'imu': 
        name='HHAR'
        if method in ["FM", "CM", "CPCHAR", "LIMU_BERT"]:
            input_tensor = torch.randn(1, 200, 6)
        elif method == 'GILE':
            input_tensor = torch.randn(1, 200, 6, 1)
        else:
            input_tensor = torch.randn(1, 1, 200, 6)
    else:
        NotADirectoryError

    if method == "ContrastSense":
        model = MoCo_model_for_mobile(transfer=True, classes=ClassesNum[name], modal=modal)
    elif method == "FM" or method == "CM":
        model = FM_model(classes=ClassesNum[name], method=method, domains=9)
    elif method == "GILE":
        args = GILE_args(name)
        args.name = name
        args.n_domains = UsersNum[args.name]
        args.n_class = ClassesNum[args.name]
        args.device = "cpu"
        model = GILE_for_mobile(args)
    elif method == "ClusterHAR":
        model = Encoder('Cluster')
    elif method == "CPCHAR":
        args = CPCHAR_args()
        args.padding = int(args.kernel_size // 2)
        args.input_size = 6
        args.transfer = True
        args.name=name
        model = Transfer_Coder(classes=ClassesNum[args.name], args=args)
    elif method == "LIMU_BERT":
        args = handle_argv('bert_classifier_' + "base_gru", 'bert_classifier_train.json', "base_gru")
        train_cfg, model_bert_cfg, model_classifier_cfg = load_bert_classifier_data_config(args)
        classifier = fetch_classifier("base_gru", model_classifier_cfg, input=model_bert_cfg.hidden, dataset_name=args.name)
        model = BERTClassifier(model_bert_cfg, classifier=classifier, frozen_bert=False)
    elif method == "Mixup":
        if name == 'Myo' or name == 'NinaPro':
            model = ConvNet(number_of_class=ClassesNum[name])
        else:
            # from baseline.ClusterHAR.TPN_model import Transfer_Coder
            model = Transfer_Coder(classes=ClassesNum[name], method='CL')

    elif method == "CALDA":
        model = CALDA_encoder(num_classes=ClassesNum[name], 
                          num_domains=UsersNum[name],
                          num_units=128,
                          modal=modal)
    elif method == "ConSSL":
        model = STCN(num_class=ClassesNum[name], transfer=True)
        
    # model = torch.quantization.convert(model)
    model.eval()
    traced_script_module = torch.jit.trace(model, input_tensor)
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    optimized_traced_model._save_for_lite_interpreter(f"mobile_models/{method}_{modal}.ptl")



def GILE_args(name):
    parser = argparse.ArgumentParser(description='argument setting of network')

    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('-version', default="shot", type=str, help='control the version of the setting')
    parser.add_argument('--store', default='lr', type=str, help='define the name head for model storing')
    parser.add_argument('-name', default='HHAR', help='datasets name', choices=['HHAR', 'MotionSense', 'UCI', 'Shoaib', 'HASC', 'ICHAR', "Merged_dataset"])
    parser.add_argument('-percent', default=1, type=float, help='how much percent of labels to use')
    parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')

    parser.add_argument('--now_model_name', type=str, default='GILE', help='the type of model')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
    parser.add_argument('--n_epoch', type=int, default=150, help='number of training epochs')
    parser.add_argument('-g', '--gpu-index', default=0, type=int, help='Gpu index.')

    parser.add_argument('--n_feature', type=int, default=6, help='name of feature dimension')
    parser.add_argument('--len_sw', type=int, default=128, help='length of sliding window')
    parser.add_argument('--d_AE', type=int, default=50, help='dim of AE')
    parser.add_argument('--sigma', type=float, default=1, help='parameter of mmd')
    parser.add_argument('--weight_mmd', type=float, default=1.0, help='weight of mmd loss')

    parser.add_argument('--test_every', type=int, default=1, help='do testing every n epochs')
    parser.add_argument('-n_target_domains', type=int, default=1, help='number of target domains')

    parser.add_argument('--beta', type=float, default=1., help='multiplier for KL')

    parser.add_argument('--x-dim', type=int, default=1152, help='input size after flattening')
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=1000., help='multiplier for y classifier')
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=1000., help='multiplier for d classifier')

    parser.add_argument('--beta_d', type=float, default=1., help='multiplier for KL d')
    parser.add_argument('--beta_x', type=float, default=0., help='multiplier for KL x')
    parser.add_argument('--beta_y', type=float, default=1., help='multiplier for KL y')

    parser.add_argument('--weight_true', type=float, default=1000.0, help='weights for classifier true')
    parser.add_argument('--weight_false', type=float, default=1000.0, help='weights for classifier false')
    parser.add_argument('--setting', default='sparse', type=str, choices=['full', 'sparse'], help='decide use tune or others')
    parser.add_argument('-cross', default='positions', type=str, help='decide to use which kind of labels')

    return parser.parse_args()


def CPCHAR_args():
    parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')
    parser.add_argument('-lr', '--learning-rate', default=5e-4, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--transfer', default=True, type=str, help='to tell whether we are doing transfer learning')
    parser.add_argument('--pretrained', default='CPC_HHAR', type=str, help='path to pretrained checkpoint')
    parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
    parser.add_argument('-name', default='HHAR', help='datasets name', choices=['HHAR', 'MotionSense', 'UCI', 'Shoaib', 'HASC', 'ICHAR', "Merged_dataset"])
    parser.add_argument('--log-every-n-steps', default=5, type=int, help='Log every n steps')
    parser.add_argument('-g', '--gpu-index', default=3, type=int, help='Gpu index.')
    parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Size of the conv filters in the encoder')

    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-j', '--workers', default=5, type=int, metavar='N', help='number of data loading workers (default: 5)')
    parser.add_argument('--store', default=None, type=str, help='define the name head for model storing')
    parser.add_argument('-e', '--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-percent', default=1, type=float, help='how much percent of labels to use')
    parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')

    parser.add_argument('--evaluate', default=False, type=bool, help='decide whether to evaluate')
    parser.add_argument('-ft', '--if-fine-tune', default=False, type=bool, help='to decide whether tune all the layers')
    parser.add_argument('-version', default="shot", type=str, help='control the version of the setting')

    return parser.parse_args()


if __name__ == "__main__":
    generate_mobile_model(modal="emg", method="ContrastSense")