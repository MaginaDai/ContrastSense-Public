import torch, time, sys
from baseline.CALDA.model import CALDA_encoder
from DeepSense import DeepSense_model
from MoCo import MoCo_model
from baseline.Mixup.ConvNet import ConvNet
from data_aug.preprocessing import ClassesNum, UsersNum
from torchsummary import summary
from torchstat import stat
from thop import profile

modal='imu'

def cost_evaluate():
    if modal == 'emg':
        name='NinaPro'
        input_size = (1, 52, 8)
    elif modal == 'imu': 
        name='HHAR'
        input_size = (1, 200, 6)
    else:
        NotADirectoryError
    
    model = MoCo_model(transfer=True, classes=ClassesNum[name], modal=modal).to('cuda')

    flops, params = profile(model, inputs=(torch.randn((1, 1, 200, 6)).to('cuda'), ))
    # model = DeepSense_model(classes=ClassesNum[name]).to('cuda')
    # model = CALDA_encoder(num_classes=ClassesNum[name], 
    #                       num_domains=UsersNum[name],
    #                       num_units=128,
    #                       modal=modal).to('cuda')
    # model = ConvNet(number_of_class=ClassesNum[name]).to('cuda')
    print({flops / 1e9})
    summary(model, input_size=input_size, batch_size=-1)

    get_model_parameter_amount(model)
    # get_model_inference_time(model)
    # get_model_memory_usage(model)
    return

def get_model_parameter_amount(model):
    print(f'The model size is: {count_parameters(model)} parameters')
    return


def get_model_inference_time(model):
    if modal == 'emg':
        name='NinaPro'
        input_tensor = torch.randn(1, 1, 52, 8).to('cuda')
    elif modal == 'imu': 
        name='HHAR'
        input_tensor = torch.randn(1, 1, 200, 6).to('cuda')
    else:
        NotADirectoryError

    start_time = time.time()

    with torch.no_grad():
        output = model(input_tensor)

    end_time = time.time()
    print(f'Inference time: {end_time - start_time} seconds')
    return

def get_model_memory_usage(model):
    torch.save(model.state_dict(), "temp.p")
    # size_MB = sys.getsizeof("temp.p") / 1e6
    # print(f'The model memory usage is: {get_model_memory_usage(model)} MB')
    # return size_MB

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary():
    summary()


if __name__ == '__main__':
    cost_evaluate()