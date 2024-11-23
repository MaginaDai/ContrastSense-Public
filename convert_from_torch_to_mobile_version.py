import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from ContrastSense import ContrastSense_model_for_mobile
from data_preprocessing.preprocessing import ClassesNum, PositionNum, UsersNum
from thop import profile

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
    # print(input_tensor.shape)

    if method == "ContrastSense":
        model = ContrastSense_model_for_mobile(transfer=True, classes=ClassesNum[name], modal=modal)
    else:
        NotADirectoryError

    flops, params = profile(model, inputs=(input_tensor,))
    # model = torch.quantization.convert(model)
    print(f'GFLOPs: {flops / 1e9}')
    model.eval()
    traced_script_module = torch.jit.trace(model, input_tensor)
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    optimized_traced_model._save_for_lite_interpreter(f"mobile_models/{method}_medium_{modal}.ptl")


if __name__ == "__main__":
    generate_mobile_model(modal="imu", method="ContrastSense")