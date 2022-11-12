import sys
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))

from baseline.CLHAR.CL_dataload import CLHARDataset4Training
from baseline.CPCHAR.dataload import ToTensor
from torchvision.transforms import transforms


class COCOA_Dataset:
    def __init__(self, transfer, version, datasets_name=None):
        self.transfer = transfer
        self.datasets_name = datasets_name
        self.version = version

    def get_dataset(self, split, percent=20, shot=None):
        return CLHARDataset4Training(self.datasets_name, self.version,
                                     transform=transforms.Compose([ToTensor()]),  # can add more transformations
                                     split=split, transfer=self.transfer, percent=percent, shot=shot)