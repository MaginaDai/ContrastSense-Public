import argparse
from shutil import ReadError
from sys import implementation
import numpy as np
import re
import pdb

from xml.sax import default_parser_list


dataset = ['HHAR', 'MotionSense', 'Shoaib', 'UCI']

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')
parser.add_argument('-name', default=["test", "test"], nargs='+', type=str, help='the interested models file')
parser.add_argument('-ft', default=False, type=bool, help='fine-tune or linear evaluation')

def avg_result(name, ft):
    eval = np.zeros([len(name), len(dataset)])
    test = np.zeros([len(name), len(dataset)])
    for i, n in enumerate(name):
        for j, data in enumerate(dataset):
            # print(i, j)
            if ft:
                dir = f'runs/{n}/{data}_ft/training.log'
            else:
                dir = f'runs/{n}/{data}_le/training.log'
            eval_pattern = r'best\seval\sacc\sis\s+tensor\(\[*(\d+\.+\d*)\]'
            test_pattern = r'test\sacc\sis\s+tensor\(\[*(\d+\.+\d*)\]'
            with open(dir, "r") as f:  # 打开文件
                content = f.read()
                # pdb.set_trace()
                eval_extract = np.float64(re.findall(eval_pattern, content))
                test_extract = np.float64(re.findall(test_pattern, content))
                if len(eval_extract) > 1 or len(test_extract) > 1:
                    # pdb.set_trace()
                    raise ReadError
                eval[i, j] = eval_extract[0]
                test[i, j] = test_extract[0]
    eval_mean = np.expand_dims(np.mean(eval, axis=1), 1)
    test_mean = np.expand_dims(np.mean(test, axis=1), 1)
    print("Eval acc is: \n {}".format(np.around(np.concatenate((eval, eval_mean), axis=1), 2)))
    print("Test acc is: \n {}".format(np.around(np.concatenate((test, test_mean), axis=1), 2)))
    return 

def results_for_each_file(files):
    eval = np.zeros([len(files)])
    test = np.zeros([len(files)])
    acc = np.zeros([len(files)])
    for i, n in enumerate(files):
        seg = n.split('_')
        dataset = seg[-1]
        name = dataset + '_ft_shot_10'
        dir = f'runs/{n}/{name}/training.log'

        eval_pattern = r'best\seval\sf1\sis\s+\(*(\d+\.+\d*)'
        test_pattern = r'test\sf1\sis\s+\(*(\d+\.+\d*)'
        test_acc_pattern = r'test\sacc\sis\s+\(*(\d+\.+\d*)'

        with open(dir, "r") as f:  # 打开文件
                content = f.read()
                # pdb.set_trace()
                eval_extract = np.float64(re.findall(eval_pattern, content))
                test_extract = np.float64(re.findall(test_pattern, content))
                test_acc_extract = np.float64(re.findall(test_acc_pattern, content))
                if len(eval_extract) > 1 or len(test_extract) > 1:
                    # pdb.set_trace()
                    raise ReadError
                eval[i] = eval_extract[0]
                test[i] = test_extract[0]
                acc[i] = test_acc_extract[0]
        
        f.close()
    
    eval = np.append(eval, np.mean(eval))
    test = np.append(test, np.mean(test))
    acc = np.append(acc, np.mean(acc))
    print(f"{np.around(eval, decimals=2)}")
    # print("test")
    print(f"{np.around(test, decimals=2)}")
    # print(f"{np.around(acc, decimals=2)}")
    # print(f"{eval}\n{test}\n{acc}")
    return

if __name__ == '__main__':
    args = parser.parse_args()
    results_for_each_file(args.name)
    # avg_result(name=args.name, ft=args.ft)