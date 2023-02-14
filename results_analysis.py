import argparse
from shutil import ReadError
from sys import implementation
import numpy as np
import re
import pdb

from xml.sax import default_parser_list


dataset = ['HASC', 'HHAR', 'MotionSense', 'Shoaib']

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')
parser.add_argument('-name', default=["test", "test"], nargs='+', type=str, help='the interested models file')
parser.add_argument('-ft', default=False, type=bool, help='fine-tune or linear evaluation')
parser.add_argument('-nad', default='ft_shot_10', type=str, help='name after datasets')
parser.add_argument('-shot', default=10, type=int, help='how many shots we use')

def avg_result(name, ft, shot=10):
    eval = np.zeros([len(name), len(dataset)])
    test = np.zeros([len(name), len(dataset)])
    test_acc = np.zeros([len(name), len(dataset)])
    for i, n in enumerate(name):
        for j, data in enumerate(dataset):
            print(n, j)
            if ft:
                dir = f'{n}/{data}_ft_shot_{shot}/training.log'
            else:
                dir = f'{n}/{data}_shot_{shot}/training.log'
            eval_pattern = r'best\seval\sf1\sis\s+\(*(\d+\.+\d*)'
            test_pattern = r'test\sf1\sis\s+\(*(\d+\.+\d*)'
            test_acc_pattern = r'test\sacc\sis\s+\(*(\d+\.+\d*)'
            with open(dir, "r") as f:  # 打开文件
                content = f.read()
                # pdb.set_trace()
                eval_extract = np.float64(re.findall(eval_pattern, content))
                test_extract = np.float64(re.findall(test_pattern, content))      
                acc_extract = np.float64(re.findall(test_acc_pattern, content))
                if len(eval_extract) > 1 or len(test_extract) > 1:
                    # pdb.set_trace()
                    raise ReadError
                eval[i, j] = eval_extract[0]
                test[i, j] = test_extract[0]
                if len(acc_extract) == 0:
                    test_acc[i, j] = 0
                else:
                    test_acc[i, j] = acc_extract[0]
                
    eval_mean = np.expand_dims(np.mean(eval, axis=1), 1)
    test_mean = np.expand_dims(np.mean(test, axis=1), 1)
    acc_mean = np.expand_dims(np.mean(test_acc, axis=1), 1)
    eval_final = np.concatenate((eval, eval_mean), axis=1)
    test_final = np.concatenate((test, test_mean), axis=1)
    test_acc_final = np.concatenate((test_acc, acc_mean), axis=1)
    
    print("Eval f1 is: \n {}".format(np.around(eval_final, 2)))
    print("Test f1 is: \n {}".format(np.around(test_final, 2)))
    print("Test acc is: \n {}".format(np.around(test_acc_final, 2)))
    print("Eval mean is: {}".format(np.around(np.mean(eval_final, axis=0), 2)))
    print("Test mean is: {}".format(np.around(np.mean(test_final, axis=0), 2)))
    print("Test acc is: {}".format(np.around(np.mean(test_acc_final, axis=0), 2)))
    print("Test std is: {}".format(np.around(np.std(test_mean), 2)))
    return np.mean(test_final, axis=0)

def results_for_each_file(files):
    eval = np.zeros([len(files)])
    test = np.zeros([len(files)])
    acc = np.zeros([len(files)])
    for i, n in enumerate(files):
        seg = n.split('_')
        dataset = seg[-1]
        name = dataset + '_ft_shot_10'
        dir = f'{n}/{name}/training.log'

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
    print(f"eval f1 {np.around(eval, decimals=2)}")
    print(f"test f1 {np.around(test, decimals=2)}")
    # print(f"{np.around(acc, decimals=2)}")
    # print(f"{eval}\n{test}\n{acc}")
    return

def transfer_ability_access(name, ft):
    eval = np.zeros([len(name), len(dataset)])
    test = np.zeros([len(name), len(dataset)])
    for i, n in enumerate(name):
        for j, data in enumerate(dataset):
            # print(n, j)
            seg = n.split('_')
            dataset_name = seg[-1]
            if data == dataset_name:
                continue
            if ft:
                dir = f'{n}/{data}_ft_shot_10/training.log'
            else:
                dir = f'{n}/{data}_le_shot_10/training.log'
            eval_pattern = r'best\seval\sf1\sis\s+\(*(\d+\.+\d*)'
            test_pattern = r'test\sf1\sis\s+\(*(\d+\.+\d*)'
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
    
    eval_mean = np.expand_dims(np.sum(eval, axis=1)/3, 1)
    test_mean = np.expand_dims(np.sum(test, axis=1)/3, 1)
    print("Eval acc is: \n {}".format(np.around(np.concatenate((eval, eval_mean), axis=1), 2)))
    print("Test acc is: \n {}".format(np.around(np.concatenate((test, test_mean), axis=1), 2)))
    return 


shots=[10, 50, 200, 500, 'full']
def avg_result_for_limited_labels(name):
    eval = np.zeros([len(name), len(shots)])
    test = np.zeros([len(name), len(shots)])
    prefix = "GILE"
    lafix = "version"
    for i, n in enumerate(name):
        for j, data in enumerate(shots):
            print(n, data)
            if data=='full':
                dir = f'baseline/GILE/runs/{prefix}_{data}_{lafix}_{n}/HHAR/training.log'
            else:
                dir = f'baseline/GILE/runs/{prefix}_shot{data}_{lafix}_{n}/HHAR/training.log'
            eval_pattern = r'best\seval\sf1\sis\s+\(*(\d+\.+\d*)'
            test_pattern = r'test\sf1\sis\s+\(*(\d+\.+\d*)'
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
    print("Eval f1 is: \n {}".format(np.around(eval, 2)))
    print("Test f1 is: \n {}".format(np.around(test, 2)))
    print("Eval mean is: {}".format(np.around(np.mean(eval, axis=0), 2)))
    print("Test mean is: {}".format(np.around(np.mean(test, axis=0), 2)))
    return
    
def avg_result_for_cross_domains(name, ft):
    eval = np.zeros([len(name)])
    test = np.zeros([len(name)])
    data="Shoaib"
    for i, n in enumerate(name):
        # print(n)
        if ft:
            dir = f'{n}/{data}_shot_10/training.log'
        else:
            dir = f'{n}/{data}_shot_10/training.log'
        eval_pattern = r'best\seval\sf1\sis\s+\(*(\d+\.+\d*)'
        test_pattern = r'test\sf1\sis\s+\(*(\d+\.+\d*)'
        test_acc_pattern = r'test\sacc\sis\s+\(*(\d+\.+\d*)'
        with open(dir, "r") as f:  # 打开文件
            content = f.read()
            # pdb.set_trace()
            eval_extract = np.float64(re.findall(eval_pattern, content))
            test_extract = np.float64(re.findall(test_pattern, content))
            if len(eval_extract) > 1 or len(test_extract) > 1:
                # pdb.set_trace()
                raise ReadError
            eval[i] = eval_extract[0]
            test[i] = test_extract[0]
    eval_mean = np.expand_dims(np.mean(eval), axis=0)
    test_mean = np.expand_dims(np.mean(test), axis=0)
    eval_final = np.concatenate((eval, eval_mean))
    test_final = np.concatenate((test, test_mean))
    print("Eval f1 is: {}".format(np.around(eval_final, 2)))
    print("Test f1 is: {}".format(np.around(test_final, 2)))
    print("Test std is: {}".format(np.around(np.std(test), 2)))
    return

if __name__ == '__main__':
    args = parser.parse_args()
    avg_result(args.name, ft=True, shot=args.shot)
    # avg_result_for_limited_labels(args.name)
    # results_for_each_file(args.name)
    # avg_result(name=args.name, ft=args.ft)
    # transfer_ability_access(args.name, ft=True)

    # avg_result_for_cross_domains(args.name, ft=True)