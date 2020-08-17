import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from reader import image_input_reader

from method.AdaBatch import *


def main():
    print("------------------------------------------------------------------------")
    print("This Tutorial is to train DensNet or WideResNet in tensorflow-gpu environment.")
    print("\nDescription -----------------------------------------------------------")
    print("Available algorithms: {Random Batch, Online Batch, Active Bias, Ada-Hard, Ada-Uniform, Ada-Boundary}")
    print("Supporting datasets: {Fashion-MNIST, CIFAR-10}. *** all the data sets will be added after acceptance. ***")
    print("For Training, we used the same training schedule:")
    print("\tbatch = 128, warm-up = 15")
    print("\tFor FMNIST: s_e = 32.0, learning rate = 0.01, epochs = 80")
    print("\tFor CIFAR-10, learning rate = 0.1 (decayed 50% and 75% of total number of epochs), epochs = 100")
    print("\tYou can change the parameters in main.py")
    print("------------------------------------------------------------------------")
    if len(sys.argv) != 8:
        print("------------------------------------------------------------------------")
        print("Run Cmd: python main.py  gpu_id  data  model_name  method_name  optimizer  log_dir")
        print("\nParamter description")
        print("gpu_id: gpu number which you want to use")
        print("data : {FMNIST, CIFAR-10}")
        print("model_name: {DenseNet-25-12, WideResNet16-8}")
        print("method_name: {Ada-Hard, Ada-Uniform, Ada-Boundary}")
        print("optimizer: {sgd, mementum}")
        print("weight sharing: {true, false}, If true, all the method share the same parames until 10 epoch (see Section 5.1 for details)")
        print("log_dir: log directory to save mini-batch loss/acc, training loss/acc and test loss/acc")
        print("------------------------------------------------------------------------")
        sys.exit(-1)

    # For user parameters
    gpu_id = int(sys.argv[1])
    data = sys.argv[2]
    model_name = sys.argv[3]
    method_name = sys.argv[4]
    optimizer = sys.argv[5]
    weight_sharing = sys.argv[6]
    log_dir = sys.argv[7]

    datapath = str(Path(os.path.dirname((os.path.abspath(__file__)))).parent) + "/dataset/" + data
    if os.path.exists(datapath):
        print("Dataset exists in ", datapath)
    else:
        print("Dataset doen't exist in ", datapath, ", please downloads and locates the data.")
        sys.exit(-1)

    # For fixed parameters (same as in the paper)
    if data == "FMNIST":
        total_epochs = 80
        s_e = 32.0
        lr_boundaries = [0]
        lr_values = [0.01, 0.01]
        input_reader = image_input_reader.ImageReader(data, datapath, 1, 60000, 10000, 28, 28, 1, 10)

    elif data == "CIFAR-10":
        total_epochs = 100
        s_e = 2.0
        lr_boundaries = [20000, 30000]
        lr_values = [0.1, 0.02, 0.004]
        input_reader = image_input_reader.ImageReader(data, datapath, 5, 50000, 10000, 32, 32, 3, 10)

    warm_up_period = 15
    batch_size = 128

    if weight_sharing == "true": # this is the same configuration we used in our paper
        pretrain = 10 # the epoch number to share the same weight for all methods, all the weights were saved in "init_weight" directory
    else:
        pretrain = 0

    if method_name == "Ada-Hard":
        AdaBatch(gpu_id, input_reader, model_name, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, 'hard', warm_up_period, pretrain=pretrain, log_dir=log_dir)
    elif method_name == "Ada-Uniform":
        AdaBatch(gpu_id, input_reader, model_name, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, 'uniform', warm_up_period, pretrain=pretrain, log_dir=log_dir)
    elif method_name == "Ada-Boundary":
        AdaBatch(gpu_id, input_reader, model_name, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, 'boundary', warm_up_period, s_e=s_e, pretrain=pretrain, log_dir=log_dir)

if __name__ == '__main__':
    print(sys.argv)
    main()
