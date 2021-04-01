## Ada-Boundary: Accelerating DNN Training via Adaptive Boundary Batch Selection

> __Publication__ </br>
> Song, H., Kim, S., Kim, M., and Lee, J., "Ada-Boundary: Accelerating DNN Training via Adaptive Boundary Batch Selection," *Machine Learning (ECML-PKDD Journal Track)*, Sep. 2020. [[Paper]](https://link.springer.com/article/10.1007/s10994-020-05903-6)

##  1. Requirement 
- Python 3
- tensorflow-gpu 2.1.0
- tensorpack libracy //use "pip install tensorpack"

##  2. Description
- We provide the training/evaluation of all compared algorithms in the paper. 
- Please do not change the structure of directories:
	- Folder **_src_** provides all the code for evaluation with compared methods.
	- Folder **_src/dataset_** contains benchmark datasets (FMNIST, CIFAR-10). Due to the lack of space, the other data will be uploaded soon. Moreover, **_.bin_** format is used for the synthetic data because they can be loaded at once in main memory.
### 3. Datasets and Weight Sharing
 - Datasets and Models (for weight sharing) can be downloaded in https://bit.ly/2YhrTsR
 - Please locate the two folder (**_dataset_**  and **_init_weight_** ) in the **_src_**  folder.

### 4. Tutorial for Evaluation.
- Training Configuration
	```python
	# All the hyperparameters of baseline methods were set to the same value described in our paper.
	# Source code provides a tutorial to train DensNet or WideResNet using a simple command.
	```
	
- Necessary Parameters
	```python
	- 'gpu_id': GPU number which you want to use (only support single gpu).
	- 'data_name': {MNIST, CIFAR-10}. # others will be supported later
	- 'model_name': {DenseNet-25-12, WideResNet16-8}
	- 'method_name': {Ada-Hard, Ada-Uniform, Ada-Boundary}.
	- 'optimizer': {sgd, momentum}
	- 'weight_sharing': {true, false} # if true, all the method share the same parameters 
	                                  # until 10 epoch (see Section 5.1 for details)
	- 'log_dir': log directory to save (1) mini-batch loss/error, (2) training loss/error,
	             and (3) test loss/error.
	```
- Running Command
	```python
	python main.py 'gpu_id' 'data_name' 'model_name' 'method_name' 'optimizer' 'weight_sharing' 'log_dir'
	
    # e.g., train on Fashion-MNIST using Ada-Boundary with weight sharing and sgd.
    # python main.py '0' 'MNIST' 'DenseNet-25-12' 'Ada-Boundary' 'sgd' 'true' 'log'
	```	
- Detail of Log File
	```python
	# convergence_log.csv
    # format: epoch, elapsed time, lr, mini-batch loss, mini-batch error, trainng loss, 
    #         training error, test loss, test error
	```	

