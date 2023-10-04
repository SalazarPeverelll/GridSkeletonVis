import torch
import sys
import os
import time
import json
sys.path.append("/home/yiming/ContrastDebugger")
from singleVis.data import NormalDataProvider

CONTENT_PATH = "/home/yiming/ContrastDebugger/EXP/cifar10"
with open('/home/yiming/ContrastDebugger/EXP/cifar10/config.json', 'r') as f:
    config = json.load(f)

# record output information
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
sys.stdout = open(os.path.join(CONTENT_PATH, now+".txt"), "w")

SETTING = config["SETTING"]
CLASSES = config["CLASSES"]
DATASET = config["DATASET"]
PREPROCESS = config["VISUALIZATION"]["PREPROCESS"]
GPU_ID = config["GPU"]
EPOCH_START = config["EPOCH_START"]
EPOCH_END = config["EPOCH_END"]
EPOCH_PERIOD = config["EPOCH_PERIOD"]

# Training parameter (subject model)
TRAINING_PARAMETER = config["TRAINING"]
NET = TRAINING_PARAMETER["NET"]
LEN = TRAINING_PARAMETER["train_num"]


# define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
sys.path.append(CONTENT_PATH)
import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))


epoch = 100
data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, device=DEVICE, epoch_name='Epoch',classes=CLASSES,verbose=1)
# if PREPROCESS:
#     data_provider._meta_data()
pred = data_provider.get_pred(epoch,data_provider.train_representation(epoch).reshape(data_provider.train_representation(epoch).shape[0], 512)).argmax(axis=1)
label = data_provider.train_labels(epoch)
testpred = data_provider.get_pred(epoch,data_provider.test_representation(epoch).reshape(data_provider.test_representation(epoch).shape[0], 512)).argmax(axis=1)
testlabel = data_provider.test_labels(epoch)
k = 0
k_1 = 0
for i in range(len(pred)):
    if pred[i] == label[i]:
        k = k +1

for i in range(len(testpred)):
    if testpred[i] == testlabel[i]:
        k_1 = k_1 +1
print(k/len(label))
print(k_1/len(testlabel))