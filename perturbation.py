from singleVis.utils import *
from singleVis.data import NormalDataProvider
import torch
import sys
sys.path.append("/home/yiming/ContrastDebugger")
import numpy as np
import os, json

from sklearn.neighbors import NearestNeighbors
from singleVis.SingleVisualizationModel import VisModel
from singleVis.utils import *

from singleVis.eval.evaluate import *
import torch
import json

from scipy import stats as stats
from singleVis.projector import DVIProjector, TimeVisProjector

from singleVis.visualizer import visualizer
def adv_gen(data,iteration,noise_scale=0.05, surrond_num=1):
            # # define the noise sclae
            noise_scale = noise_scale
            # # the enhanced image list
            enhanced_images = []
            # # add n version noise image for each image
            for _ in range(surrond_num):
                # copy original data
                perturbed_images = np.copy(data)
                # add Gussian noise
                noise = np.random.normal(loc=0, scale=noise_scale, size=perturbed_images.shape)
                perturbed_images += noise
                # make sure all the pxiels will be put in the range of 0 to 1
                np.clip(perturbed_images, 0, 1, out=perturbed_images)
                enhanced_images.append(perturbed_images)
            enhanced_images = np.concatenate(enhanced_images, axis=0)
            print("the shape of enhanced_images",enhanced_images.shape)
            # enhanced_images = enhanced_images.to(self.DEVICE)
            enhanced_images = torch.Tensor(enhanced_images)
            enhanced_images = enhanced_images.to(data_provider.DEVICE)
            
            repr_model = feature_function(iteration,net)
            border_centers = batch_run(repr_model, enhanced_images)

            return border_centers

def feature_function(epoch,model):
        model_path = os.path.join(data_provider.content_path, "Model")
        model_location = os.path.join(model_path, "{}_{:d}".format("Epoch", epoch), "subject_model.pth")
        # model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")), strict=False)
        model.to(data_provider.DEVICE)
        model.eval()

        fea_fn = model.feature
        return fea_fn

def if_border(data):
    mesh_preds = data_provider.get_pred(epoch, data)
    mesh_preds = mesh_preds + 1e-8

    sort_preds = np.sort(mesh_preds, axis=1)
    diff = (sort_preds[:, -1] - sort_preds[:, -2]) / (sort_preds[:, -1] - sort_preds[:, 0])
    border = np.zeros(len(diff), dtype=np.uint8) + 0.05
    border[diff < 0.15] = 1
        
    return border

DEVICE='cuda:0'

CONTENT_PATH = "/home/yiming/ContrastDebugger/EXP/mnist"
sys.path.append(CONTENT_PATH)
with open(os.path.join(CONTENT_PATH, "config_dvi_modi.json"), "r") as f:
    config = json.load(f)
# config = config[VIS_METHOD]

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

# Training parameter (visualization model)
VISUALIZATION_PARAMETER = config["VISUALIZATION"]
LAMBDA1 = VISUALIZATION_PARAMETER["LAMBDA1"]
LAMBDA2 = VISUALIZATION_PARAMETER["LAMBDA2"]
B_N_EPOCHS = VISUALIZATION_PARAMETER["BOUNDARY"]["B_N_EPOCHS"]
L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
ENCODER_DIMS = VISUALIZATION_PARAMETER["ENCODER_DIMS"]
DECODER_DIMS = VISUALIZATION_PARAMETER["DECODER_DIMS"]
S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]

VIS_MODEL_NAME = 'trustvis_sk'
# VIS_MODEL_NAME = 'dvi_grid_base_al_only'
# VIS_MODEL_NAME = 'timevis'
EVALUATION_NAME = VISUALIZATION_PARAMETER["EVALUATION_NAME"]

# Define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

model = VisModel(ENCODER_DIMS, DECODER_DIMS)

projector = DVIProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)
# projector = TimeVisProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)

n_neighbors = 15
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
epoch = 6

sys.path.append(CONTENT_PATH)
import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))

data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, device=DEVICE, epoch_name='Epoch',classes=CLASSES,verbose=1)

training_data_path = os.path.join(data_provider.content_path, "Training_data")
training_data = torch.load(os.path.join(training_data_path, "training_dataset_data.pth"),
                                   map_location="cpu")

training_data = training_data.to(data_provider.DEVICE).cpu().numpy()
noise_scale = 0.8 ### 0.8 or 0.1
X = adv_gen(training_data,epoch, noise_scale,1)



training_emd = projector.batch_project(epoch,X)

training_new_data = projector.batch_inverse(epoch,training_emd )
pred =  data_provider.get_pred(epoch, X).argmax(axis=1)
new_pred = data_provider.get_pred(epoch, training_new_data).argmax(axis=1)
k = 0
b = 0
old_border_list = if_border(X )
new_border_list = if_border(training_new_data)
for i in range(len(pred)):
    if pred[i] != new_pred[i]:
        k = k+1
        if old_border_list[i] == 1:
            b = b + 1

m = 0
for i in range(len(pred)):
    if old_border_list[i] != new_border_list[i]:
        m = m+1

testing_data_path = os.path.join(data_provider.content_path, "Testing_data")
testing_data = torch.load(os.path.join(testing_data_path, "testing_dataset_data.pth"),
                                   map_location="cpu")

testing_data = testing_data.to(data_provider.DEVICE).cpu().numpy()

Xtest = adv_gen(testing_data,epoch, noise_scale,1)

testing_emd = projector.batch_project(epoch,Xtest)

testing_new_data = projector.batch_inverse(epoch,testing_emd )
testpred =  data_provider.get_pred(epoch, Xtest).argmax(axis=1)
new_testpred = data_provider.get_pred(epoch, testing_new_data).argmax(axis=1)
ktest = 0
btest = 0
old_testborder_list = if_border(Xtest)
new_testborder_list = if_border(testing_new_data)
for i in range(len(testpred)):
    if testpred[i] != new_testpred[i]:
        ktest = ktest+1
        if old_testborder_list[i] == 1:
            btest = btest + 1

mtest = 0
for i in range(len(testpred)):
    if old_testborder_list[i] != new_testborder_list[i]:
        mtest = mtest+1

print('vis error num:',k,'vis error on boundary: ', b ,'boundary flip:',m )
print('vis error num:',ktest,'vis error on boundary: ', btest ,'boundary flip:',mtest )