import torch
import sys,os
import time
import argparse

from singleVis.SingleVisualizationModel import SingleVisualizationModel
from singleVis.data import NormalDataProvider
from singleVis.eval.evaluator import SegEvaluator
from singleVis.projector import EvalProjector

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('--exp','-e', type=str)
args = parser.parse_args()

CONTENT_PATH = args.content_path
EXP = args.exp
sys.path.append(CONTENT_PATH)
from config import config

# record output information
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
sys.stdout = open(os.path.join(CONTENT_PATH, "Model", "exp_{}".format(EXP), now+".txt"), "w")

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
LAMBDA = VISUALIZATION_PARAMETER["LAMBDA"]
S_LAMBDA = VISUALIZATION_PARAMETER["S_LAMBDA"]
B_N_EPOCHS = VISUALIZATION_PARAMETER["BOUNDARY"]["B_N_EPOCHS"]
L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
INIT_NUM = VISUALIZATION_PARAMETER["INIT_NUM"]
ALPHA = VISUALIZATION_PARAMETER["ALPHA"]
BETA = VISUALIZATION_PARAMETER["BETA"]
MAX_HAUSDORFF = VISUALIZATION_PARAMETER["MAX_HAUSDORFF"]
HIDDEN_LAYER = VISUALIZATION_PARAMETER["HIDDEN_LAYER"]
S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
T_N_EPOCHS = VISUALIZATION_PARAMETER["T_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]

# SEGMENTS = VISUALIZATION_PARAMETER["SEGMENTS"]
# RESUME_SEG = VISUALIZATION_PARAMETER["RESUME_SEG"]

# define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

content_path = CONTENT_PATH
sys.path.append(content_path)

import Model.model as subject_model
# net = resnet18()
net = eval("subject_model.{}()".format(NET))
classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


########################################################################################################################
#                                                    TRAINING SETTING                                                  #
########################################################################################################################
data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, classes=CLASSES,verbose=1)
if PREPROCESS:
    data_provider.initialize(LEN//10, l_bound=L_BOUND)

model = SingleVisualizationModel(input_dims=512, output_dims=2, units=256, hidden_layer=HIDDEN_LAYER)
projector = EvalProjector(vis_model=model, content_path=CONTENT_PATH, device=DEVICE, exp=EXP)


# ########################################################################################################################
# #                                                      VISUALIZATION                                                   #
# ########################################################################################################################

# from singleVis.visualizer import visualizer

# vis = visualizer(data_provider, projector, 200)
# save_dir = os.path.join(data_provider.content_path, "img")
# os.system("mkdir -p {}".format(save_dir))

# for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
#     vis.savefig(i, path=os.path.join(save_dir, "{}_{}_tnn.png".format(DATASET, i)))
#     # data = data_provider.train_representation(i)
#     # labels = data_provider.train_labels(i)
#     # selected = labels != np.array(noise_labels)
#     # data = data[selected]
#     # labels = np.array(noise_labels)[selected]
#     # vis.savefig_cus(i, data, labels, labels, path=os.path.join(save_dir, "{}_{}_tnn.png".format(DATASET, i)))


########################################################################################################################
#                                                       EVALUATION                                                     #
########################################################################################################################

EVAL_EPOCH_DICT = {
    "cifar10": [20,100,200],
    "fmnist": [10,30,50],
    "mnist":[4,12,20]
}

eval_epochs = EVAL_EPOCH_DICT[DATASET]

evaluator = SegEvaluator(data_provider, projector, EXP)
# evaluator.save_epoch_eval(eval_epochs[0], 10, temporal_k=3, save_corrs=True, file_name="test_evaluation_tnn")
evaluator.save_epoch_eval(eval_epochs[0], 15, temporal_k=5, save_corrs=False, file_name="test_evaluation_hybrid")
# evaluator.save_epoch_eval(eval_epochs[0], 20, temporal_k=7, save_corrs=False, file_name="test_evaluation_tnn")

# evaluator.save_epoch_eval(eval_epochs[1], 10, temporal_k=3, save_corrs=True, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[1], 15, temporal_k=5, save_corrs=False, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[1], 20, temporal_k=7, save_corrs=False, file_name="test_evaluation_tnn")

# evaluator.save_epoch_eval(eval_epochs[2], 10, temporal_k=3, save_corrs=True, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[2], 15, temporal_k=5, save_corrs=False, file_name="test_evaluation_tnn")
# evaluator.save_epoch_eval(eval_epochs[2], 20, temporal_k=7, save_corrs=False, file_name="test_evaluation_tnn")