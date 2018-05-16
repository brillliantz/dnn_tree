import os
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# use limited GPU resources
# Assume that we are on a CUDA machine, then this should print a CUDA device:
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("torch device: ", device)
