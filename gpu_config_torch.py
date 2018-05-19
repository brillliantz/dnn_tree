import os
# import torch


gpu_no = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)

# use limited GPU resources
# Assume that we are on a CUDA machine, then this should print a CUDA device:
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("torch device: ", device)
