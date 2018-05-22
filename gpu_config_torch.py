import os
# import torch


gpu_no = [3,
          #2,
          #4,
          #6,
          ]
gpu_no = [str(i) for i in gpu_no]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_no)

# use limited GPU resources
# Assume that we are on a CUDA machine, then this should print a CUDA device:
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("torch device: ", device)
