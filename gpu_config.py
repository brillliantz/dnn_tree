# use limited GPU resources
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import tensorflow as tf
gpuconfig = tf.ConfigProto()
gpuconfig.gpu_options.allow_growth = True
