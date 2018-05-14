# use limited GPU resources
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf
gpuconfig = tf.ConfigProto()
gpuconfig.gpu_options.allow_growth = True
