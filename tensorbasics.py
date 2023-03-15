import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#getting rid of unnnecessary error message "2023-03-15 05:07:35.774627: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
#To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
#2023-03-15 05:07:37.213964: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3495 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6
#"

import tensorflow as tf

#initialization of Tensors
x = tf.constant(4)
print(x)