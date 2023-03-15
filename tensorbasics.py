import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#getting rid of unnnecessary error message "2023-03-15 05:07:35.774627: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
#To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
#2023-03-15 05:07:37.213964: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3495 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6
#"

import tensorflow as tf

#initialization of Tensors
x = tf.constant(4, shape=(1,1), dtype=tf.float32)
print(x)
y =  tf.constant([[1,2,3],[4,5,6]])
print(y)

a = tf.ones((3,3)) #3 x 3 matrix with values of ones
print(a)
b = tf.zeros((2,3)) #2 x 3 matrix of zeros
print(b)
c = tf.eye(3) # I for the identity matrix(eye)
print(c)
d = tf.random.normal((3,3), mean=0, stddev=1) #standard normal distribution
print(d)
e = tf.random.uniform((1,3), minval=0, maxval=1) #unifrom distribution
print(e)
f = tf.range(9) #a vector of 0,1,2,3,4,5,6,7,8
print(f)
g = tf.range(start=1, limit=10, delta=2) #delta means a step.after a value the program skips to the next two value of the range
print(g)
x = tf.cast(x, dtype=tf.float64) #a way to convert between different data types
print(x)
#tf.float (15,32,64), tf.int (8,16,32,64) tf.bool