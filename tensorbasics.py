import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#getting rid of unnnecessary error message "2023-03-15 05:07:35.774627: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
#To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
#2023-03-15 05:07:37.213964: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3495 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6
#"

import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

#initialization of Tensors
x = tf.constant(4, shape=(1,1), dtype=tf.float32)

y = tf.constant([[1,2,3],[4,5,6]])


a = tf.ones((3,3)) #3 x 3 matrix with values of ones

b = tf.zeros((2,3)) #2 x 3 matrix of zeros

c = tf.eye(3) # I for the identity matrix(eye)

d = tf.random.normal((3,3), mean=0, stddev=1) #standard normal distribution

e = tf.random.uniform((1,3), minval=0, maxval=1) #unifrom distribution

f = tf.range(9) #a vector of 0,1,2,3,4,5,6,7,8

g = tf.range(start=1, limit=10, delta=2) #delta means a step.after a value the program skips to the next two value of the range

x = tf.cast(x, dtype=tf.float64) #a way to convert between different data types
#tf.float (15,32,64), tf.int (8,16,32,64) tf.bool


#Mathematical Operations
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])

z = tf.add(x,y) #addition 1 x 3 matrix tensors. z = x + y will also work
print(z)
h = tf.subtract(x,y) #subtraction
print(h)
i = tf.divide(x,y) #division x/y works too
print(i)
j = tf.multiply(x,y) #multiplication
print(j)

k = tf.tensordot(x, y, axes=1) #dot product of matrix
print(k)
l = tf.reduce_sum(x*y,axis=0) #dot product
print(l)
m = x ** 5 #x exponent 5
print(m)

#exampe of matrix multiplication
x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
n = tf.matmul(x,y) # x @ y also works
print(n)

