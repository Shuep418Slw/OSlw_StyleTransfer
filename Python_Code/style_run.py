#from https://github.com/hzy46/fast-neural-style-tensorflow
#and I modify some code for test
#you can run this .py on raspberry pi

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import style_trans_we as stwe
import tf2lw
import cv2
from PIL import Image

def myconv2d(x, input_filters, output_filters, kernel, strides,we_np, mode='CONSTANT'):
    with tf.variable_scope('conv'):
        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(we_np, name='weight',trainable=False)
        x_padded = tf.pad(x, [[0, 0], [int(kernel / 2), int(kernel / 2)], [int(kernel / 2), int(kernel / 2)], [0, 0]], mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')

def relu(input):
    relu = tf.nn.relu(input)
    # convert nan to zero (nan != nan)
    nan_to_zero = tf.where(tf.equal(relu, relu), relu, tf.zeros_like(relu))
    return nan_to_zero

def resize_conv2d(x, input_filters, output_filters, kernel, strides, training,we_np):
    with tf.variable_scope('conv_transpose'):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return myconv2d(x_resized, input_filters, output_filters, kernel, strides,we_np)

def instance_norm2(x,input_filters,gamma_np,beta_np):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    gamma=tf.constant(gamma_np, name='IN_gamma')
    beta=tf.constant(beta_np, name='IN_beta')
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))*gamma+beta

def instance_norm3(x,input_filters,gamma_np,beta_np):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    gamma=tf.constant(gamma_np, name='IN_gamma')
    beta=tf.constant(beta_np, name='IN_beta')
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))*gamma+beta,mean,var

def residual(x, filters, kernel, strides,we1_np,we2_np):
    with tf.variable_scope('residual'):
        conv1 = myconv2d(x, filters, filters, kernel, strides,we1_np)
        conv2 = myconv2d(relu(conv1), filters, filters, kernel, strides,we2_np)
        residual = x + conv2
        return residual


mean_list=[]
var_list=[]

def model(image):
    training=False
    relist=[]
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='CONSTANT')

    with tf.variable_scope('conv1'):
        temp,tempm,tempv=instance_norm3(myconv2d(image, 3, 32, 9, 1,stwe.conv1_conv_weight),32,stwe.conv1_IN_gamma,stwe.conv1_IN_beta)
        conv1 = relu(temp)
        relist.append(conv1)
        mean_list.append(tempm)
        var_list.append(tempv)
    with tf.variable_scope('conv2'):
        temp, tempm, tempv = instance_norm3(myconv2d(conv1, 32, 64, 3, 2,stwe.conv2_conv_weight),64,stwe.conv2_IN_gamma,stwe.conv2_IN_beta)
        conv2 = relu(temp)
        relist.append(conv2)
        mean_list.append(tempm)
        var_list.append(tempv)
    with tf.variable_scope('conv3'):
        globaltest1=myconv2d(conv2, 64, 128, 3, 1, stwe.conv3_conv_weight)
        globaltest=tf.nn.avg_pool(globaltest1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        temp, tempm, tempv = instance_norm3(globaltest, 128, stwe.conv3_IN_gamma,stwe.conv3_IN_beta)
        conv3 = relu(temp)
        relist.append(conv3)
        mean_list.append(tempm)
        var_list.append(tempv)
    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1,stwe.res1_residual_conv_weight,stwe.res1_residual_conv_1_weight)
        relist.append(res1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1,stwe.res2_residual_conv_weight,stwe.res2_residual_conv_1_weight)
        relist.append(res2)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1,stwe.res3_residual_conv_weight,stwe.res3_residual_conv_1_weight)
        relist.append(res3)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1,stwe.res4_residual_conv_weight,stwe.res4_residual_conv_1_weight)
        relist.append(res4)
    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1,stwe.res5_residual_conv_weight,stwe.res5_residual_conv_1_weight)
        relist.append(res5)
    print(res5.get_shape())
    with tf.variable_scope('deconv1'):
        # deconv1 = relu(instance_norm(conv2d_transpose(res5, 128, 64, 3, 2)))
        deconv1 = relu(instance_norm2(resize_conv2d(res5, 128, 64, 3, 1, training,stwe.deconv1_conv_transpose_conv_weight),64,stwe.deconv1_IN_gamma,stwe.deconv1_IN_beta))
        relist.append(deconv1)
    with tf.variable_scope('deconv2'):
        # deconv2 = relu(instance_norm(conv2d_transpose(deconv1, 64, 32, 3, 2)))
        deconv2 = relu(instance_norm2(resize_conv2d(deconv1, 64, 32, 3, 1, training,stwe.deconv2_conv_transpose_conv_weight),32,stwe.deconv2_IN_gamma,stwe.deconv2_IN_beta))
        relist.append(deconv2)
    with tf.variable_scope('deconv3'):
        # deconv_test = relu(instance_norm(conv2d(deconv2, 32, 32, 2, 1)))
        deconv3 = tf.nn.tanh(instance_norm2(myconv2d(deconv2, 32, 3, 9, 1,stwe.deconv3_conv_weight),3,stwe.deconv3_IN_gamma,stwe.deconv3_IN_beta))
        relist.append(deconv3)

    y = (deconv3+1)*127.5
    return y,relist

img=np.array(Image.open('picture.jpg'))
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

img=cv2.resize(img,(512,512),interpolation=cv2.INTER_LINEAR)

pic_shape=np.shape(img)
im=np.resize(img,[1,pic_shape[0],pic_shape[1],3])
im=im.astype(np.float32)
im[0,:,:,0]=im[0,:,:,0]-_R_MEAN
im[0,:,:,1]=im[0,:,:,1]-_G_MEAN
im[0,:,:,2]=im[0,:,:,2]-_B_MEAN

image=tf.constant(im,dtype=tf.float32)

yout,relist=model(image)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)


print("run.....")
yout_np=sess.run(yout)
print('Complete')

print('save...')
yout_np=yout_np[0,:,:,:]
yout_np=yout_np.astype(np.uint8)
imsave=Image.fromarray(yout_np)
imsave.save("tf_out.png")

print('Complete')
