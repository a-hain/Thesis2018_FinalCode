"""
This module performs activation maximization on an AlexNet implementation with
TensorFlow.
An image will be optimized to activate a unit as much as possible.
All relevant variables to be controlled by the user can be found in the
am_constants module.

Original documentation for the AlexNet used here.
################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details:
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
################################################################################

Special files and modules required:
    bvlc_alexnet.npy
    caffe_classes
    am_constants
    am_tools

The former two belong to the network, the latter two have been introduced by
me for activation maximization. (No guarantee that this code structure is the
most sensible one)
To use this method on other networks, the code in the ACTIVATION MAXIMIZATION
part will need to be adapted and a new layer_dict (mapping names to actual
network layers) will need to be hardcoded at the current stage. Aside from that,
not much should be needed to change (but this remains to be tested ^^)
Antonia Hain, 2018
"""

import os
import numpy as np
import time
import cv2
import tensorflow as tf
from caffe_classes import class_names
import am_constants
import am_tools
import sys

def main():
    train_x = np.zeros((1, 227,227,3)).astype(np.float32)
    train_y = np.zeros((1, 1000))
    xdim = train_x.shape[1:]
    ydim = train_y.shape[1]


    #### NETWORK ###################################################################
    # only slightly adapted by me

    with open("bvlc_alexnet.npy", "rb") as alexnet_file:
        net_data = np.load(alexnet_file, encoding="latin1").item()

    # added: set summary folder, if summary is desired
    if am_constants.TENSORBOARD_ACTIVATED:
        summary_dir = "summary/"

    def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
        '''From https://github.com/ethereon/caffe-tensorflow
        '''
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

        if group==1:
            conv = convolve(input, kernel)
        else:
            input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
            kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
        return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

    # I changed this to shape (1,)+xdim as gradient would be 0 on undefined shapes
    # so for now, you can only feed 1 image at a time
    x_in = tf.placeholder(dtype=tf.float32, shape=(1,) + xdim,  name='input_placeholder')

    # added for border regularizer
    center_distance = tf.placeholder(dtype=tf.float32, shape=xdim)

    if am_constants.TENSORBOARD_ACTIVATED:
        #plug image placeholder into tensorboard
        tf.summary.image("generated_image", x_in, 4)

    #conv1
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0], trainable=False)
    conv1b = tf.Variable(net_data["conv1"][1], trainable=False)
    conv1_in = conv(x_in, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv2
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0], trainable=False)
    conv2b = tf.Variable(net_data["conv2"][1], trainable=False)
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    #lrn2
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    kernels = [(3,3),(5,5),(7,7)]

    #maxpool2
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0], trainable=False)
    conv3b = tf.Variable(net_data["conv3"][1], trainable=False)
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0], trainable=False)
    conv4b = tf.Variable(net_data["conv4"][1], trainable=False)
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    #conv5
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0], trainable=False)
    conv5b = tf.Variable(net_data["conv5"][1], trainable=False)
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #fc6
    fc6W = tf.Variable(net_data["fc6"][0], trainable=False)
    fc6b = tf.Variable(net_data["fc6"][1], trainable=False)
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    fc6_softmax = tf.nn.softmax(fc6)

    #fc7
    fc7W = tf.Variable(net_data["fc7"][0], trainable=False)
    fc7b = tf.Variable(net_data["fc7"][1], trainable=False)
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    #fc8
    fc8W = tf.Variable(net_data["fc8"][0], trainable=False)
    fc8b = tf.Variable(net_data["fc8"][1], trainable=False)
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

    #prob
    prob = tf.nn.softmax(fc8)

    ######## ACTIVATION MAXIMIZATION ###############################################

    # to make selection of layer of target unit easier. still requires knowledge of
    # the network but oh well
    layer_dict = {
        'conv1' : conv1[0,5,5],
        'conv2' : conv2[0,5,5],
        'conv3' : conv3[0,5,5],
        'conv4' : conv4[0,5,5],
        'conv5' : conv5[0,5,5],
        'fc6' : fc6[0],
        'fc7' : fc7[0],
        'fc8' : fc8[0],
        'prob' : prob[0]
        }


    # doing this in a try except block to put out somewhat more helpful errors
    try:
        target_layer = layer_dict[am_constants.LAYER_KEY]

        # selects random unit within layer boundaries or specific unit based
        # on configuration
        if am_constants.RANDOM_UNIT:
            idx = np.random.randint(0,target_layer.shape[0])
            TO_MAXIMIZE = target_layer[idx]
        else:
            TO_MAXIMIZE = target_layer[am_constants.UNIT_INDEX]
            idx = am_constants.UNIT_INDEX

    except KeyError: # should happen when layer_dict[am_constants.LAYER_KEY] fails
        raise KeyError("Your selected layer key seems to not exist. Please check in am_constants.py")
    except ValueError: # should happen when am_constants.UNIT_INDEX cannot be accessed in the layer
        raise ValueError("Something went wrong selecting the unit to maximize. Have you made sure your UNIT_INDEX in am_constants is within boundaries?")


    # boolean for convenience
    trans_robust = am_constants.LARGER_IMAGE or am_constants.JITTER

    # define loss (potentially including L2 loss) and gradient
    # loss is negative activation of the unit + (if l2 activated) lambda * image^2
    loss = -tf.reduce_sum(TO_MAXIMIZE) + am_constants.L2_ACTIVATED * am_constants.L2_LAMBDA * tf.reduce_sum(tf.multiply(x_in,x_in))

    # regularize: borders. ignore for now, not working
    if am_constants.BORDER_REG_ACTIVATED:
        loss = am_tools.reg_border(x_in, center_distance, loss)

    # get gradient based on loss
    grad = tf.gradients([loss], [x_in])

    # more tensorboard
    if am_constants.TENSORBOARD_ACTIVATED:
        tf.summary.scalar("loss", tf.reduce_mean(loss))
        summaries = tf.summary.merge_all()

    t = time.time() # to meaure computation time

    # main part
    with tf.Session() as sess:

        if am_constants.TENSORBOARD_ACTIVATED:
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            train_writer = tf.summary.FileWriter(summary_dir, sess.graph)

        # upscaling and jitter use similar algorithms
        # boolean determining if either of them is in place to make things easier
        if am_constants.LARGER_IMAGE and am_constants.JITTER:
            raise Exception("Upscaling and jitter cannot be used simultaneously.")

        if am_constants.LARGER_IMAGE:
            tmp_dimx = am_constants.IMAGE_SIZE_X
            tmp_dimy = am_constants.IMAGE_SIZE_Y
        else:
            tmp_dimx = 227
            tmp_dimy = 227

        if am_constants.RANDOMIZE_INPUT:
            im_big = np.random.randint(-127,128,(1,tmp_dimx,tmp_dimy,3))
        else:
            im_big = np.full((1,tmp_dimx,tmp_dimy,3),0)

        im_big = np.array(im_big,dtype=np.float)

        # create distance image for border region punishment
        # as it is now a placeholder this has to be included here, though it may
        # end up unused.
        distances = [[am_constants.BORDER_FACTOR * np.sqrt((tmp_dimx/2-x)*(tmp_dimx/2-x) + (tmp_dimy/2-y)*(tmp_dimy/2-y))**(am_constants.BORDER_EXP) for y in range(tmp_dimy)] for x in range(tmp_dimx)]
        distances = np.stack((distances, distances, distances),axis=2)

        # get subimage if transformation robustness is used
        if trans_robust:
            im, distances_crop, indx, indy, rand_x, rand_y = am_tools.get_section(im_big, distances, xdim)
        else:
            im = im_big
            distances_crop = distances

        sess.run(tf.global_variables_initializer())

        # get first loss
        output, _loss, _grad= sess.run([prob, loss, grad], feed_dict = {x_in:im, center_distance: distances_crop})

        print("Initial loss:",_loss)

        i = 0 # step counter
        avg_steptime = 0 # step time counter
        loss_list = [-100000 for _ in range(am_constants.LOSS_COUNT)] # list containing last 50 losses

        # while current loss diverges from average of last losses by a factor
        # LOSS_GOAL or more, continue, alternatively stop if we took too many steps
        while np.abs(1 - _loss/np.mean(loss_list)) > am_constants.LOSS_GOAL and i < am_constants.MAX_STEPS:

            # start measuring time for this step
            start = time.time()

            # add previous loss to list. order of the losses doesn't matter so just
            # reassign the value that was assigned 50 iterations ago
            loss_list[i % am_constants.LOSS_COUNT] = _loss

            # update image according to gradient. doing this manually because
            # gradient was computed on placeholder, but is applied to image
            im = im - am_constants.ETA * np.asarray(_grad[0])

            # regularize: clipping pixels with small contribution
            if am_constants.CONTRIBUTION_CLIPPING_ACTIVATED and i % am_constants.CONTRIBUTION_CLIPPING_FREQUENCY == 0:
                im = am_tools.reg_clip_contrib(im,_grad)

            # plug image crop back into big image
            if trans_robust:
                im_big = am_tools.plug_in_section(im_big, im, indx, indy, rand_x, rand_y)
            else:
                im_big = im

            # regularize: clipping pixels with small norm
            if am_constants.NORM_CLIPPING_ACTIVATED and i % am_constants.NORM_CLIPPING_FREQUENCY == 0:
                im_big = am_tools.reg_clip_norm(im_big)

            # regularize: gaussian blur
            if am_constants.BLUR_ACTIVATED and i % am_constants.BLUR_FREQUENCY == 0:
                im_big = am_tools.reg_blur(im_big)

            # get subimage for this step if transformation robustness is used
            if trans_robust:
                im, distances_crop, indx, indy, rand_x, rand_y = am_tools.get_section(im_big, distances, xdim)
            else:
                im = im_big
                distances_crop = distances

            # get new probability, loss, gradient potentially summary
            if am_constants.TENSORBOARD_ACTIVATED:
                output, _fc6, _loss, _grad, _summ = sess.run([prob, fc6, loss, grad, summaries], feed_dict = {x_in:im, center_distance: distances_crop})
                train_writer.add_summary(_summ, i)
            else:
                output, _fc6, _loss, _grad = sess.run([prob, fc6, loss, grad], feed_dict = {x_in:im, center_distance: distances_crop})

                # get probability for complete (downsized) image for the
                # convergence criterion
                # one may remove this. it remains to be tested how much of
                # an actual difference it makes for convergence, but it seems
                # to be very costly
                if am_constants.LARGER_IMAGE:
                    resized = np.zeros((1,227,227,3))
                    resized[0,:,:,0] = cv2.resize(im_big[0,:,:,0],(227,227))
                    resized[0,:,:,1] = cv2.resize(im_big[0,:,:,1],(227,227))
                    resized[0,:,:,2] = cv2.resize(im_big[0,:,:,2],(227,227))
                    _loss = sess.run(loss, feed_dict = {x_in:resized, center_distance: distances_crop})


            # increase steps
            i += 1

            # get time that was needed for this step
            avg_steptime += time.time() - start

        if am_constants.LARGER_IMAGE:
            # get total probability if we created a larger image
            # won't work well if it's not quadratic
            resized = np.zeros((1,227,227,3))
            resized[0,:,:,0] = cv2.resize(im_big[0,:,:,0],(227,227))
            resized[0,:,:,1] = cv2.resize(im_big[0,:,:,1],(227,227))
            resized[0,:,:,2] = cv2.resize(im_big[0,:,:,2],(227,227))
            output = sess.run(prob, feed_dict = {x_in:resized})

    sess.close()

    comptime = time.time()-t
    avg_steptime /= i

    # print computation cost statistics for better overview
    print("\nComputation time", comptime)
    print("\nAvg steptime", avg_steptime)
    print("Steps: ",i)

    ########## GET OUTPUT CLASS ####################################################
    # this was taken and minimally adapted from the original AlexNet class
    # prints top 5 object classes and saves them in a list as well
    print("\nOutput following:\n")

    top5 = []
    for input_im_ind in range(output.shape[0]): # index of input image

        inds = np.argsort(output)[input_im_ind,:]

        # ADDED THIS MYSELF
        winneridx = np.argmax(output[input_im_ind])
        print(f"Winner is {class_names[winneridx]} at index {winneridx} with prob {output[input_im_ind,winneridx]}\n")

        for j in range(5):
            print(class_names[inds[-1-j]], output[input_im_ind, inds[-1-j]])
            top5 += [str(class_names[inds[-1-j]]) + " (" + str(output[input_im_ind, inds[-1-j]]) + ")"]
        print()

    print(top5)

    ######## SAVE DATA ###########################################################
    if trans_robust:
        outputim = im_big[0]
    else:
        outputim = im[0]

    # check if computation converged
    finish = (i < am_constants.MAX_STEPS)

    # using these paths on my system, change or comment out on other computer
    am_tools.save_am_data(outputim, idx, finish, i, comptime, avg_steptime, top5,
         impath= am_constants.SAVE_PATH,
         paramfilename= am_constants.SAVE_PATH + "parameters.csv")

if __name__ == '__main__':
    main()
