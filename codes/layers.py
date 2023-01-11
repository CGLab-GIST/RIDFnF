from itertools import accumulate
import sys
import math
import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, initializers, activations
from tensorflow.keras.initializers import GlorotUniform, zeros, ones, Constant, RandomUniform
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate

_module_dcflash = tf.load_op_library('./dcflash_lib.so')

@tf.RegisterGradient("CombFlashWithWarping")
def _comb_flash_with_warping_grad(op, grad):
    img = op.inputs[0]
    flash = op.inputs[1]
    wgtComb = op.inputs[2]
    wgtWarp = op.inputs[3]
    combWinSize = op.get_attr('comb_win_size')
    warpWinSize = op.get_attr('warp_win_size')
    outGradWgtComb, outGradWgtWarp = _module_dcflash.comb_flash_with_warping_grad(grad, img, flash, wgtComb, wgtWarp, comb_win_size=combWinSize, warp_win_size=warpWinSize)
    return [None, None, outGradWgtComb, outGradWgtWarp]
comb_flash_with_warping = _module_dcflash.comb_flash_with_warping

def conv2d(x, config):
    return layers.Conv2D(filters=config['numFilters'],
                         kernel_size=(config['convSize'], config['convSize']),
                         activation=config["convActivation"],
                         padding='same',
                         strides=(1, 1),
                         kernel_initializer=GlorotUniform(),
                         bias_initializer=zeros())(x)

def conv2d_last(x, config):
    return layers.Conv2D(
        filters=config['numOutput'],
        kernel_size=(config['convSize'], config['convSize']),
        padding='same', strides=(1, 1),
        kernel_initializer=GlorotUniform(),
        bias_initializer=zeros())(x)  

def conv2d_last_convolutional_kernel(x, config):
    return layers.Conv2D(
        filters=config['numOutputConvolK'],
        kernel_size=(config['convSize'], config['convSize']),
        padding='same', strides=(1, 1),
        kernel_initializer=GlorotUniform(),
        bias_initializer=zeros())(x)  

def ConvolutionNet(config, x):
    # x: (Batch, H, W, 80)
    x = conv2d(x, config) # 1st layer
    for i in range(8): # 2~9 layers
        # x: (Batch, H, W, 80)
        x = conv2d(x, config)

    w_comb = conv2d_last(x, config) # combination weight
    w_convol_k = conv2d_last_convolutional_kernel(x, config) # convolutional kernel k_c
    return w_comb, w_convol_k

###########################################################################
def MainNet(config, input):
    # input: (B, H, W, numChannels)
    candidates = input[:, :, :, config['CANDIDATE_POS']
        :config['CANDIDATE_POS'] + 3 * config['numCandidates']]  # [I_F I_N]
    # I_F : correlated estimate (flash image).
    # I_N : independent estimate (noisy image).

    # x: input (B, H, W, numInputChannels)
    x = tf.concat([candidates], axis=3) # [I_F I_N]

    # produce two sets of weights (w_G, w_C)
    w_comb, w_convol_k = ConvolutionNet(config, x)

    # relu for normalizing w_comb
    wgts_comb = activations.relu(w_comb)
    sum_w_comb = tf.reduce_sum(wgts_comb, axis=-1, keepdims=True)
    wgts_comb = wgts_comb / tf.maximum(sum_w_comb, 1e-10)

    # relu for normalizing w_convol_k
    wgts_convol_k = activations.relu(w_convol_k)
    sum_w_convol_k = tf.reduce_sum(wgts_convol_k, axis=-1, keepdims=True)
    wgts_convol_k = wgts_convol_k / tf.maximum(sum_w_convol_k, 1e-10)

    # noisy image I^N, flash image I^F
    img_noisy = candidates[:, :, :, 3:6]
    img_flash = candidates[:, :, :, 0:3]  
    
    output = comb_flash_with_warping(img_noisy, img_flash, wgts_comb, wgts_convol_k, \
        comb_win_size=config["kernelSize"], warp_win_size=config["kernelSizeConvolK"])

    return output
###########################################################################