# tensor mutation. environment in ["tensorflow", "pytorch"]
# tensorflow: NHWC   pytorch: NCHW
import copy
import random

import tensorflow
import tensorflow as tf
import torch as torch
from DataStruct.globalConfig import GlobalConfig
random_cropping_h = random.randint(1,GlobalConfig.h)
random_cropping_w = random.randint(1,GlobalConfig.w)
def tensor_mutation(tensor, environment, mutation_strategy):
    this_mutation = mutation_strategy
    if this_mutation == "WDC":
        if environment == "tensorflow":
            result_tensor = tf.concat([tensor,tensor], 2)
            return result_tensor
        elif environment == "pytorch":
            result_tensor = torch.cat([tensor,tensor],dim = 3)
            return result_tensor

    elif this_mutation == "HDC":
        if environment == "tensorflow":
            result_tensor = tf.concat([tensor,tensor], 1)
            return result_tensor
        elif environment == "pytorch":
            result_tensor = torch.cat([tensor,tensor],dim = 2)
            return result_tensor

    elif this_mutation == "CDC":
        if environment == "tensorflow":
            result_tensor = tf.concat([tensor,tensor], 3)
            return result_tensor
        elif environment == "pytorch":
            result_tensor = torch.cat([tensor,tensor],dim = 1)
            return result_tensor

    elif this_mutation == "BDC":
        if environment == "tensorflow":
            result_tensor = tf.concat([tensor,tensor], 0)
            return result_tensor
        elif environment == "pytorch":
            result_tensor = torch.cat([tensor,tensor],dim = 0)
            return result_tensor

    elif this_mutation == "WDP":
        if environment == "tensorflow":
            padding_shape = [tensor.shape[0],tensor.shape[1],1,tensor.shape[3]]
            zero_tensor = tf.zeros(padding_shape)
            result_tensor = tf.concat([tensor,zero_tensor], 2)
            return result_tensor
        elif environment == "pytorch":
            padding_shape = (tensor.shape[0],tensor.shape[1],tensor.shape[2],1)
            zero_tensor = torch.zeros(size=padding_shape)
            result_tensor = torch.cat([tensor,zero_tensor], 3)
            return result_tensor

    elif this_mutation == "HDP":
        if environment == "tensorflow":
            padding_shape = [tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]]
            zero_tensor = tf.zeros(padding_shape)
            result_tensor = tf.concat([tensor, zero_tensor], 1)
            return result_tensor
        elif environment == "pytorch":
            padding_shape = (tensor.shape[0], tensor.shape[1], 1, tensor.shape[3])
            zero_tensor = torch.zeros(size=padding_shape)
            result_tensor = torch.cat([tensor, zero_tensor], 2)
            return result_tensor

    elif this_mutation == "CDP":
        if environment == "tensorflow":
            padding_shape = [tensor.shape[0], tensor.shape[1], tensor.shape[2], 1]
            zero_tensor = tf.zeros(padding_shape)
            result_tensor = tf.concat([tensor, zero_tensor], 3)
            return result_tensor
        elif environment == "pytorch":
            padding_shape = (tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])
            zero_tensor = torch.zeros(size=padding_shape)
            result_tensor = torch.cat([tensor, zero_tensor], 1)
            return result_tensor

    elif this_mutation == "BDP":
        if environment == "tensorflow":
            padding_shape = [1, tensor.shape[1], tensor.shape[2], tensor.shape[3]]
            zero_tensor = tf.zeros(padding_shape)
            result_tensor = tf.concat([tensor, zero_tensor], 0)
            return result_tensor
        elif environment == "pytorch":
            padding_shape = (1, tensor.shape[1], tensor.shape[2], tensor.shape[3])
            zero_tensor = torch.zeros(size=padding_shape)
            result_tensor = torch.cat([tensor, zero_tensor], 0)
            return result_tensor

    elif this_mutation == "HWDT":
        if environment == "tensorflow":
            result_tensor = tf.transpose(tensor, perm=[0,2,1,3])
            return result_tensor
        elif environment == "pytorch":
            result_tensor = tensor.transpose(2,3)
            return result_tensor

    elif this_mutation == "RC":
        if environment == "tensorflow":
            result_tensor = tf.image.crop_to_bounding_box(tensor, 0, 0, random_cropping_h, random_cropping_w)
            return result_tensor
        elif environment == "pytorch":
            result_tensor = copy.deepcopy(tensor[:,:,0:random_cropping_h,0:random_cropping_w])
            return result_tensor

    elif this_mutation == "FT":
        if environment == "tensorflow":
            result_tensor = tf.cast(tensor, dtype= tensorflow.float32)
            return result_tensor
        elif environment == "pytorch":
            result_tensor = tensor.type(dtype=torch.float32)
            return result_tensor

    elif this_mutation == "DT":
        if environment == "tensorflow":
            result_tensor = tf.cast(tensor, dtype=tensorflow.double)
            return result_tensor
        elif environment == "pytorch":
            result_tensor = tensor.type(dtype=torch.double)
            return result_tensor

    elif this_mutation == "BFT":
        if environment == "tensorflow":
            result_tensor = tf.cast(tensor, dtype=tensorflow.bfloat16)
            return result_tensor
        elif environment == "pytorch":
            result_tensor = tensor.type(dtype=torch.bfloat16)
            return result_tensor
    else:
        print("need more tensor_mutation_strategies!")
