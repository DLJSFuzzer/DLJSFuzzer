# 用于非DLJSFuzzer方法，cell不重复，不添加任何操作

import tensorflow as tf

def deep_copy(tensor):
    res = tf.tile(tensor, [1, 1, 1, 1])
    return res


class GeneralTFNet(tf.keras.Model):
    def __init__(self,in_channel,final_module,channels):
        super(GeneralTFNet, self).__init__()
        self.in_channel = in_channel
        self.final_module = final_module
        self.channels = channels

    @tf.function
    def call(self, inputs):
        # 各节点的张量。
        tensors = []
        # 判断某张量是否有初始值。
        tensors_isnull = [True] * len(self.channels)
        tensors.append(inputs)
        tensors_isnull[0] = False
        for i in range(len(self.channels) - 1):
            # 随意赋一个同类型的初始值
            tensors.append(True)
        final_point = 0
        for eachOperation in self.final_module:
            final_point = eachOperation.toIndex
            fromIndex = eachOperation.fromIndex
            input = tensors[fromIndex]
            toIndex = eachOperation.toIndex
            # in_channel表示操作的入通道数
            operator_in_channel = self.channels[fromIndex] * self.in_channel
            operator = eachOperation.operator

            # if operator != -1 and operator != 0:
            #     operator = -1
            # indentity
            if operator == -1:
                # print("tensorflow执行了了操作-1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    tensors[toIndex] = deep_copy(tensors[fromIndex])
                    # tensor = copy.deepcopy(tensors_isnull)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    temp = tf.concat([tensors[toIndex], input], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            # 1 × 1 convolution of C channels
            elif operator == 1:
                # print("tensorflow执行了操作1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:H、W、InChannel、OutChannel
                    filter = tf.ones((1, 1,operator_in_channel , self.in_channel), dtype=tf.float32)
                    thisresult = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
                    tensors[toIndex] = deep_copy(thisresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    filter = tf.ones((1, 1, operator_in_channel, self.in_channel), dtype=tf.float32)
                    thisresult = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
                    temp = tf.concat([tensors[toIndex], thisresult], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            # 3 × 3 depthwise convolution
            elif operator == 2:
                # print("tensorflow执行了操作2", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:H、W、InChannel、OutChannel的倍数
                    filter = tf.ones((3, 3, operator_in_channel, 1), dtype=tf.float32)
                    thisresult = tf.nn.depthwise_conv2d(input=input, filter=filter, strides=[1, 1, 1, 1],
                                                        padding='SAME')
                    tensors[toIndex] = deep_copy(thisresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    filter = tf.ones((3, 3, operator_in_channel, 1), dtype=tf.float32)
                    thisresult = tf.nn.depthwise_conv2d(input=input, filter=filter, strides=[1, 1, 1, 1],
                                                        padding='SAME')
                    temp = tf.concat([tensors[toIndex], thisresult], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                # 为了debug,先把这两步注释掉。
                # tensors[toIndex] = tf.layers.batch_normalization(tensors[toIndex])
                # tensors[toIndex] = tf.nn.relu(tensors[toIndex])
            elif operator == 3:
                # print("tensorflow执行了操作3", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 各filter参数顺序:H、W、InChannel、OutChannel
                    depthwise_filter = tf.ones((3, 3, operator_in_channel, 1), tf.float32)
                    pointwise_filter = tf.constant(value=1.0, shape=[1, 1, operator_in_channel, self.in_channel],
                                                   dtype=tf.float32)
                    tempresult = tf.nn.separable_conv2d(input=input, depthwise_filter=depthwise_filter,
                                                        pointwise_filter=pointwise_filter,
                                                        strides=[1, 1, 1, 1], padding='SAME')

                    tensors[toIndex] = deep_copy(tempresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    depthwise_filter = tf.ones((3, 3, operator_in_channel, 1), dtype=tf.float32)
                    pointwise_filter = tf.ones((1, 1, operator_in_channel, self.in_channel), dtype=tf.float32)
                    tempresult = tf.nn.separable_conv2d(input=input, depthwise_filter=depthwise_filter,
                                                        pointwise_filter=pointwise_filter,
                                                        strides=[1, 1, 1, 1], padding='SAME')
                    temp = tf.concat([tensors[toIndex], tempresult], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            elif operator == 4:
                # print("tensorflow执行了了操作4 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.max_pool2d(input=input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.max_pool2d(input=input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            elif operator == 5:
                # print("tensorflow执行了了操作5 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.avg_pool2d(input=input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.avg_pool2d(input=input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            # 3 × 3 convolution of C channels
            elif operator == 6:
                # print("tensorflow执行了操作6", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:H、W、InChannel、OutChannel
                    filter = tf.ones((3, 3, operator_in_channel, self.in_channel), dtype=tf.float32)
                    thisresult = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
                    tensors[toIndex] = deep_copy(thisresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    filter = tf.ones((3, 3, operator_in_channel, self.in_channel), dtype=tf.float32)
                    thisresult = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
                    temp = tf.concat([tensors[toIndex], thisresult], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #3*3 conv2D_transpose of C channels
            #注：参数代表的是正向卷积的过程，但我要做的是反卷积。
            elif operator==7:
                # print("tensorflow执行了操作6", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:H、W、InChannel、OutChannel
                    filter = tf.ones((3, 3, self.in_channel, operator_in_channel), dtype=tf.float32)
                    inputshape = tf.shape(input)
                    output_shape = [inputshape[0],inputshape[1],inputshape[2],self.in_channel]
                    thisresult = tf.nn.conv2d_transpose(input=input, filters=filter,output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME')
                    tensors[toIndex] = deep_copy(thisresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    filter = tf.ones((3, 3, self.in_channel, operator_in_channel), dtype=tf.float32)
                    inputshape = tf.shape(input)
                    output_shape = [inputshape[0],inputshape[1],inputshape[2],self.in_channel]
                    thisresult = tf.nn.conv2d_transpose(input=input, filters=filter,output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME')
                    temp = tf.concat([tensors[toIndex], thisresult], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #relu
            elif operator == 8:
                # print("tensorflow执行了了操作8 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.relu(input)
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.relu(input)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #sigmoid
            elif operator == 9:
                # print("tensorflow执行了了操作9 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.sigmoid(input)
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.sigmoid(input)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #tanh
            elif operator == 10:
                # print("tensorflow执行了了操作10 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.tanh(input)
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.tanh(input)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #relu
            elif operator == 11:
                # print("tensorflow执行了了操作11 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.leaky_relu(input)
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.leaky_relu(input)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #prelu
            elif operator == 12:
                # print("tensorflow执行了了操作12 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = input
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = input
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #elu
            elif operator == 13:
                # print("tensorflow执行了了操作13 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.elu(input)
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.elu(input)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #batchnorm
            elif operator == 14:
                # print("tensorflow执行了了操作14 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    thismean, thisvariance = tf.nn.moments(input, axes=[0, 1, 2], keepdims=True)
                    result = tf.nn.batch_normalization(input, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    thismean, thisvariance = tf.nn.moments(input, axes=[0, 1, 2], keepdims=True)
                    result = tf.nn.batch_normalization(input, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #CBR Block（convolution+batchnorm+relu）
            elif operator == 15:
                # print("tensorflow执行了了操作15 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False

                    filter = tf.ones((1, 1,operator_in_channel , self.in_channel), dtype=tf.float32)
                    conv_result = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')

                    thismean, thisvariance = tf.nn.moments(conv_result, axes=[0, 1, 2], keepdims=True)
                    batchnorm_result = tf.nn.batch_normalization(conv_result, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    result = tf.nn.relu(batchnorm_result)
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。

                    filter = tf.ones((1, 1, operator_in_channel, self.in_channel), dtype=tf.float32)
                    conv_result = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')

                    thismean, thisvariance = tf.nn.moments(conv_result, axes=[0, 1, 2], keepdims=True)
                    batchnorm_result = tf.nn.batch_normalization(conv_result, mean=thismean, variance=thisvariance,
                                                                 offset=None, scale=None, variance_epsilon=1e-5)
                    result = tf.nn.relu(batchnorm_result)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #maxpool sigmoid
            elif operator == 16:
                # print("tensorflow执行了了操作16 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    maxpool_result = tf.nn.max_pool2d(input=input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                    result = tf.nn.sigmoid(maxpool_result)
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    maxpool_result = tf.nn.max_pool2d(input=input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                    result = tf.nn.sigmoid(maxpool_result)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #maxpool softmax
            elif operator == 17:
                # print("tensorflow执行了了操作17 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    maxpool_result = tf.nn.max_pool2d(input=input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                    result = tf.nn.softmax(maxpool_result)
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    maxpool_result = tf.nn.max_pool2d(input=input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                    result = tf.nn.softmax(maxpool_result)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            # reducemean in hw
            elif operator == 18:
                # print("tensorflow执行了了操作18 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.reduce_min(input,axis=[1,2],keepdims=True)
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.reduce_min(input,axis=[1,2],keepdims=True)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            # transpose without data transfer
            elif operator == 19:
                # print("tensorflow执行了了操作19 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.transpose(input,[0,2,1,3])
                    tensors[toIndex] = deep_copy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.transpose(input,[0,2,1,3])
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = deep_copy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            # # 0-dim tensor
            # elif operator == 20:
            #     # print("tensorflow执行了了操作20 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
            #     if tensors_isnull[toIndex] == True:
            #         tensors_isnull[toIndex] = False
            #         result = tf.nn.leaky_relu(input,float(tf.constant(0.2, tf.float32)))
            #         tensors[toIndex] = deep_copy(result)
            #         # #log
            #         # print("tensor"+str(toIndex)+":")
            #         # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #     else:
            #         # concat用于数组之间的操作，因此需要先进行类型转换。
            #         result = tf.nn.leaky_relu(input,float(tf.constant(0.2, tf.float32)))
            #         temp = tf.concat([tensors[toIndex], result], 3)
            #         tensors[toIndex] = deep_copy(temp)
            #         # #log
            #         # print("tensor"+str(toIndex)+":")
            #         # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
        return deep_copy(tensors[final_point])