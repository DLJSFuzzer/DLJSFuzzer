# 用于非DLJSFuzzer方法，cell不重复，不添加任何操作
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralTorchNet(nn.Module):
    def __init__(self,in_channel,final_module,channels):
        super(GeneralTorchNet, self).__init__()
        self.in_channel = in_channel
        self.final_module = final_module
        self.channels = channels
    def forward(self,x):
        # 各节点的张量。
        tensors = []
        # 判断某张量是否有初始值。
        tensors_isnull = [True] * len(self.channels)
        tensors.append(x)
        tensors_isnull[0] = False
        for i in range(len(self.channels) - 1):
            # 随意赋一个同类型的初始值
            tensors.append(True)
        final_point = 0
        for eachOperation in self.final_module:
            fromIndex = eachOperation.fromIndex
            final_point = eachOperation.toIndex
            input = tensors[fromIndex]
            toIndex = eachOperation.toIndex
            # 本NAS规定所有操作出通道数与入通道数相同。
            operator_in_channel = self.channels[fromIndex]*self.in_channel
            operator = eachOperation.operator

            # if operator != -1 and operator != 0:
            #     operator = -1
            # indentity
            if operator == -1:
                # print("pytorch执行了了操作-1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    tensors[toIndex] = tensors[fromIndex].clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    temp = torch.cat((tensors[toIndex], input), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 1 × 1 convolution of C channels
            elif operator == 1:
                # print("pytorch执行了操作1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是1*1卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    filter = torch.Tensor(np.ones([self.in_channel, operator_in_channel, 1, 1]))
                    thisresult = F.conv2d(input=input, weight=filter, stride=[1, 1], padding=0)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是1*1卷积的代码。
                    filter = torch.Tensor(np.ones([self.in_channel, operator_in_channel, 1, 1]))
                    thisresult = F.conv2d(input=input, weight=filter, stride=[1, 1], padding=0)
                    tensors[toIndex] = torch.cat((tensors[toIndex], thisresult), 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 3 × 3 depthwise convolution
            elif operator == 2:
                # print("pytorch执行了操作2", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:OutChannel、InChannel/groups、H、W
                    filter = torch.ones((operator_in_channel, 1, 3, 3), dtype=torch.float32)
                    # 注：pytorch中dw卷积加pw卷积是普通卷积操作加groups来调节的。
                    # thisresult = F.conv2d(input=input,weight=filter,stride=1,padding=[1,1],groups=operator_in_channel)
                    pad = torch.nn.ZeroPad2d(padding=(1, 1, 1, 1))
                    input = pad(input)
                    thisresult = F.conv2d(input=input, weight=filter, stride=1, padding=0,
                                                            groups=operator_in_channel)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    filter = torch.ones((operator_in_channel, 1, 3, 3), dtype=torch.float32)
                    # thisresult = F.conv2d(input=input,weight=filter,stride=1,padding=[1,1],groups=operator_in_channel)
                    pad = torch.nn.ZeroPad2d(padding=(1, 1, 1, 1))
                    input = pad(input)
                    thisresult = F.conv2d(input=input, weight=filter, stride=1, padding=0,
                                                            groups=operator_in_channel)
                    tensors[toIndex] = torch.cat((tensors[toIndex], thisresult), 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            elif operator == 3:
                # print("pytorch执行了操作3", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:OutChannel、InChannel/groups、H、W
                    depthwise_filter = torch.ones((operator_in_channel, 1, 3, 3), dtype=torch.float32)
                    pointwise_filter = torch.ones((self.in_channel, operator_in_channel, 1, 1),
                                                  dtype=torch.float32)
                    pad = torch.nn.ZeroPad2d(padding=(1, 1, 1, 1))
                    input = pad(input)
                    depthwise_temp = torch.nn.functional.conv2d(input=input, weight=depthwise_filter, stride=1,
                                                                padding=0, groups=operator_in_channel).clone().detach()
                    pointwise_temp = torch.nn.functional.conv2d(input=depthwise_temp, weight=pointwise_filter, stride=1,
                                                                padding=0).clone().detach()
                    tensors[toIndex] = pointwise_temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    depthwise_filter = torch.ones((operator_in_channel, 1, 3, 3), dtype=torch.float32)
                    pointwise_filter = torch.ones((self.in_channel, operator_in_channel, 1, 1),
                                                  dtype=torch.float32)
                    pad = torch.nn.ZeroPad2d(padding=(1, 1, 1, 1))
                    input = pad(input)
                    depthwise_temp = torch.nn.functional.conv2d(input=input, weight=depthwise_filter, stride=1,
                                                                padding=0, groups=operator_in_channel).clone().detach()
                    pointwise_temp = torch.nn.functional.conv2d(input=depthwise_temp, weight=pointwise_filter, stride=1,
                                                                padding=0).clone().detach()
                    tensors[toIndex] = torch.cat((tensors[toIndex], pointwise_temp), 1).clone().detach()
            elif operator == 4:
                # print("pytorch执行了了操作4 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            elif operator == 5:
                # print("pytorch执行了了操作5 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 注意：一定要有count_include_pad=False,不计算补的0，和tensorflow保持一致。
                    result = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True,
                                                count_include_pad=False)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True,
                                                count_include_pad=False)(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 3 × 3 convolution of C channels
            elif operator == 6:
                # print("pytorch执行了操作6 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是3*3卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    filter = torch.Tensor(np.ones([self.in_channel, operator_in_channel, 3, 3]))
                    thisresult = F.conv2d(input=input, weight=filter, stride=[1, 1], padding=1)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是3*3卷积的代码。
                    filter = torch.Tensor(np.ones([self.in_channel, operator_in_channel, 3, 3]))
                    thisresult = F.conv2d(input=input, weight=filter, stride=[1, 1], padding=1)
                    tensors[toIndex] = torch.cat((tensors[toIndex], thisresult), 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #3*3 conv2D_transpose of C channels
            #注：参数代表的是正向卷积的过程，但我要做的是反卷积。
            elif operator == 7:
                # print("pytorch执行了操作7 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是3*3卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    filter = torch.Tensor(np.ones([operator_in_channel, self.in_channel, 3, 3]))
                    thisresult = F.conv_transpose2d(input=input, weight=filter, stride=[1, 1], padding=1)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是3*3卷积的代码。
                    filter = torch.Tensor(np.ones([operator_in_channel, self.in_channel, 3, 3]))
                    thisresult = F.conv_transpose2d(input=input, weight=filter, stride=[1, 1], padding=1)
                    tensors[toIndex] = torch.cat((tensors[toIndex], thisresult), 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #relu
            elif operator == 8:
                # print("pytorch执行了了操作8 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = F.relu(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = F.relu(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #sigmoid
            elif operator == 9:
                # print("pytorch执行了了操作9 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.sigmoid(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.sigmoid(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #tanh
            elif operator == 10:
                # print("pytorch执行了了操作10 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.tanh(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.tanh(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #leakyrelu
            elif operator == 11:
                # print("pytorch执行了了操作11 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.nn.LeakyReLU(negative_slope=0.2)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.nn.LeakyReLU(negative_slope=0.2)(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #prelu。
            elif operator == 12:
                # print("pytorch执行了了操作12 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = input
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = input
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #ELU
            elif operator == 13:
                # print("pytorch执行了了操作13 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.nn.ELU()(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.nn.ELU()(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #batchnorm
            elif operator == 14:
                # print("pytorch执行了了操作14 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.nn.BatchNorm2d(operator_in_channel)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.nn.BatchNorm2d(operator_in_channel)(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #CBR Block（convolution+batchnorm+relu）
            elif operator == 15:
                # print("pytorch执行了了操作15 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    filter = torch.Tensor(np.ones([self.in_channel, operator_in_channel, 1, 1]))
                    conv_result = F.conv2d(input=input, weight=filter, stride=[1, 1], padding=0)
                    batchnorm_result = torch.nn.BatchNorm2d(self.in_channel)(conv_result)
                    result = F.relu(batchnorm_result)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    filter = torch.Tensor(np.ones([self.in_channel, operator_in_channel, 1, 1]))
                    conv_result = F.conv2d(input=input, weight=filter, stride=[1, 1], padding=0)
                    batchnorm_result = torch.nn.BatchNorm2d(self.in_channel)(conv_result)
                    result = F.relu(batchnorm_result)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #maxpool sigmoid
            elif operator == 16:
                # print("pytorch执行了了操作16 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    maxpool_result = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)(input)
                    result = torch.sigmoid(maxpool_result)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    maxpool_result = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)(input)
                    result = torch.sigmoid(maxpool_result)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #maxpool softmax
            elif operator == 17:
                # print("pytorch执行了了操作17 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    maxpool_result = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)(input)
                    result = F.softmax(maxpool_result)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    maxpool_result = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)(input)
                    result = F.softmax(maxpool_result)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #reducemean in hw
            elif operator == 18:
                # print("pytorch执行了了操作18 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.mean(input, dim=(2, 3), keepdim=True)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.mean(input, dim=(2, 3), keepdim=True)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # transpose without data transfer
            elif operator == 19:
                # print("pytorch执行了了操作19 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.transpose(input, 2, 3)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.transpose(input, 2, 3)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 0-dim tensor
            elif operator == 20:
                # print("pytorch执行了了操作20 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.nn.LeakyReLU(negative_slope=float(torch.tensor(0.2)))(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.nn.LeakyReLU(negative_slope=float(torch.tensor(0.2)))(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])


        return tensors[final_point].clone().detach()