# encoding = 'utf-8'
import copy
import os
import subprocess
import time

import numpy

from DataStruct.globalConfig import GlobalConfig
from DataStruct.edge import edge
import tensorflowjs as tfjs
import random
import numpy as np
import tensorflow as tf

import torch
from Method import util
from Method import tensor_mutation

from Database import Input_Helper_Mapping as InputHelper
from Database import Chrome_Helper_Mapping as ChromeHelper
from Database import Msedge_Helper_Mapping as MsedgeHelper

def select_tensor_mutation_strategy():
    total = 0
    weights = GlobalConfig.tensor_mutation_strategy_weight
    chart = [0] * len(weights)
    for p in range(len(weights)):
        total += weights[p]
        chart[p] = total

    thisRan = random.random() * total
    for p in range(len(weights)):
        if thisRan <= chart[p]:
            return GlobalConfig.tensor_mutation_strategies[p]
def get_input_tensor(x, dtype, environment, mutation_strategy):
    if environment=="tensorflow":
        tensor_NCHW=tf.convert_to_tensor(x, dtype=dtype)
        tensor_NHWC=tf.transpose(tensor_NCHW,[0,2,3,1])
        # tensor mutation
        if GlobalConfig.abalation_mode in ["model_only", "none"]:
            return tensor_NHWC
        else:
            mutated_tensor = tensor_mutation.tensor_mutation(tensor_NHWC, environment, mutation_strategy)
            return mutated_tensor
    if environment=="pytorch":
        # tensor mutation
        if GlobalConfig.abalation_mode in ["model_only", "none"]:
            return torch.Tensor(x).type(dtype=dtype)
        else:
            mutated_tensor = tensor_mutation.tensor_mutation(torch.Tensor(x).type(dtype=dtype), environment, mutation_strategy)
            return mutated_tensor





def exe_module():
    #一定要运行时再import，否则上来就会初始化错误的模型
    from Method.Models.general_testnet_tensorflow import GeneralTFNet
    from Method.Models.general_testnet_pytorch import GeneralTorchNet

    # final_module = [edge(FromIndex=0,ToIndex=1,Operator=-1)]

    # final_module = [edge(FromIndex=0,ToIndex=1,Operator=1),
    #                 edge(FromIndex=0,ToIndex=2,Operator=2),
    #                 edge(FromIndex=0,ToIndex=3,Operator=3),
    #                 edge(FromIndex=0,ToIndex=4,Operator=4),
    #                 edge(FromIndex=0,ToIndex=5,Operator=5),
    #                 edge(FromIndex=0,ToIndex=6,Operator=6),
    #                 edge(FromIndex=0,ToIndex=7,Operator=7),
    #                 edge(FromIndex=0,ToIndex=8,Operator=8),
    #                 edge(FromIndex=0,ToIndex=9,Operator=9),
    #                 edge(FromIndex=0,ToIndex=10,Operator=10),
    #                 edge(FromIndex=0,ToIndex=11,Operator=11),
    #                 edge(FromIndex=0,ToIndex=12,Operator=12),
    #                 edge(FromIndex=0,ToIndex=13,Operator=13),
    #                 edge(FromIndex=0,ToIndex=14,Operator=14),
    #                 edge(FromIndex=1,ToIndex=2,Operator=-1),
    #                 edge(FromIndex=2,ToIndex=3,Operator=2),
    #                 edge(FromIndex=3,ToIndex=4,Operator=3),
    #                 edge(FromIndex=4,ToIndex=5,Operator=4),
    #                 edge(FromIndex=5,ToIndex=6,Operator=5),
    #                 edge(FromIndex=6,ToIndex=7,Operator=6),
    #                 edge(FromIndex=7,ToIndex=8,Operator=7),
    #                 edge(FromIndex=8,ToIndex=9,Operator=8),
    #                 edge(FromIndex=9,ToIndex=10,Operator=9),
    #                 edge(FromIndex=10,ToIndex=11,Operator=10),
    #                 edge(FromIndex=11,ToIndex=12,Operator=11),
    #                 edge(FromIndex=12,ToIndex=13,Operator=12),
    #                 edge(FromIndex=13,ToIndex=14,Operator=13),
    #                 edge(FromIndex=14,ToIndex=15,Operator=14)
    #                 ]

    final_module = GlobalConfig.final_module
    initial_module = [edge(FromIndex=0,ToIndex=1,Operator=1)]
    # initial_module = [edge(FromIndex=0,ToIndex=1,Operator=1),
    #                 edge(FromIndex=0,ToIndex=2,Operator=2),
    #                 edge(FromIndex=0,ToIndex=3,Operator=3),
    #                 edge(FromIndex=0,ToIndex=4,Operator=4),
    #                 edge(FromIndex=0,ToIndex=5,Operator=5),
    #                 edge(FromIndex=0,ToIndex=6,Operator=6),
    #                 edge(FromIndex=0,ToIndex=7,Operator=7),
    #                 edge(FromIndex=0,ToIndex=8,Operator=8),
    #                 edge(FromIndex=0,ToIndex=9,Operator=9),
    #                 edge(FromIndex=0,ToIndex=10,Operator=10),
    #                 edge(FromIndex=0,ToIndex=11,Operator=11),
    #                 edge(FromIndex=0,ToIndex=12,Operator=12),
    #                 edge(FromIndex=0,ToIndex=13,Operator=13),
    #                 edge(FromIndex=0,ToIndex=14,Operator=14),
    #                 edge(FromIndex=1,ToIndex=2,Operator=-1),
    #                 edge(FromIndex=2,ToIndex=3,Operator=2),
    #                 edge(FromIndex=3,ToIndex=4,Operator=3),
    #                 edge(FromIndex=4,ToIndex=5,Operator=4),
    #                 edge(FromIndex=5,ToIndex=6,Operator=5),
    #                 edge(FromIndex=6,ToIndex=7,Operator=6),
    #                 edge(FromIndex=7,ToIndex=8,Operator=7),
    #                 edge(FromIndex=8,ToIndex=9,Operator=8),
    #                 edge(FromIndex=9,ToIndex=10,Operator=9),
    #                 edge(FromIndex=10,ToIndex=11,Operator=10),
    #                 edge(FromIndex=11,ToIndex=12,Operator=11),
    #                 edge(FromIndex=12,ToIndex=13,Operator=12),
    #                 edge(FromIndex=13,ToIndex=14,Operator=13),
    #                 edge(FromIndex=14,ToIndex=15,Operator=14),
    #                 edge(FromIndex=15,ToIndex=16,Operator=15),
    #                 edge(FromIndex=16,ToIndex=17,Operator=16),
    #                 edge(FromIndex=17,ToIndex=18,Operator=17),
    #                 edge(FromIndex=18,ToIndex=19,Operator=18),
    #                 edge(FromIndex=19,ToIndex=20,Operator=19)
    #                 ]

    activation_types = ["relu","sigmoid","tanh","leakyrelu","prelu","elu"]
    activation = activation_types[random.randint(0,len(activation_types)-1)]
    # channels = [1,1,2,3,2,3,4,2,2,3,4,5,6,7,8,8,1,1,1,1,1,1]
    if GlobalConfig.abalation_mode in ["tensor_only","none"]:
        channels = [1,1]
    else:
        channels = GlobalConfig.channels

    #准备输入的numpy
    n, c, h, w = 0, 0, 0, 0
    input_corpus = None
    if GlobalConfig.dataset == 'random':
        n = GlobalConfig.batch
        c = GlobalConfig.c0
        h = GlobalConfig.h
        w = GlobalConfig.w

        input_corpus = np.random.randn(n, c, h, w)
    else:
        # 这两句用于处理main中os调用的错误。
        current_path = os.path.dirname(__file__)
        os.chdir(current_path)
        data = np.load('../Dataset/'+GlobalConfig.dataset+'/inputs.npz')

        # # For complete dataset
        # input_corpus = data[data.files[0]]


        #TODO For the first batch and the first channel
        input_corpus_total =data[data.files[0]]
        input_corpus = copy.deepcopy(input_corpus_total[0][0]
                                     .reshape(1, 1, input_corpus_total.shape[2],
                                              input_corpus_total.shape[3]))

    this_tensor_mutation_strategy = select_tensor_mutation_strategy()
    # TODO back
    # Handle with torch
    torch_input = get_input_tensor(input_corpus, dtype = torch.float32, environment = "pytorch", mutation_strategy = this_tensor_mutation_strategy)
    # torch_net = TorchNet(channels=channels,final_module=final_module,in_channel=c,activation_type=activation)
    if GlobalConfig.abalation_mode in ["tensor_only","none"]:
        final_module = initial_module
        GlobalConfig.final_module = initial_module
        torch_net = GeneralTorchNet(channels=channels,final_module=initial_module,in_channel=torch_input.shape[1])
    else:
        torch_net = GeneralTorchNet(channels=channels,final_module=final_module,in_channel=torch_input.shape[1])
    torch_output = torch_net(torch_input)
    torch_output_numpy = torch_output.detach().numpy()


    # Handlie with tf
    tensorflow_input = get_input_tensor(input_corpus, dtype=tf.float32, environment="tensorflow", mutation_strategy = this_tensor_mutation_strategy)
    # tensorflow_net = TFNet(channels=channels,final_module=final_module,in_channel=c, activation_type=activation)
    if GlobalConfig.abalation_mode in ["tensor_only","none"]:
        final_module = initial_module
        GlobalConfig.final_module = initial_module
        tensorflow_net = GeneralTFNet(channels=channels, final_module=initial_module, in_channel=tensorflow_input.shape[3])
    else:
        tensorflow_net = GeneralTFNet(channels=channels, final_module=final_module, in_channel=tensorflow_input.shape[3])
    tensorflow_output = tensorflow_net(tensorflow_input)
    tensorflow_output_numpy = tf.transpose(tensorflow_output, [0, 3, 1, 2]).numpy()


    # Model Converter
    # current_path = os.path.dirname(__file__)
    # os.chdir(current_path)
    saved_path = f"{GlobalConfig.absolutePath}\\TF_Model"
    export_path = f"{GlobalConfig.absolutePath}\\TFJS_Model"

    tf.saved_model.save(tensorflow_net, saved_path)
    tfjs.converters.convert_tf_saved_model(saved_path, export_path)

    # url垃圾回收机制响应过慢，删除文件后再创建会出现url冲突
    tfjs_model_name = util.model_name_generator()
    os.rename(export_path + "\\model.json", export_path + "\\" + tfjs_model_name)
    tfjs_model_url = GlobalConfig.HttpServer + tfjs_model_name

    while not os.path.exists(export_path + "\\" + tfjs_model_name):
        time.sleep(1)

    print(f"model{str(GlobalConfig.alreadyMutatetime)}_saved_ok!")

    # 核心目的：尽量让数据库包含的条目尽可能少
    # 情况1：random数据集随时变化，因此要每次都重新导入
    # 轻狂2：数据集只要导入一次就够了，所以在第0次突变时导入
    if (GlobalConfig.dataset == 'random') or (GlobalConfig.alreadyMutatetime == 0):
        input_content = np.array(tensorflow_input).flatten().tolist()
        InputHelper.insert_input_helper(util.list_to_str(tensorflow_input.shape),
                                        util.list_to_str(input_content), tfjs_model_url)
    else:
        InputHelper.set_ready(tfjs_model_url)


    # Handle with TFJS_Chrome
    p = subprocess.Popen(GlobalConfig.Command_chrome_start)

    print(util.getFinalModule_in_str())
    print("Fetch tfjs_output from Chrome")
    while ChromeHelper.get_ready_then_fetch() is None:
        time.sleep(1)

    print("tfjs_output from Chrome has been fetched!")
    chrome_output_type = ChromeHelper.get_ready_then_fetch()[0]

    # 检查chrome上是否出现了crush
    if chrome_output_type == "error":
        error_msg = ChromeHelper.get_ready_then_fetch()[2]

        # 处理输入数据库
        if GlobalConfig.dataset == "random":
            InputHelper.clear_input_helper()
        else:
            InputHelper.reset_ready()
        # 如果不要保存模型，则应该删除模型
        if GlobalConfig.save_TFJS_model == "TRUE":
            print(f"model_{GlobalConfig.alreadyMutatetime} has been saved")
        else:
            util.clear_dir(export_path)
        # 杀掉进程，把浏览器关掉
        p.kill()
        p = subprocess.Popen(GlobalConfig.Command_chrome_shut)
        # 清空数据库，减小搜索压力
        ChromeHelper.clear_chrome_finish()
        # 抛出异常
        raise Exception(error_msg)
    #TODO 没有出现错误，将输出张量保存下来
    json_tatal = ChromeHelper.get_ready_then_fetch()[1]
    tfjs_chrome_output_temp = np.array(util.json_combiner("chrome", json_tatal)).reshape(tensorflow_output.shape)
    tfjs_chrome_output = tf.convert_to_tensor(tfjs_chrome_output_temp)
    tfjs_Chrome_output_numpy = tf.transpose(tfjs_chrome_output, [0, 3, 1, 2]).numpy()
    p.kill()
    # Chrome is finish, clear for Edge
    ChromeHelper.clear_chrome_finish()


    # Handle with TFJS_Edge
    p = subprocess.Popen(GlobalConfig.Command_edge_start)
    print("Fetch tfjs_output from Edge")
    while ChromeHelper.get_ready_then_fetch() is None:
        time.sleep(1)
    print("tfjs_output from Edge has been fetched!")
    msedge_output_type = ChromeHelper.get_ready_then_fetch()[0]

    # 检查Edge上是否出现了crush
    if msedge_output_type == "error":
        error_msg = ChromeHelper.get_ready_then_fetch()[2]

        # 处理输入数据库
        if GlobalConfig.dataset == "random":
            InputHelper.clear_input_helper()
        else:
            InputHelper.reset_ready()
        # 如果不要保存模型，则应该删除模型
        if GlobalConfig.save_TFJS_model == "TRUE":
            print(f"model_{GlobalConfig.alreadyMutatetime} has been saved")
        else:
            util.clear_dir(export_path)
        # 杀掉进程，把浏览器关掉
        p.kill()
        # 清空数据库，减小搜索压力
        ChromeHelper.clear_chrome_finish()
        # 抛出异常
        raise Exception(error_msg)

    #TODO 没有出现错误，将输出张量保存下来
    json_tatal = ChromeHelper.get_ready_then_fetch()[1]
    tfjs_msedge_output_temp = np.array(util.json_combiner("msedge", json_tatal)).reshape(tensorflow_output.shape)
    tfjs_msedge_output = tf.convert_to_tensor(tfjs_msedge_output_temp)
    tfjs_Msedge_output_numpy = tf.transpose(tfjs_msedge_output, [0, 3, 1, 2]).numpy()
    p.kill()
    os.system(GlobalConfig.Command_edge_shut)
    # Edge is finish, clear for net round
    ChromeHelper.clear_chrome_finish()

    # For the three origin frameworks
    diff_numpy_1 = torch_output_numpy - tensorflow_output_numpy
    # For TFJS_Chrome
    diff_numpy_2 = tfjs_Chrome_output_numpy - torch_output_numpy
    diff_numpy_3 = tfjs_Chrome_output_numpy - tensorflow_output_numpy
    diff_numpy_4 = tfjs_Chrome_output_numpy - tfjs_Msedge_output_numpy

    # For TFJS_Msedge
    diff_numpy_5 = tfjs_Msedge_output_numpy - torch_output_numpy
    diff_numpy_6 = tfjs_Msedge_output_numpy - tensorflow_output_numpy


    diff_1_max = np.max(np.abs(diff_numpy_1))
    diff_1_mean = np.mean(np.abs(diff_numpy_1))

    diff_2_max = np.max(np.abs(diff_numpy_2))
    diff_2_mean = np.mean(np.abs(diff_numpy_2))
    diff_3_max = np.max(np.abs(diff_numpy_3))
    diff_3_mean = np.mean(np.abs(diff_numpy_3))
    diff_4_max = np.max(np.abs(diff_numpy_4))
    diff_4_mean = np.mean(np.abs(diff_numpy_4))
    diff_5_max = np.max(np.abs(diff_numpy_5))
    diff_5_mean = np.mean(np.abs(diff_numpy_5))
    diff_6_max = np.max(np.abs(diff_numpy_6))
    diff_6_mean = np.mean(np.abs(diff_numpy_6))


    avg_diff_max = (diff_1_max+diff_2_max+diff_3_max+diff_4_max
                    +diff_5_max+diff_6_max)/6.0
    avg_diff_mean = (diff_1_mean + diff_2_mean + diff_3_mean
                     + diff_4_mean + diff_5_mean + diff_6_mean
                     )/6.0
    tensor_mean = numpy.mean(input_corpus)

    # 情况一：random由于要重新导入，因此直接删除条目
    # 情况二：数据集，不用重新导入，只要设为not_ready即可
    if GlobalConfig.dataset == "random":
        InputHelper.clear_input_helper()
    else:
        InputHelper.reset_ready()

    ChromeHelper.clear_chrome_finish()
    util.clear_dir("..\\TFJS_output_storage\\Chrome_output_storage")
    # time.sleep(2)
    util.clear_dir("..\\TFJS_output_storage\\Edge_output_storage")
    print("killing...")
    # time.sleep(5)
    # p = subprocess.Popen(GlobalConfig.Command_edge_shut)
    # p.kill()

    if GlobalConfig.save_TFJS_model == "TRUE":
        print(f"model_{GlobalConfig.alreadyMutatetime} has been saved")
    else:
        util.clear_dir(export_path)

    return diff_1_max,diff_1_mean,\
           diff_2_max,diff_2_mean,\
           diff_3_max,diff_3_mean,\
           diff_4_max,diff_4_mean,\
           diff_5_max,diff_5_mean,\
           diff_6_max, diff_6_mean,\
           avg_diff_max,avg_diff_mean,this_tensor_mutation_strategy
# print(exe_module())


