# 注：本方法用于运行除DLJSFuzzer外的全部模型
# encoding = 'utf-8'
import copy
import json
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

from Database import Input_Helper_Mapping as InputHelper
from Database import Chrome_Helper_Mapping as ChromeHelper
from Database import Msedge_Helper_Mapping as MsedgeHelper

def get_input_tensor(x, dtype, environment):
    if environment=="tensorflow":
        tensor_NCHW=tf.convert_to_tensor(x, dtype=dtype)
        tensor_NHWC=tf.transpose(tensor_NCHW,[0,2,3,1])
        return tensor_NHWC
    if environment=="pytorch":
        return torch.Tensor(x).type(dtype=dtype)

def general_exe_module(data,model_structure,channels):
    #一定要运行时再import，否则上来就会初始化错误的模型
    from Method.Models.general_testnet_pytorch import GeneralTorchNet
    from Method.Models.general_testnet_tensorflow import GeneralTFNet

    final_module = model_structure
    input_corpus = data
    channels = channels

    GlobalConfig.batch = input_corpus.shape[0]
    GlobalConfig.c0 = input_corpus.shape[1]
    GlobalConfig.h = input_corpus.shape[2]
    GlobalConfig.w = input_corpus.shape[3]
    n = GlobalConfig.batch
    c = GlobalConfig.c0
    h = GlobalConfig.h
    w = GlobalConfig.w

    # Handle with torch
    torch_input = get_input_tensor(input_corpus, dtype = torch.float32, environment = "pytorch")
    torch_net = GeneralTorchNet(channels=channels,final_module=final_module,in_channel=c)
    torch_output = torch_net(torch_input)
    torch_output_numpy = torch_output.detach().numpy()

    # Handlie with tf
    tensorflow_input = get_input_tensor(input_corpus, dtype=tf.float32, environment="tensorflow")
    tensorflow_net = GeneralTFNet(channels=channels,final_module=final_module,in_channel=c)
    tensorflow_output = tensorflow_net(tensorflow_input)
    tensorflow_output_numpy = tf.transpose(tensorflow_output, [0, 3, 1, 2]).numpy()

    # JS part begin

    # Model Converter
    # current_path = os.path.dirname(__file__)
    # os.chdir(current_path)
    saved_path = f"{GlobalConfig.absolutePath}\\TF_Model"
    export_path = f"{GlobalConfig.absolutePath}\\TFJS_Model"


    tf.saved_model.save(tensorflow_net, saved_path)
    #模型大小不超过30MB(一定要指定shard的大小，否则会导致参数加载不全的bug)
    tfjs.converters.convert_tf_saved_model(saved_path, export_path , weight_shard_size_bytes=30000000)

    # #为了修复参数加载不全的bug，需要将group1-shard1of1.bin改成.shard,同时调整model.json中的path后面的路径的后缀。
    # os.rename(export_path + "\\group1-shard1of1.bin", export_path + "\\group1-shard1of1.shard")
    # f = open(export_path + "\\model.json", 'r')
    # params = json.load(f)
    # params['weightsManifest'][0]['paths'] = ['group1-shard1of1.shard']
    # f.flush()
    # f.close()
    # f = open(export_path + "\\model.json", 'w')
    # json.dump(params,f)
    # f.flush()
    # f.close()
    # time.sleep(10)

    # url垃圾回收机制响应过慢，删除文件后再创建会出现url冲突，因此要重新命名以防止url冲突。
    tfjs_model_name = util.model_name_generator()
    os.rename(export_path + "\\model.json", export_path + "\\" + tfjs_model_name)
    tfjs_model_url = GlobalConfig.HttpServer + tfjs_model_name

    while not os.path.exists(export_path + "\\" + tfjs_model_name):
        time.sleep(1)

    print(f"model{str(GlobalConfig.alreadyMutatetime)}_saved_ok!")

    # 将输入数据导入数据库
    input_content = np.array(tensorflow_input).flatten().tolist()
    InputHelper.insert_input_helper(util.list_to_str(tensorflow_input.shape),
                                    util.list_to_str(input_content), tfjs_model_url)

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
        p.kill()
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
    # p.kill()
    #这里有个bug，用os杀进程吧
    os.system(GlobalConfig.Command_edge_shut)
    # p = subprocess.Popen(GlobalConfig.Command_edge_shut)
    # p.kill()
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

    # 防止数据库中数据过多，删除条目
    InputHelper.clear_input_helper()

    ChromeHelper.clear_chrome_finish()
    util.clear_dir("..\\TFJS_output_storage\\Chrome_output_storage")
    time.sleep(1)
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
           avg_diff_max,avg_diff_mean,tensor_mean