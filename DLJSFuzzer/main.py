# coding=utf-8
import copy
import datetime
import os

from DataStruct.population import Population
from DataStruct.genetypeQueue import GenetypeQueue
from DataStruct.globalConfig import GlobalConfig
from DataStruct.worker import Worker
from DataStruct.controller import Controller
from DataStruct.generalDataGenerator import GeneralDataGenerator
from DataStruct.generalStructureGenerator import GeneralStructureGenerator
from Method.initialize import initialize
from Method.util import getFinalModule_in_str,getChannels_in_str
from Method.general_model_executor import general_exe_module
import csv
import json

from Method import util
from Database import db_helper

# 本方法用于在globalConfig中初始化，包含DLJSFuzzer需要的参数和其它方法需要的参数。
def globalInit():
    # step1:配置globalConfig
    print("正在初始化globalConfig")
    out = open(file='./' + 'result.csv' , mode='w', newline='')
    writer = csv.writer(out,delimiter = ",")
    GlobalConfig.data_generator = GeneralDataGenerator()
    GlobalConfig.structure_generator = GeneralStructureGenerator()
    GlobalConfig.N = 0
    GlobalConfig.alreadyMutatetime = 0
    GlobalConfig.flatOperatorMaps = []
    GlobalConfig.resGenetype = []
    GlobalConfig.P = Population()
    GlobalConfig.Q = GenetypeQueue()
    GlobalConfig.final_module = []
    GlobalConfig.channels = []
    GlobalConfig.outFile = out
    GlobalConfig.writer = writer
    writer.writerow(["No","torch_tf_max_diff","torch_tf_mean_diff",
                     "tfjschrome_torch_max_diff","tfjschrome_torch_mean_diff",
                     "tfjschrome_tf_max_diff","tfjschrome_tf_mean_diff",
                     "tfjschrome_tfjsmsedge_max_diff","tfjschrome_tfjsmsedge_mean_diff",
                     "tfjsmsedge_torch_max_diff","tfjsmsedge_torch_mean_diff",
                     "tfjsmsedge_tf_max_diff","tfjsmsedge_tf_mean_diff",
                     "avg_max_diff","avg_mean_diff",
                     "channels", "model", "fail_time"])
    out.flush()



# TODO 清空，防止出现问题，这里好像用不了相对路径！
absolutePath = GlobalConfig.absolutePath
util.clear_dir(f"{absolutePath}\\TFJS_output_storage\\Chrome_output_storage")
util.clear_dir(f"{absolutePath}\\TFJS_output_storage\\Edge_output_storage")
util.clear_dir(f"{absolutePath}\\TFJS_Model")
util.clear_dir(f"{absolutePath}\\Crush_logs")
db_helper.clear_db()

# 初始化
globalInit()

# 执行方法
# case1:DLJSFuzzer
if GlobalConfig.method == "dljsfuzzer":
    print("正在初始化种群")
    initialize(GlobalConfig.P)
    print("种群初始化完成")
    print("开始构建controller节点")
    controller = Controller()
    print("controller节点构建完成")
    print("开始构建worker节点")
    worker = Worker()
    print("worker节点构建完成")

    #主流程
    t = 0
    avg = 0
    # start_time = 0
    # finish_time = 0
    # overall = 0

    print("开始进行突变")

    while t < GlobalConfig.maxMutateTime:
        # if t >= 9900 and t < 10000:
        #     start_time = datetime.datetime.now()

        controller.excute()

        # if t >= 9900 and t < 10000:
        #     end_time = datetime.datetime.now()
        #     overall += (end_time - start_time).microseconds
        #     print(start_time)
        #     print(end_time)
        #     print(overall)
        try:

            # if t >= GlobalConfig.maxMutateTime - 50:
            #     start_time = datetime.datetime.now()

            torch_tf_max_diff, torch_tf_mean_diff, \
            tfjschrome_torch_max_diff, tfjschrome_torch_mean_diff, \
            tfjschrome_tf_max_diff, tfjschrome_tf_mean_diff, \
            tfjschrome_tfjsmsedge_max_diff, tfjschrome_tfjsmsedge_mean_diff, \
            tfjsmsedge_torch_max_diff, tfjsmsedge_torch_mean_diff, \
            tfjsmsedge_tf_max_diff, tfjsmsedge_tf_mean_diff, \
            avg_max_diff, avg_mean_diff = worker.excute()
            print("第" + str(t) + "轮已经完成")

            # 写入结果
            GlobalConfig.writer.writerow([str(t),
                                          str(torch_tf_max_diff),
                                          str(torch_tf_mean_diff),
                                          str(tfjschrome_torch_max_diff),
                                          str(tfjschrome_torch_mean_diff),
                                          str(tfjschrome_tf_max_diff),
                                          str(tfjschrome_tf_mean_diff),
                                          str(tfjschrome_tfjsmsedge_max_diff),
                                          str(tfjschrome_tfjsmsedge_mean_diff),
                                          str(tfjsmsedge_torch_max_diff),
                                          str(tfjsmsedge_torch_mean_diff),
                                          str(tfjsmsedge_tf_max_diff),
                                          str(tfjsmsedge_tf_mean_diff),
                                          str(avg_max_diff),
                                          str(avg_mean_diff),
                                          getChannels_in_str(),
                                          getFinalModule_in_str(),
                                          str(GlobalConfig.fail_time)])

        except Exception as e:
            GlobalConfig.fail_time += 1
            record_path = f"./Crush_logs/crush_log_{str(GlobalConfig.fail_time)}.json"
            model_info = util.getFinalModule_in_str_formal()
            js_content = {"model_inf": model_info, "error_message": str(e)}
            js_str = json.dumps(js_content, indent=2)

            current_path = os.path.dirname(__file__)
            os.chdir(current_path)
            with open(record_path, 'w', encoding="utf-8") as file:
                file.write(js_str)
                file.close()

            print("本轮突变失败！")
            # print(e)
        t = t + 1
        GlobalConfig.alreadyMutatetime = t


    # print("avg runtime:", avg / 50)
    # sec = overall // 1000000
    # micro = overall % 1000000
    # ms = str(micro)
    # while len(ms) < 6:
    #     ms = "0" + ms
    # print(f"overall_time: {sec}.{ms} sec")

    #最后的筛选
    # while(len(GlobalConfig.resGenetype) < GlobalConfig.resultNum):
    #     controller.excute()
    #     thisg=GlobalConfig.Q.pop()
    #     GlobalConfig.resGenetype.append(thisg)

#case2: predoo,cradle,muffin,lemon
#注：其它方法datagenerator不会返回空数组，当model_structure为None时停止执行。
#而运行predoo时，model.csv中应该有足够多的from：0，to：1，index：对应的操作，当data_generator返回空数组时停止执行。
elif GlobalConfig.method in ["predoo","cradle","muffin","lemon","graphfuzz","gandalf"]:
    # t 表示总轮次（包含异常和正常）
    t = 0
    while True:
        # 生成输入数据
        data = GlobalConfig.data_generator.getData()
        # 如果方法是predoo，返回空数组时停止执行。
        if GlobalConfig.method == "predoo" and len(data) == 0:
            break
        # 生成模型结构，生成的是类似原先final_module的结构体
        model_structure,channels = GlobalConfig.structure_generator.getModelStructure()
        #如果没有model了,model_structure为None就停止执行
        if model_structure == None:
            break
        try:
            torch_tf_max_diff, torch_tf_mean_diff,\
            tfjschrome_torch_max_diff, tfjschrome_torch_mean_diff,\
            tfjschrome_tf_max_diff, tfjschrome_tf_mean_diff,\
            tfjschrome_tfjsmsedge_max_diff, tfjschrome_tfjsmsedge_mean_diff,\
            tfjsmsedge_torch_max_diff, tfjsmsedge_torch_mean_diff,\
            tfjsmsedge_tf_max_diff, tfjsmsedge_tf_mean_diff,\
            avg_max_diff, avg_mean_diff, tensor_mean = general_exe_module(data=data,model_structure=model_structure,channels=channels)
            # 如果没有bug则写入结果，如果有bug则直接跳到异常处理，不写入结果且不反馈。
            GlobalConfig.writer.writerow([str(t),
                                          str(torch_tf_max_diff),
                                          str(torch_tf_mean_diff),
                                          str(tfjschrome_torch_max_diff),
                                          str(tfjschrome_torch_mean_diff),
                                          str(tfjschrome_tf_max_diff),
                                          str(tfjschrome_tf_mean_diff),
                                          str(tfjschrome_tfjsmsedge_max_diff),
                                          str(tfjschrome_tfjsmsedge_mean_diff),
                                          str(tfjsmsedge_torch_max_diff),
                                          str(tfjsmsedge_torch_mean_diff),
                                          str(tfjsmsedge_tf_max_diff),
                                          str(tfjsmsedge_tf_mean_diff),
                                          str(avg_max_diff),
                                          str(avg_mean_diff),
                                          getChannels_in_str(),
                                          getFinalModule_in_str(),
                                          str(GlobalConfig.fail_time)])
            GlobalConfig.outFile.flush()
            # 如果无异常则feedback
            if GlobalConfig.error_cal_mode == "max":
                print(avg_max_diff)
                GlobalConfig.data_generator.feedback(data,avg_max_diff)
            elif GlobalConfig.error_cal_mode == "mean":
                print(avg_mean_diff)
                GlobalConfig.data_generator.feedback(copy.deepcopy(data),avg_mean_diff)
        except Exception as e:
            GlobalConfig.fail_time += 1
            current_path = os.path.dirname(__file__)
            os.chdir(current_path)
            record_path = f"./Crush_logs/crush_log_{str(GlobalConfig.fail_time)}.json"
            model_info = util.getFinalModule_in_str_formal()
            js_content = {"model_inf": model_info, "error_message": str(e)}
            js_str = json.dumps(js_content, indent=2)

            with open(record_path, 'w', encoding="utf-8") as file:
                file.write(js_str)
                file.close()
        t = t + 1

