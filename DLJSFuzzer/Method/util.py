# coding=utf-8

from DataStruct.globalConfig import GlobalConfig
import os
import shutil
from datetime import date
import time
import json


def getChannels_in_str():
    result = ""
    for i in range(len(GlobalConfig.channels)):
        result = result+str(i)+":"
        result = result+str(GlobalConfig.channels[i])+" "
    return result


def getFinalModule_in_str():
    result = ""
    for eachEdge in GlobalConfig.final_module:
        result = result+"from:"+str(eachEdge.fromIndex)+" to:"+str(eachEdge.toIndex)+\
                 " operator:"+str(eachEdge.operator)+"  "
    return result


def getFinalModule_in_str_formal():
    result = ""
    for eachEdge in GlobalConfig.final_module:
        result = result+"from:"+str(eachEdge.fromIndex)+" to:"+str(eachEdge.toIndex)+\
                 " operator:"+GlobalConfig.basicOps[eachEdge.operator + 1]+"  "
    return result


def clear_dir(dir_path):
    dir_for_clear = dir_path
    for file_name in os.listdir(dir_for_clear):
        file_path = f"{dir_for_clear}\\{file_name}"
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            continue


def model_name_generator():
    t = time.localtime()
    result = f"model{GlobalConfig.alreadyMutatetime}_{str(date.today()).replace('-','_')}" \
             f"_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.json"
    return result


def list_to_str(list):
    thisStr = [str(item) for item in list]
    return ",".join(thisStr)


def str_to_list(this_string):
    result = [float(this_ele) for this_ele in this_string.split(',')]
    return result


def json_combiner(browser, json_total):
    current_path = os.path.dirname(__file__)
    os.chdir(current_path)
    res = []
    for i in range(json_total):
        if browser == "chrome":
            this_json = f"../TFJS_output_storage/Chrome_output_storage/output_save{str(i + 1)}.json"
        else:
            this_json = f"../TFJS_output_storage/Edge_output_storage/output_save{str(i + 1)}.json"

        while not os.path.exists(this_json):
            time.sleep(0.1)

        this_list = []

        try:
            with open(this_json, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                this_dict = data['content']
                for j in range(len(this_dict)):
                    this_list.append(this_dict[str(j)])
        except Exception as e:
            print(f"visiting {this_json} denied, try again")
            time.sleep(0.5)
            with open(this_json, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                this_dict = data['content']
                for j in range(len(this_dict)):
                    this_list.append(this_dict[str(j)])

        res = res + this_list
    return res







