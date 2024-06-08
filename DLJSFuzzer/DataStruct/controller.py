# coding=utf-8
import copy

from Method.asyncTournamentSelect import asyncTournamentSelect
from DataStruct.globalConfig import GlobalConfig
from Method.mutation import mutation
from Method.genetypeCompare import genetypeCompare

def check(g, p):
    for i in range(p.size):
        check_g = p.genetypes[i]
        if genetypeCompare(g, check_g):
            return False
        else:
            continue
    return True
class Controller:
    a=0
    def __init__(self):
        return
    def excute(self):
        # TODO
        list=asyncTournamentSelect(GlobalConfig.P)
        for g in list:
            #只加入突变出的新结构，已有结构不加入
            while True:
                new_g = copy.deepcopy(g)
                mutation(new_g)
                # 没有相同结构就加入种群
                if check(new_g,GlobalConfig.P) == True:
                    GlobalConfig.Q.push(new_g)
                    break
        return