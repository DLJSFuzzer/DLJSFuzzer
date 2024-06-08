# 本方法用于除DLJSFuzzer外全部方法的模型结构生成
import copy
import os

from DataStruct.globalConfig import GlobalConfig
from DataStruct.flatOperatorMap import FlatOperatorMap
from DataStruct.operation import Operator

def parse_Type(Type):
    if Type in GlobalConfig.basicOps:
        res = GlobalConfig.basicOps.index(Type) - 1
    else:
        res = -1
    return res

def Decode(type, ch):
    res = 1
    same_channel_operators = [-1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14]
    if type in same_channel_operators:
        res = ch

    return res

def search_zero(in_degree, size):
    for i in range(size):
        if in_degree[i] == 0:
            return i
    return -1

def decodeChannel(f):
    global mainPath
    global branches
    #注：输入类型为flatOperaotrMap

    #先把f.chanels扩大
    f.channels = [0]*f.size
    f.channels[0] = 1
    in_degree = [0]*f.size
    for j in range(f.size):
        for i in range(f.size):
            if f.Map[i][j].m != 0:
                in_degree[j] += 1

    #最多拓扑f.size轮
    for times in range(f.size):
        # 找到入度为0的点
        target = search_zero(in_degree, f.size)
        if target < 0:
            print("Error! Circle exits!")
            return

        # mainPath.append(target + 1);
        # length = len(mainPath)
        # if length > 1:
        #     FromIndex = mainPath[length - 2] - 1
        #     ToIndex = target
        #     Operation = f.Map[FromIndex][ToIndex].m
        #     branches.append(edge(FromIndex,ToIndex,Operation))

            # for toIndex in range(f.size):
                # if toIndex == ToIndex:
                #     continue
                # if f.Map[FromIndex][toIndex].m != 0:
                #     Operation = f.Map[FromIndex][toIndex].m
                #     branches.append(edge(FromIndex, toIndex, Operation))


        in_degree[target] = -1
        for j in range(f.size):
            if f.Map[target][j].m != 0:

                # #用于引导和测试模型的专用语句 mark
                # if f.Map[target][j].m != 4:
                #     f.Map[target][j].m = 1;

                in_degree[j] -= 1
                f.channels[j] += Decode(f.Map[target][j].m, f.channels[target])
                Operation = f.Map[target][j].m
    # #打印各点的channels
    # print("各点的channels为：")
    # for i in range(len(f.channels)):
    #     print(i)
    #     print(f.channels[i])
    return

class edge:
    fromIndex = 0
    toIndex = 0
    # 为了方便格式转化，且涉及的操作仅为基本操作，故只保留操作号，不保留层号
    operator = 0
    index = ""
    def __init__(self, FromIndex, ToIndex, Operator):
        self.fromIndex = FromIndex
        self.toIndex = ToIndex
        self.operator = Operator

class GeneralStructureGenerator:
    model_csv = None
    def __init__(self):
        #这两句防止路径出错
        current_path = os.path.dirname(__file__)
        os.chdir(current_path)
        self.model_csv = open('../model.csv', encoding='utf-8')
    def getModelStructure(self):
        model_structure = None
        channels = []
        this_model_str = self.model_csv.readline()
        if this_model_str == "":
            model_structure = None
            channels = []
            return model_structure,channels
        else:
            operators = this_model_str.split(" ")
            operators[0] = operators[0][1:]
            node_num = int(operators[-2].split(',')[1][3:])
            this_model = FlatOperatorMap(size=node_num + 1)
            final_model = []
            for x in range(this_model.size):
                for y in range(this_model.size):
                    this_model.Map[x][y] = Operator(0, 0)
            for each_operator in operators:
                if each_operator == '"\n':
                    continue
                eachstr = each_operator.split(',')
                fromIndex = int(eachstr[0][5:])
                toIndex = int(eachstr[1][3:])
                type = parse_Type(eachstr[2][9:])
                this_model.Map[fromIndex][toIndex] = Operator(0, type)
                final_model.append(edge(FromIndex=fromIndex, ToIndex=toIndex, Operator=type))
            decodeChannel(this_model)

            model_structure = copy.deepcopy(final_model)
            channels = copy.deepcopy(this_model.channels)

            # 最后记录结果的时候要用到，因此需要写回到GlobalConfig里面
            GlobalConfig.channels = channels
            GlobalConfig.final_module = model_structure

            return model_structure,channels