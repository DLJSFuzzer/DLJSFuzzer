import json
import copy
import os
import networkx as nx
from DataStruct.flatOperatorMap import FlatOperatorMap
from DataStruct.operation import Operator
from DataStruct.globalConfig import GlobalConfig
from DataStruct.edge import edge

def edge_match(edge1, edge2):
    if edge1['operator'] == edge2['operator']:
        return True
    else:
        return False
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
                branches.append(edge(target, j, Operation))
    # #打印各点的channels
    # print("各点的channels为：")
    # for i in range(len(f.channels)):
    #     print(i)
    #     print(f.channels[i])
    return

def parse_Type(Type):
    if Type in GlobalConfig.basicOps:
        res = GlobalConfig.basicOps.index(Type) - 1
    else:
        res = -1
    return res

if __name__ == '__main__':
    global mainPath
    global branches

    file_path = '../Muffin_model/'
    total_edit_distance = 0
    Maps = []
    model_num = 0
    for root, dirs, files in os.walk(file_path):
        for dir in dirs:
            if not os.path.exists(file_path + dir + '/models/model.json'):
                continue
            InputPath = open(file_path + dir + '/models/model.json', encoding="utf-8")
            json_dic = json.load(InputPath)
            model = json_dic['model_structure']
            first_list = json_dic['input_id_list']
            if len(first_list) > 1:
                print("非法格式，模型存在多个输入！")
                continue
            last_list = json_dic['output_id_list']
            if len(last_list) > 1:
                print("非法格式，模型存在多个输出！")
                continue
            # First_Layer 表示输入层
            # Last_layer 表示最后的输出层
            first_layer = first_list[0]
            last_layer = last_list[0]

            f = FlatOperatorMap(size=last_layer + 1)
            G = nx.DiGraph()
            for x in range(f.size):
                for y in range(f.size):
                    f.Map[x][y] = Operator(0, 0)

            model = json_dic['model_structure']
            for layer in range(first_layer, last_layer + 1):
                id = str(layer)
                this_layer_inf = model[id]
                from_ids = this_layer_inf['pre_layers']
                to_id = layer
                Type = this_layer_inf["type"]
                if Type in ['concatenate', 'average', 'maximum', 'minimum', 'add', 'subtract', 'multiply', 'dot']:
                    if id != str(last_layer):
                        Type = 'concatenate'
                    else:
                        Type = 'identity'

                if Type == "input_object" or Type == 'concatenate':
                    continue
                for from_id in from_ids:
                    from_layer = model[str(from_id)]
                    if from_layer['type'] in ['concatenate', 'average', 'maximum', 'minimum', 'add', 'subtract',
                                              'multiply', 'dot']:
                        for true_from_id in from_layer["pre_layers"]:
                            # f.Map[true_from_id][to_id] = Type
                            f.Map[true_from_id][to_id] = Operator(0, parse_Type(Type))
                            G.add_edge(v_of_edge=true_from_id, u_of_edge=to_id, operator=parse_Type(Type))

                    else:
                        # f.Map[from_id][to_id] = Type
                        f.Map[from_id][to_id] = Operator(0, parse_Type(Type))
                        G.add_edge(v_of_edge=from_id, u_of_edge=to_id, operator=parse_Type(Type))
            Maps.append(G)
    edit_distances = []
    for i in range(0, len(Maps)-1):
        j = i + 1
        print("from",i,"to",j,":")
        # delete too large maps
        # if Maps[i].number_of_edges() > 100:
        #     print("This Map is too large, delete it.")
        #     continue
        distance = nx.graph_edit_distance(Maps[i], Maps[j], edge_match = edge_match, timeout = 1)
        print(distance)
        if distance != None:
            edit_distances.append(distance)
    average_edit_distance = sum(edit_distances)/len(edit_distances)
    max_edit_distance = max(edit_distances)
    min_edit_distance = min(edit_distances)
    print("The average edit distance is:")
    print(average_edit_distance)
    print("The max edit distance is:")
    print(max_edit_distance)
    print("The min edit distance is:")
    print(min_edit_distance)
