import networkx as nx
import csv
from DataStruct.flatOperatorMap import FlatOperatorMap
from DataStruct.operation import Operator
from DataStruct.globalConfig import GlobalConfig
from DataStruct.edge import edge

# Calculate the average edit distance, that is, the average edit distance between two adjacent graphs


def parse_Type(Type):
    if Type in GlobalConfig.basicOps:
        res = GlobalConfig.basicOps.index(Type) - 1
    else:
        res = -1
    return res

def edge_match(edge1, edge2):
    if edge1['operator'] == edge2['operator']:
        return True
    else:
        return False




if __name__ == '__main__':
    global in_degree

    # source dir
    file_path = '../result.csv'

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        model_structrues = [row[16] for row in reader]

    # 取最后K组数据
    K = 10000
    model_structrues = model_structrues[len(model_structrues) - K:]
    Maps = []
    Round = 0
    model_structrues = model_structrues[1:]
    for model_structure in model_structrues:

        Round += 1

        edge_infos = model_structure.split("  ")
        G = nx.DiGraph()
        point_num = 0

        for edge_info in edge_infos:
            elements = edge_info.split(",")
            from_id = 0
            to_id = 0
            op = 0
            for element in elements:
                all_config = element.split(" ")
                for each_config in all_config:
                    if each_config.__contains__("from"):
                        key = int(each_config[each_config.find(":") + 1:])
                        from_id = key
                        if key > point_num:
                            point_num = key
                    if each_config.__contains__("operator"):
                        key = int(parse_Type(each_config[each_config.find(":") + 1:]))
                        op = key
                    else:
                        if each_config.__contains__("to"):
                            key = int(each_config[each_config.find(":") + 1:])
                            to_id = key
                            if key > point_num:
                                point_num = key
            G.add_edge(v_of_edge = from_id, u_of_edge= to_id, operator = op)

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
