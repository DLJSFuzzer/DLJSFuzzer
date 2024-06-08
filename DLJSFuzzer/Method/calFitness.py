# coding=utf-8

from .flatMap import toFlatMap

def calFitness(g):
    #todo
    toFlatMap(g)
    #一定要用到的时候再import，否则会初始化错误的值
    from Method.module_executor import exe_module
    torch_tf_max_diff, torch_tf_mean_diff, \
    tfjschrome_torch_max_diff, tfjschrome_torch_mean_diff, \
    tfjschrome_tf_max_diff, tfjschrome_tf_mean_diff, \
    tfjschrome_tfjsmsedge_max_diff, tfjschrome_tfjsmsedge_mean_diff, \
    tfjsmsedge_torch_max_diff, tfjsmsedge_torch_mean_diff, \
    tfjsmsedge_tf_max_diff, tfjsmsedge_tf_mean_diff, \
    avg_max_diff, avg_mean_diff, this_tensor_mutation_strategy = exe_module()
    return torch_tf_max_diff, torch_tf_mean_diff, \
            tfjschrome_torch_max_diff, tfjschrome_torch_mean_diff, \
            tfjschrome_tf_max_diff, tfjschrome_tf_mean_diff, \
            tfjschrome_tfjsmsedge_max_diff, tfjschrome_tfjsmsedge_mean_diff, \
            tfjsmsedge_torch_max_diff, tfjsmsedge_torch_mean_diff, \
            tfjsmsedge_tf_max_diff, tfjsmsedge_tf_mean_diff, \
            avg_max_diff, avg_mean_diff, this_tensor_mutation_strategy
