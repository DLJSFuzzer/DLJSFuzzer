# 本组件用于除DLJSFuzzer外全部方法的数据生成
import copy
import os
import queue
from DataStruct.globalConfig import GlobalConfig
import numpy as np

class GeneralDataGenerator:
    # 注：全部变量只在predoo中被使用
    # 扰动量级
    disturbance = []
    # predoo输入队列
    q = None
    # 语料库初始大小
    corpus_size = 0
    # 能允许的误差上界
    threshold = 0.0

    def __init__(self):
        # 处理化predoo的参数
        self.disturbance = [0.0001,0.000001,0.00000001]
        self.q = queue.Queue()
        self.corpus_size = 100
        for i in range(0,self.corpus_size):
            n = GlobalConfig.batch
            c = GlobalConfig.c0
            h = GlobalConfig.h
            w = GlobalConfig.w
            x = np.random.randn(n, c, h, w)
            self.q.put(x)
        self.threshold = 1e-6
        return
    def getData(self):
        if GlobalConfig.method=="predoo":
            if self.q.empty() == False:
                return self.q.get()
            else:
                return []
        elif GlobalConfig.method in ["cradle","muffin","lemon"]:
            if GlobalConfig.dataset == 'random':
                n = GlobalConfig.batch
                c = GlobalConfig.c0
                h = GlobalConfig.h
                w = GlobalConfig.w

                return np.random.randn(n, c, h, w)
            else:
                # 这两句用于处理main中os调用的错误。
                current_path = os.path.dirname(__file__)
                os.chdir(current_path)
                data = np.load('../Dataset/' + GlobalConfig.dataset + '/inputs.npz')

                # # For complete dataset
                # input_corpus = data[data.files[0]]

                # TODO For the first batch and the first channel
                input_corpus_total = data[data.files[0]]
                return copy.deepcopy(input_corpus_total[0][0]
                                             .reshape(1, 1, input_corpus_total.shape[2],
                                                      input_corpus_total.shape[3]))
    def feedback(self,data,error):
        # 只有predoo方法会用到
        # 如果误差大于上界，就在队列中加入几个扰动后的数据，否则什么都不做。
        if error > self.threshold:
            n = GlobalConfig.batch
            c = GlobalConfig.c0
            h = GlobalConfig.h
            w = GlobalConfig.w
            for each_disturbance in self.disturbance:
                disturbance_array = each_disturbance * np.ones((n, c, h, w), np.float64)
                generated_array = copy.deepcopy(data) + disturbance_array
                self.q.put(copy.deepcopy(generated_array))
        return