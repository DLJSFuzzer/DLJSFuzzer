import time

import numpy
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from gevent import pywsgi
import numpy as np
from Database import Chrome_Helper_Mapping as chromeHelper
from Database import Input_Helper_Mapping as inputHelper
from Method import util

app = Flask(__name__)

@app.route('/getInput', methods=['get'])
@cross_origin(support_credentials=True)
def provideInput():
    print("get_GetReq")
    waiting_flag = True
    while inputHelper.check_ready_then_fetch() is None:
        if waiting_flag:
            print("waiting")
            waiting_flag = False
        time.sleep(0.01)

    print("ready_to_return")
    temp_input = inputHelper.check_ready_then_fetch()[0]
    # a_numpy = np.random.randn(shape[0], shape[1], shape[2], shape[3])
    ret = {
            'Shape': util.str_to_list(temp_input[0]),
            'Input': util.str_to_list(temp_input[1]),
            'Model_Url': temp_input[2]
        }
    print("Get_Handled")
    return ret


@app.route('/sendOutput', methods=['post'])
@cross_origin(support_credentials=True)
def sendOutput():
    print("get_PostReq")
    chrome_output = request.json
    ret_type = chrome_output['ret_type']
    if ret_type == "tensor":
        json_total = chrome_output['files_total']
        chromeHelper.insert_chrome_helper_tensor(ret_type, json_total)
    else:
        error_stack = chrome_output['err_msg']
        chromeHelper.insert_chrome_helper_error(ret_type, error_stack)
    print("Post_Handled")
    return "ok"


@app.route("/")
def hello_world():
    return "Hello, World!"


if __name__ == '__main__':
    app.run(port=5000)
