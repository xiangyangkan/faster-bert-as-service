"""模型前处理和后处理

"""

import os
import numpy as np
import multiprocessing
import ujson
from datetime import datetime


def input_tokenizer(input_str, tokenizer, max_seq_length):
    # 预处理字符
    tokens_a = tokenizer.tokenize(input_str)
    # 如果超过长度限制，则进行截断
    if len(input_str) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_length = len(input_ids)
    input_mask = [1] * input_length
    segment_ids = [0] * input_length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    return input_ids, input_mask, segment_ids


def format_input(input_str, tokenizer, max_seq_length, engine_mode=True, enable_debug=False):
    """
    输入数据格式化
    :param input_str: 输入的句子
    :param tokenizer: 分词器
    :param max_seq_length: 最大序列长度
    :param engine_mode: 推理模式
    :param enable_debug: 是否启用DEBUG

    :return:
    """
    input_ids, input_mask, segment_ids = input_tokenizer(input_str, tokenizer, max_seq_length)
    if engine_mode:
        model_data = {
            "inputs": [
                {
                    "name": "segment_ids",
                    "datatype": "INT32",
                    "data": segment_ids,
                    "shape": [1, max_seq_length],
                },
                {
                    "name": "input_ids",
                    "datatype": "INT32",
                    "data": input_ids,
                    "shape": [1, max_seq_length],
                },
                {
                    "name": "input_mask",
                    "datatype": "INT32",
                    "data": input_mask,
                    "shape": [1, max_seq_length],
                }],
            "outputs": [
                {
                    "name": "output",
                    "parameters": {
                        "binary_data": True
                    }
                }
            ]
        }
        if enable_debug:
            for name in ["last_transformer_output"]:
                model_data["outputs"].append({"name": name})
    else:
        model_data = {
            "inputs": [
                {
                    "name": "label_ids",
                    "datatype": "INT32",
                    "data": [0],
                    "shape": [1, 1],
                },
                {
                    "name": "segment_ids",
                    "datatype": "INT32",
                    "data": segment_ids,
                    "shape": [1, max_seq_length],
                },
                {
                    "name": "input_ids",
                    "datatype": "INT32",
                    "data": input_ids,
                    "shape": [1, max_seq_length],
                },
                {
                    "name": "input_mask",
                    "datatype": "INT32",
                    "data": input_mask,
                    "shape": [1, max_seq_length],
                }
            ]
        }
    return model_data


def triton_post(input_str, tokenizer, max_seq_length, url, session, engine_mode=True, enable_debug=False):
    input_str = format_input(input_str, tokenizer, max_seq_length, engine_mode, enable_debug)
    if engine_mode:
        # engine 版本
        binary_res = session.post(url, ujson.dumps(input_str)).content
        if enable_debug:
            return binary_res
        else:
            return np.frombuffer(binary_res[-3072:], dtype='<f4')
    else:
        # savemodel 版本
        res = session.post(url, ujson.dumps({'inputs': input_str})).json()
        embedding_output = res.get('outputs')[0].get('data')
        return embedding_output
