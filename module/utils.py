# -*- coding: utf-8 -*-
# @Date    : 2020/9/1 9:41 下午
# @Author  : Du Jing
# @FileName: utils
# ---- Description ----
#

import tensorflow as tf


def params_usage(train_variables):
    total = 0
    prompt = []
    for v in train_variables:
        shape = v.get_shape()
        cnt = 1
        for dim in shape:
            cnt *= dim.value
        prompt.append('{} with shape {} has {}'.format(v.name, shape, cnt))
        print(prompt[-1])
        total += cnt
    prompt.append('totaling {}'.format(total))
    print(prompt[-1])
    return '\n'.join(prompt)


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))