# -*- coding: utf-8 -*-
# @Date    : 2020/9/1 9:33 下午
# @Author  : Du Jing
# @FileName: __init__.py
# ---- Description ----
#

'''
模型说明 ———— 借鉴了Transformer的结构：
    - 输入进行标签平滑
    - 位置按照正余弦编码，编码向量拼接入输入特征向量
    - 6个子层，包括多头注意力层和前馈层，每层的输出进行dropout，且每层使用残差连接
'''