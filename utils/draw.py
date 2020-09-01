# -*- coding: utf-8 -*-
# @Date    : 2020/9/1 9:54 下午
# @Author  : Du Jing
# @FileName: draw
# ---- Description ----
#
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'plot_confusion_matrix'
]

def plot_confusion_matrix(matrix, classes, title="Confusion Matrix", save_name="Confusion Matrix.png"):
    plt.figure(figsize=(7, 5.5), dpi=300)
    # 在混淆矩阵中每格的概率值
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    thresh = matrix.max() / 2.
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = matrix[y_val][x_val]
        plt.text(x_val, y_val, "%0.2f" % (c,), fontsize=15, va='center', ha='center', color="white" if matrix[x_val, y_val] > thresh else "black")

    plt.imshow(matrix, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('True label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(save_name, format='png')
    plt.show()