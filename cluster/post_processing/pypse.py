

import numpy as np
from queue import Queue


def get_dis(sv1, sv2):
    return np.linalg.norm(sv1 - sv2)


def pse_py(text, similarity_vectors, label, kernel_num,border_num, dis_threshold=0.8):
    print("border",border_num,"kernel",kernel_num)

    pred = np.zeros(text.shape)
    queue = Queue(maxsize=0)
    points = np.array(np.where(label > 0)).transpose((1, 0))


    for point_idx in range(points.shape[0]):
        y, x = points[point_idx, 0], points[point_idx, 1]
        label_value = label[y, x]
        queue.put((y, x, label_value))      #kernel
        pred[y, x] = label_value


    # 计算kernel的值
    d = {}
    for i in range(kernel_num):
        kernel_idx = label == i
        kernel_similarity_vector = similarity_vectors[kernel_idx].mean(0)  # 4
        d[i] = kernel_similarity_vector

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    kernal = text.copy()
    while not queue.empty():
        (y, x, label_value) = queue.get()
        cur_kernel_sv = d[label_value]
        for j in range(4):
            tmpx = x + dx[j]
            tmpy = y + dy[j]
            if tmpx < 0 or tmpy >= kernal.shape[0] or tmpy < 0 or tmpx >= kernal.shape[1]:
                continue
            if kernal[tmpy, tmpx] == 0 or pred[tmpy, tmpx] > 0:
                continue
            if np.linalg.norm(similarity_vectors[tmpy, tmpx] - cur_kernel_sv) >= dis_threshold:
                continue
            queue.put((tmpy, tmpx, label_value))
            pred[tmpy, tmpx] = label_value
    return pred
