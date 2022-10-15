import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def euclidean_distances(x, y, squared=True):
    x_square = np.expand_dims(np.einsum('ij, ij->i', x, x), axis=1)
    y_square = np.expand_dims(np.einsum('ij, ij->i', y, y), axis=0)

    distances = np.dot(x, y.T)
    distances *= -2
    distances += x_square
    distances += y_square
    np.maximum(distances, 0, distances)
    np.sqrt(distances, distances)
    return distances

def partition_arg_topK(matrix, K, axis=0):
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis ==0 :
        row_index = np.arange(matrix.shape[1-axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1-axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

#change the FEATURE_ROOT to where the pkl file is
FEATURE_ROOT = '.'
PHOTO_FEATURE = os.path.join(FEATURE_ROOT, 'photo.pkl')
SKETCH_FEATURE = os.path.join(FEATURE_ROOT, 'sketch.pkl')
photo_data = pickle.load(open(PHOTO_FEATURE, 'rb'))
sketch_data = pickle.load(open(SKETCH_FEATURE, 'rb'))

# the feature is extracted by our full model and the name is the origin one of the offical dataset of ChairV2
photo_name, photo_feature = photo_data['name'], photo_data['feature']
sketch_name, sketch_feature = sketch_data['name'], sketch_data['feature']

distances = euclidean_distances(sketch_feature, photo_feature)

gt_list = list(range(len(photo_name)))
gt_list = np.asarray(gt_list)
test_item = len(gt_list)
gt_list = np.reshape(gt_list, (test_item, 1))
topK = partition_arg_topK(distances, 10, axis=1)

recall_1 = topK[:, 0, None] == gt_list
print(recall_1)