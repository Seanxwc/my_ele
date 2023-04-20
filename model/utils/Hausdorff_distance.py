# """HausdorffDistance."""
#
# from collections import abc
# from abc import ABCMeta
# from scipy.ndimage import morphology
# import numpy as np
# from mindspore.common.tensor import Tensor
# from mindspore._checkparam import Validator as validator
# from .metric import Metric
#
# import numpy as np
# from mindspore import Tensor
# from mindspore.nn.metrics import HausdorffDistance
#
#
# class _ROISpatialData(metaclass=ABCMeta):
#     # 产生感兴趣区域（ROI）。支持裁剪和空间数据。应提供空间的中心和大小，如果没有，则必须提供ROI的起点和终点坐标。
#     def __init__(self, roi_center=None, roi_size=None, roi_start=None, roi_end=None):
#
#         if roi_center is not None and roi_size is not None:
#             roi_center = np.asarray(roi_center, dtype=np.int16)
#             roi_size = np.asarray(roi_size, dtype=np.int16)
#             self.roi_start = np.maximum(roi_center - np.floor_divide(roi_size, 2), 0)
#             self.roi_end = np.maximum(self.roi_start + roi_size, self.roi_start)
#         else:
#             if roi_start is None or roi_end is None:
#                 raise ValueError("Please provide the center coordinates, size or start coordinates and end coordinates"
#                                  " of ROI.")
#             # ROI起始坐标
#             self.roi_start = np.maximum(np.asarray(roi_start, dtype=np.int16), 0)
#             # ROI终点坐标
#             self.roi_end = np.maximum(np.asarray(roi_end, dtype=np.int16), self.roi_start)
#
#     def __call__(self, data):
#         sd = min(len(self.roi_start), len(self.roi_end), len(data.shape[1:]))
#         slices = [slice(None)] + [slice(s, e) for s, e in zip(self.roi_start[:sd], self.roi_end[:sd])]
#         return data[tuple(slices)]
#
#
# class HausdorffDistance(Metric):
#     def __init__(self, distance_metric="euclidean", percentile=None, directed=False, crop=True):
#         super(HausdorffDistance, self).__init__()
#         string_list = ["euclidean", "chessboard", "taxicab"]
#         distance_metric = validator.check_value_type("distance_metric", distance_metric, [str])
#         # 计算Hausdorff距离的参数，支持欧氏、"chessboard"、"taxicab"三种测量方法。
#         self.distance_metric = validator.check_string(distance_metric, string_list, "distance_metric")
#         # 表示最大距离分位数，取值范围为0-100，它表示的是计算步骤4中，选取的距离能覆盖距离的百分比
#         self.percentile = percentile if percentile is None else validator.check_value_type("percentile",
#                                                                                            # 可分为定向和非定向Hausdorff距离，默认为非定向Hausdorff距离，指定percentile参数得到Hausdorff距离的百分位数。                                                                                 percentile, [float])
#                                                                                            self.directed = directed if directed is None else validator.check_value_type(
#             "directed", directed, [bool])
#         self.crop = crop if crop is None else validator.check_value_type("crop", crop, [bool])
#         self.clear()
#
#     def _is_tuple_rep(self, tup, dim):
#         # 通过缩短或重复输入返回包含dim值的元组。
#         result = None
#         if not self._is_iterable_sequence(tup):
#             result = (tup,) * dim
#         elif len(tup) == dim:
#             result = tuple(tup)
#
#         if result is None:
#             raise ValueError(f"Sequence must have length {dim}, but got {len(tup)}.")
#
#         return result
#
#     def _is_tuple(self, inputs):
#         # 判断是否是一个元组
#         if not self._is_iterable_sequence(inputs):
#             inputs = (inputs,)
#
#         return tuple(inputs)
#
#     def _is_iterable_sequence(self, inputs):
#         if isinstance(inputs, Tensor):
#             return int(inputs.dim()) > 0
#         return isinstance(inputs, abc.Iterable) and not isinstance(inputs, str)
#
#     def _create_space_bounding_box(self, image, func=lambda x: x > 0, channel_indices=None, margin=0):
#         data = image[[*(self._is_tuple(channel_indices))]] if channel_indices is not None else image
#         data = np.any(func(data), axis=0)
#         nonzero_idx = np.nonzero(data)
#         margin = self._is_tuple_rep(margin, data.ndim)
#
#         box_start = list()
#         box_end = list()
#         for i in range(data.ndim):
#             if nonzero_idx[i].size <= 0:
#                 raise ValueError("did not find nonzero index at the spatial dim {}".format(i))
#             box_start.append(max(0, np.min(nonzero_idx[i]) - margin[i]))
#             box_end.append(min(data.shape[i], np.max(nonzero_idx[i]) + margin[i] + 1))
#         return box_start, box_end
#
#     def _calculate_percent_hausdorff_distance(self, y_pred_edges, y_edges):
#
#         surface_distance = self._get_surface_distance(y_pred_edges, y_edges)
#
#         if surface_distance.shape == (0,):
#             return np.inf
#
#         if not self.percentile:
#             return surface_distance.max()
#         # self.percentile表示最大距离分位数，取值范围为0-100，它表示的是计算步骤4中，选取的距离能覆盖距离的百分比
#         if 0 <= self.percentile <= 100:
#             return np.percentile(surface_distance, self.percentile)
#
#         raise ValueError(f"percentile should be a value between 0 and 100, get {self.percentile}.")
#
#     def _get_surface_distance(self, y_pred_edges, y_edges):
#         # 使用欧式方法求表面距离
#         if not np.any(y_pred_edges):
#             return np.array([])
#
#         if not np.any(y_edges):
#             dis = np.inf * np.ones_like(y_edges)
#         else:
#             if self.distance_metric == "euclidean":
#                 dis = morphology.distance_transform_edt(~y_edges)
#             elif self.distance_metric == "chessboard" or self.distance_metric == "taxicab":
#                 dis = morphology.distance_transform_cdt(~y_edges, metric=self.distance_metric)
#
#         surface_distance = dis[y_pred_edges]
#
#         return surface_distance
#
#     def clear(self):
#         """清楚历史数据"""
#         self.y_pred_edges = 0
#         self.y_edges = 0
#         self._is_update = False
#
#     def update(self, *inputs):
#         """
#         更新输入数据
#         """
#         if len(inputs) != 3:
#             raise ValueError('HausdorffDistance need 3 inputs (y_pred, y, label), but got {}'.format(len(inputs)))
#         y_pred = self._convert_data(inputs[0])
#         y = self._convert_data(inputs[1])
#         label_idx = inputs[2]
#
#         if y_pred.size == 0 or y_pred.shape != y.shape:
#             raise ValueError("Labelfields should have the same shape, but got {}, {}".format(y_pred.shape, y.shape))
#
#         y_pred = (y_pred == label_idx) if y_pred.dtype is not bool else y_pred
#         y = (y == label_idx) if y.dtype is not bool else y
#
#         res1, res2 = None, None
#         if self.crop:
#             if not np.any(y_pred | y):
#                 res1 = np.zeros_like(y_pred)
#                 res2 = np.zeros_like(y)
#
#             y_pred, y = np.expand_dims(y_pred, 0), np.expand_dims(y, 0)
#             box_start, box_end = self._create_space_bounding_box(y_pred | y)
#             cropper = _ROISpatialData(roi_start=box_start, roi_end=box_end)
#             y_pred, y = np.squeeze(cropper(y_pred)), np.squeeze(cropper(y))
#
#         self.y_pred_edges = morphology.binary_erosion(y_pred) ^ y_pred if res1 is None else res1
#         self.y_edges = morphology.binary_erosion(y) ^ y if res2 is None else res2
#         self._is_update = True
#
#     def eval(self):
#         """
#         计算定向或者非定向的Hausdorff distance.
#         """
#         # 要先执行clear操作
#         if self._is_update is False:
#             raise RuntimeError('Call the update method before calling eval.')
#
#         # 计算A到B的距离
#         hd = self._calculate_percent_hausdorff_distance(self.y_pred_edges, self.y_edges)
#         # 计算定向的豪斯多夫
#         if self.directed:
#             return hd
#         # 计算非定向的豪斯多夫
#         hd2 = self._calculate_percent_hausdorff_distance(self.y_edges, self.y_pred_edges)
#         # 如果计算的是定向的，那直接返回hd，如果是计算非定向，那hd和hd2谁大返回谁
#         return max(hd, hd2)
#
#
#
#
#
#
#
#
#
# if __name__ == "__main__":
#     x = Tensor(np.array([[3, 0, 1], [1, 3, 0], [1, 0, 2]]))
#     y = Tensor(np.array([[0, 2, 1], [1, 2, 1], [0, 0, 1]]))
#     metric = HausdorffDistance()
#     metric.clear()
#     metric.update(x, y, 0)
#
#     x1 = Tensor(np.array([[3, 0, 1], [1, 3, 0], [1, 0, 2]]))
#     y1 = Tensor(np.array([[0, 2, 1], [1, 2, 1], [0, 0, 1]]))
#     metric.update(x1, y1, 0)
#
#     distance = metric.eval()
#     print(distance)


import torch


def torch2D_Hausdorff_distance(x, y):  # Input be like (Batch,width,height)
    x = x.float()
    y = y.float()
    distance_matrix = torch.cdist(x, y, p=2)  # p=2 means Euclidean Distance

    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

    value = torch.cat((value1, value2), dim=1)

    return value.max(1)[0]


if __name__ == "__main__":
    # u = torch.Tensor([[[1.0, 0.0],
    #                    [0.0, 1.0],
    #                    [-1.0, 0.0],
    #                    [0.0, -1.0]],
    #                   [[1.0, 0.0],
    #                    [0.0, 1.0],
    #                    [-1.0, 0.0],
    #                    [0.0, -1.0]],
    #                   [[2.0, 0.0],
    #                    [0.0, 2.0],
    #                    [-2.0, 0.0],
    #                    [0.0, -4.0]]])
    #
    # v = torch.Tensor([[[0.0, 0.0],
    #                    [0.0, 2.0],
    #                    [-2.0, 0.0],
    #                    [0.0, -3.0]],
    #                   [[2.0, 0.0],
    #                    [0.0, 2.0],
    #                    [-2.0, 0.0],
    #                    [0.0, -4.0]],
    #                   [[1.0, 0.0],
    #                    [0.0, 1.0],
    #                    [-1.0, 0.0],
    #                    [0.0, -1.0]]])
    #
    # print("Input shape is (B,W,H):", u.shape, v.shape)
    # HD = torch2D_Hausdorff_distance(u, v)
    # print("Hausdorff Distance is:", HD)
    # print(HD.mean())


    u = torch.Tensor([[[[1.0, 0.0],
                       [0.0, 1.0],
                       [-1.0, 0.0],
                       [0.0, -1.0]],
                      [[1.0, 0.0],
                       [0.0, 1.0],
                       [-1.0, 0.0],
                       [0.0, -1.0]],
                      [[2.0, 0.0],
                       [0.0, 2.0],
                       [-2.0, 0.0],
                       [0.0, -4.0]]]])

    v = torch.Tensor([[[2.0, 0.0],
                       [0.0, 2.0],
                       [-2.0, 0.0],
                       [0.0, -4.0]]])

    print("Input shape is (B,W,H):", u.shape, v.shape)
    HD = torch2D_Hausdorff_distance(u, v)
    print("Hausdorff Distance is:", HD)
    print(HD.mean())