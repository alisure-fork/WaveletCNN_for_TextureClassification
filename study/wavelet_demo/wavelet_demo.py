import os

import cv2
import numpy as np
from alisuretool.Tools import Tools
from wavelet_haar import wavelet_transform


level = 4
name = "demo.png"
input_image = "./input/{}".format(name)
result_path = Tools.new_dir("./result/{}".format(os.path.splitext(name)[0]))

image_data = cv2.imread(input_image)
print(image_data.shape)


# 获得小波数据
wavelets = []
for i in range(image_data.shape[-1]):
    wavelet = wavelet_transform(image_data[:, :, i], level)
    wavelets.append(wavelet)


# 画图展示
result_image_datas_all = []
for wavelet_i, wavelet_c in enumerate(wavelets):
    result_image_datas = []
    for i in range(level):
        shape_i = wavelet_c["LL_{}".format(i + 1)].shape
        result_image_data = np.zeros(shape=(shape_i[0] * 2, shape_i[1] * 2), dtype=np.uint8)
        result_image_data[: shape_i[0], : shape_i[1]] = np.asarray(wavelet_c["LL_{}".format(i + 1)], dtype=np.uint8)
        result_image_data[shape_i[0]: , : shape_i[1]] = np.asarray(wavelet_c["LH_{}".format(i + 1)], dtype=np.uint8)
        result_image_data[: shape_i[0], shape_i[1]: ] = np.asarray(wavelet_c["HL_{}".format(i + 1)], dtype=np.uint8)
        result_image_data[shape_i[0]: , shape_i[1]: ] = np.asarray(wavelet_c["HH_{}".format(i + 1)], dtype=np.uint8)
        cv2.imwrite(filename=os.path.join(result_path, "{}_{}.bmp".format(wavelet_i, i + 1)), img=result_image_data)
        result_image_datas.append(result_image_data)
        pass
    result_image_datas_all.append(result_image_datas)
    pass

for result_image_datas_i, result_image_datas in enumerate(result_image_datas_all):
    result_image_data = result_image_datas[0]
    for i in range(1, level):
        shape_i = result_image_datas[i].shape
        result_image_data[: shape_i[0], : shape_i[1]] = result_image_datas[i]
        pass
    cv2.imwrite(filename=os.path.join(result_path, "{}.bmp".format(result_image_datas_i)), img=result_image_data)
    pass

print()
