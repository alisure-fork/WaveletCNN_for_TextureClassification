import numpy as np


def wavelet_transform_y(img):
    row, col = img.shape[:2]
    size = row / 2

    img_even = img[1::2]
    img_odd = img[0::2]
    if len(img_even) != len(img_odd):
        img_odd = img_odd[:-1]
    c = (img_even + img_odd) / 2.

    d = abs(img_odd - img_even)

    return size, c, d


def wavelet_transform_x(img):
    tmp = np.fliplr(img.T)
    size, dst_L, dst_H = wavelet_transform_y(tmp)
    dst_L = np.flipud(dst_L.T)
    dst_H = np.flipud(dst_H.T)
    return size, dst_L, dst_H


def wavelet_transform(img, n=1):
    row, col = img.shape[:2]

    wavelets = {}

    roi = img[0:row, 0:col]
    for i in range(0, n):
        y_size, wavelet_l, wavelet_h = wavelet_transform_y(roi)

        x_size, wavelet_ll, wavelet_lh = wavelet_transform_x(wavelet_l)
        wavelets["LL_" + str(i + 1)] = wavelet_ll
        wavelets["LH_" + str(i + 1)] = wavelet_lh

        x_size, wavelet_hl, wavelet_hh = wavelet_transform_x(wavelet_h)
        wavelets["HL_" + str(i + 1)] = wavelet_hl
        wavelets["HH_" + str(i + 1)] = wavelet_hh

        roi = wavelet_ll
        pass

    return wavelets
