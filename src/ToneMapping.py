#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import math

def simple_bilateral_approximation(data, sigmaS, sigmaR):
    h, w = data.shape
    data_min, data_max = np.min(data), np.max(data)
    
    # level grid
    grid_h = int(np.ceil(h / sigmaS))
    grid_w = int(np.ceil(w / sigmaS))
    grid_d = int(np.ceil((data_max - data_min) / sigmaR))
    
    grid_data = np.zeros((grid_h, grid_w, grid_d), dtype=np.float32)
    grid_weight = np.zeros((grid_h, grid_w, grid_d), dtype=np.float32)
    
    # 將影像累加到網格上
    for i in range(h):
        for j in range(w):
            # 計算網格索引
            y = int(i / sigmaS)
            x = int(j / sigmaS)
            z = int((data[i, j] - data_min) / sigmaR)
            # 防止索引超出範圍
            y = min(y, grid_h - 1)
            x = min(x, grid_w - 1)
            z = min(z, grid_d - 1)
            
            grid_data[y, x, z] += data[i, j]
            grid_weight[y, x, z] += 1.0
    
    # 分別對每個深度層進行 2D 高斯平滑
    for z in range(grid_d):
        grid_data[:, :, z] = cv2.GaussianBlur(grid_data[:, :, z], (7, 7), sigmaX=0)
        grid_weight[:, :, z] = cv2.GaussianBlur(grid_weight[:, :, z], (7, 7), sigmaX=0)
    
    # devideed by zero
    grid_weight[grid_weight == 0] = 1e-5
    grid_filtered = grid_data / grid_weight
    
    # 用最近鄰法進行上採樣回原影像尺寸
    output = np.zeros_like(data, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            y = int(i / sigmaS)
            x = int(j / sigmaS)
            z = int((data[i, j] - data_min) / sigmaR)
            y = min(y, grid_h - 1)
            x = min(x, grid_w - 1)
            z = min(z, grid_d - 1)
            output[i, j] = grid_filtered[y, x, z]
    
    return output

def main_process(img):
    height, width = np.shape(img)[:2]
    img = img / img.max()
    epsilon = 1e-8
    # avoid 0
    val = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2] + epsilon
    log_val = np.log(val)
    
    space_sigma = min(width, height) / 500  
    range_sigma = (np.amax(log_val) - np.amin(log_val)) / 1 
    imgg = simple_bilateral_approximation(log_val, space_sigma, range_sigma)

    # base 的影響程度
    factor1 = 1
    # detail 的影響程度
    factor2 = 1
    # log -> exponential 復原
    reconstructed_luminance = np.exp(factor1 * imgg + factor2 * (log_val - imgg))
    reconstructed_luminance = reconstructed_luminance.astype('float32')
    
    # 復原到原始 rgb 通道
    out = np.zeros(img.shape)
    out[:, :, 0] = img[:, :, 0] * (reconstructed_luminance / val)
    out[:, :, 1] = img[:, :, 1] * (reconstructed_luminance / val)
    out[:, :, 2] = img[:, :, 2] * (reconstructed_luminance / val)
    
    outt = np.clip(np.power(out, 1.0 / 1.8) * 255, 0, 255)
    outt = outt.astype('uint8')
    cv2.imwrite('tonemap_hand_made_32_1.png', outt)

# run
import sys
temp = cv2.imread(sys.argv[1])
main_process(temp)

