import numpy as np
import cv2

def my_bilateral_filter(img, diameter, sigma_color, sigma_space):
    """
    Implement bilateral filtering on the 2D array img, diameter is the window diameter, sigma_color is the color space standard deviation,
    sigma_space is the spatial standard deviation.
    """
    radius = diameter // 2
    height, width = img.shape
    padded = np.pad(img, pad_width=radius, mode='reflect')
    filtered = np.zeros_like(img)
    
    x, y = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    spatial_weight = np.exp(-(x**2 + y**2) / (2 * sigma_space**2))
    
    for i in range(height):
        for j in range(width):
            region = padded[i:i+diameter, j:j+diameter]
            intensity_diff = region - img[i, j]
            intensity_weight = np.exp(-(intensity_diff**2) / (2 * sigma_color**2))
            weight = spatial_weight * intensity_weight
            filtered[i, j] = np.sum(weight * region) / np.sum(weight)
    return filtered

def mapping(img):
    height, width = np.shape(img)[:2]
    img = img / img.max()
    epsilon = 1e-6
    val = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2] + epsilon
    log_val = np.log(val)
    
    # argument
    space_sigma = min(width, height) / 16
    range_sigma = (np.amax(log_val) - np.amin(log_val)) / 10
    
    imgg = my_bilateral_filter(log_val.astype(np.float32), diameter=9, sigma_color=range_sigma, sigma_space=space_sigma)
    
    gamma = 1.5
    # base + detail = main_channel
    main_channel = np.exp(gamma * imgg + (log_val - imgg))
    main_channel = main_channel.astype('float32')
    
    out = np.zeros(img.shape)
    out[:, :, 0] = img[:, :, 0] * (main_channel / val)
    out[:, :, 1] = img[:, :, 1] * (main_channel / val)
    out[:, :, 2] = img[:, :, 2] * (main_channel / val)
    
    # 提升亮度
    gain = 50
    out = np.clip(out * gain, 0, 1)
    
    # gamma correction
    outt = np.clip(np.power(out, 1.0 / 5) * 255, 0, 255)
    # out_corrected = np.where(out <= 0.0031308, 12.92 * out, 1.055 * np.power(out, 1/4) - 0.055)
    # outt = np.clip(out_corrected * 255, 0, 255)
    outt = outt.astype('uint8')
    # cv2.imwrite('tonemap_custom.png', outt)
    return outt

# if __name__ == "__main__":
#     # 載入 圖片
#     temp = cv2.imread("output.hdr", cv2.IMREAD_UNCHANGED)
#     if temp is None:
#         print("影像讀取失敗，請確認路徑是否正確。")
#     else:
#         mapping(temp)
