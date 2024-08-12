import numpy as np
import cv2
from tqdm import tqdm
from utils import RGB_to_gray

def Gaussian_filter(filter_size):
    sigma_space = 75
    pad_size = filter_size // 2
    gaussian_weight = np.zeros((filter_size, filter_size))
    for i in range(filter_size):
        for j in range(filter_size):
            gaussian_weight[i,j] = -((i-pad_size)**2 + (j-pad_size)**2) / (2 * sigma_space**2)
    return gaussian_weight

def Bilateral_filter(image, filter_size):
    pad_size = filter_size // 2
    sigma_color = 75
    gaussian_weight = Gaussian_filter(filter_size)
    bilateral_weight = np.zeros((filter_size, filter_size))
    for i in range(filter_size):
        for j in range(filter_size):
            bilateral_weight[i,j] = - ((image[i,j] - image[pad_size,pad_size])**2) / (2 * sigma_color**2)
    total_weight = np.exp(gaussian_weight + bilateral_weight)

    return total_weight / np.sum(total_weight)

def Mid_filter(disparity, image, depth, filter_size=3):
    image = RGB_to_gray(image) 

    height, width = disparity.shape
    pad_size = filter_size // 2
    pad_width = ((pad_size, pad_size), (pad_size, pad_size))
    padded_image = np.pad(image, pad_width, mode='constant')

    histogram = np.zeros((height, width, depth)).astype(np.float32)
    bin_disparity =  np.zeros((height, width, depth))

    for d in range(depth):
        bin_disparity[:, :, d] = (disparity == d)

    pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0,0))
    padded_bin_disparity = np.pad(bin_disparity, pad_width, mode='constant')

    for h in tqdm(range(height)):
        for w in range(width):
            for d in range(depth):
                binary_filter = padded_bin_disparity[h:h+filter_size, w:w+filter_size, d]
                histogram[h, w, d] = np.sum(binary_filter * Bilateral_filter(padded_image[h:h+filter_size, w:w+filter_size], filter_size))
    
    for d in range(depth):
        print_img = histogram[:, :, d]
        cv2.imshow('histogram', print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()    
    
    aggregated_disparity = np.argmax(histogram, axis=2)

    print_img = aggregated_disparity.astype(np.uint8) * int(255 / depth)
    cv2.imshow('aggregated_disparity',print_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    return aggregated_disparity