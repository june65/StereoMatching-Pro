import numpy as np
import cv2
from utils import RGB_to_gray, RGB_to_CIELab
from tqdm import tqdm

def ASW(left_image, right_image, depth, kernel_size=3, specular=False):

    left_image_CIELab = RGB_to_CIELab(left_image) 
    right_image_CIELab = RGB_to_CIELab(right_image)  
    left_CV, right_CV, left_specular_mask, right_specular_mask =  cost_volume(left_image, right_image, depth, kernel_size, specular)
    return disparity_map(left_CV, right_CV, left_image_CIELab, right_image_CIELab, depth, kernel_size, specular, left_specular_mask, right_specular_mask)

def chromaticity(image,x,y,direction):
    Lamd = (image[y,x,:] - image[y+direction[0],x+direction[1],:]).astype(np.float64)
    flag = np.abs(np.sum(image[y,x,:]) - np.sum(image[y+direction[0],x+direction[1],:]))
    if flag!=0:
        Lamd /= flag
    return Lamd

#VERSION1
def cost_volume(left_image, right_image, depth, kernel_size, specular=False):

    height, width, _ = left_image.shape
    left_specular_mask = np.zeros((height, width, 1)).astype(np.float64)
    right_specular_mask = np.zeros((height, width, 1)).astype(np.float64)

    if (specular):
        left_specular_masks = np.zeros((height, width)).astype(np.float64)
        threshold = 3
        directions = [(1, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)]
        for index, direction in enumerate(directions):
            for w in range(1,width-2):
                for h in range(1,height-2):
                    epsilon = chromaticity(left_image,w,h,direction) - chromaticity(left_image,w+direction[0],h+direction[1],direction)
                    if np.abs(np.sum(epsilon)) > threshold:
                        left_specular_masks[h,w] += 1
                        left_specular_masks[h+direction[1],w+direction[0]] += 1
                        left_specular_masks[h-direction[1],w-direction[0]] += 1

        left_specular_mask[left_specular_masks >= 12] = 1
        
        right_specular_masks = np.zeros((height, width)).astype(np.float64)
        for index, direction in enumerate(directions):
            for w in range(1,width-2):
                for h in range(1,height-2):
                    epsilon = chromaticity(right_image,w,h,direction) - chromaticity(right_image,w+direction[0],h+direction[1],direction)
                    if np.abs(np.sum(epsilon)) > threshold:
                        right_specular_masks[h,w] += 1
                        right_specular_masks[h+direction[1],w+direction[0]] += 1
                        right_specular_masks[h-direction[1],w-direction[0]] += 1

        right_specular_mask[right_specular_masks >= 12] = 1
        
        print_img = right_specular_masks.astype(np.uint8) * int(255/24)
        cv2.imshow('specular_mask_sum',print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print_img = right_specular_mask.astype(np.uint8) * int(255)
        cv2.imshow('specular_mask',print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    left_image = RGB_to_gray(left_image) 
    right_image = RGB_to_gray(right_image)
    
    left_costvolume = np.full((height, width ,depth), np.inf)
    right_costvolume = np.full((height, width ,depth) ,np.inf)

    for d in range(depth):
        for w in range(width):
            if w+d < width:
                left_costvolume[:,w,d] = np.abs(left_image[:,w] - right_image[:,w+d])
            if w-d >=0:
                right_costvolume[:,w,d] = np.abs(left_image[:,w-d] - right_image[:,w])

    return left_costvolume, right_costvolume, left_specular_mask, right_specular_mask


def disparity_map(left_costvolume, right_costvolume, left_image_CIELab, right_image_CIELab, depth, kernel_size, specular=False, left_specular_mask=None, right_specular_mask=None):

    height, width, _ = left_image_CIELab.shape
    pad_size = kernel_size // 2
    
    pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0,0))
    padded_left_disparity = np.pad(left_costvolume, pad_width, mode='constant')
    padded_right_disparity = np.pad(right_costvolume, pad_width, mode='constant')
    
    padded_left_image= np.pad(left_image_CIELab, pad_width, mode='constant')
    padded_right_image = np.pad(right_image_CIELab, pad_width, mode='constant')

    left_disparity_conv = np.full((height, width, depth), np.inf)
    right_disparity_conv =  np.full((height, width, depth) ,np.inf)

    weight_g_kernel = np.zeros((kernel_size, kernel_size)) 
    for i in range(kernel_size):
        for j in range(kernel_size):
            distance = np.sqrt((i - pad_size)**2 + (j - pad_size)**2)
            weight_g_kernel[i, j] = distance
    
    r_c = 7
    r_p = 36 

    if (specular):
        r_s = 36 

    for h in tqdm(range(height)):
        for w in range(width):
            for d in range(depth):
                if w+d < width:
                    weight_C_left = np.sqrt(np.sum(np.square(padded_left_image[h:h+kernel_size, w:w+kernel_size, :] - np.full((kernel_size, kernel_size, 3), padded_left_image[h+pad_size, w+pad_size, :])),axis=2))
                    weight_C_right_d = np.sqrt(np.sum(np.square(padded_right_image[h:h+kernel_size, w+d:w+d+kernel_size, :] - np.full((kernel_size, kernel_size, 3), padded_right_image[h+pad_size, w+d+pad_size, :])),axis=2)) 
                    if (specular):
                        weight_S = left_specular_mask[h,w] + right_specular_mask[h,w+d] 
                        weight = np.exp(-(((weight_C_left + weight_C_right_d) / r_c) + ((weight_g_kernel * 2) / r_p) + (weight_S / r_s)))

                    else:
                        weight = np.exp(-(((weight_C_left + weight_C_right_d) / r_c) + ((weight_g_kernel * 2) / r_p)))
                    weight /= np.sum(weight)
                    left_disparity_conv[h, w, d] = np.sum(padded_left_disparity[h:h+kernel_size, w:w+kernel_size, d] * weight)
                if w-d >= 0:
                    weight_C_left_d = np.sqrt(np.sum(np.square(padded_left_image[h:h+kernel_size, w-d:w-d+kernel_size, :] - np.full((kernel_size, kernel_size, 3), padded_left_image[h+pad_size, w-d+pad_size, :])),axis=2))
                    weight_C_right = np.sqrt(np.sum(np.square(padded_right_image[h:h+kernel_size, w:w+kernel_size, :] - np.full((kernel_size, kernel_size, 3), padded_right_image[h+pad_size, w+pad_size, :])),axis=2))   
                    if (specular):
                        weight_S = right_specular_mask[h,w] + left_specular_mask[h,w-d] 
                        weight = np.exp(-(((weight_C_left + weight_C_right_d) / r_c) + ((weight_g_kernel * 2) / r_p) + (weight_S / r_s)))
                    else:
                        weight = np.exp(-(((weight_C_left_d + weight_C_right) / r_c) + ((weight_g_kernel * 2) / r_p)))
                    weight /= np.sum(weight)
                    right_disparity_conv[h, w, d] = np.sum(padded_right_disparity[h:h+kernel_size, w:w+kernel_size, d] * weight)

    left_disparity = np.argmin(left_disparity_conv, axis=2)
    right_disparity = np.argmin(right_disparity_conv, axis=2)
    '''
    print_img = left_disparity.astype(np.uint8) * int(255 / depth)
    cv2.imshow('right_disparity_map',print_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print_img = right_disparity.astype(np.uint8) * int(255 / depth)
    cv2.imshow('left_disparity_map',print_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    '''
    return left_disparity, right_disparity, left_disparity_conv

'''
#VERSION2

def weight(image, X1, Y1, X2, Y2):
    k = 1
    r_c = 7
    r_p = 36
    del_c = np.sqrt(np.sum(np.square(image[X1,Y1,:] - image[X2,Y2,:])))
    del_g = np.sqrt((X1-X2)**2 + (Y1-Y2)**2)
    return k * np.exp(-(del_c / r_c + del_g / r_p))

def cost_volume(left_image, right_image, left_image_CIELab, right_image_CIELab, depth, kernel_size):
    
    height, width = left_image.shape
    left_costvolume = np.full((height, width, depth), np.inf)
    right_costvolume = np.full((height, width, depth) ,np.inf)

    kernel_vectors = []
    for i in range(-kernel_size//2+1, kernel_size//2+1):
        for j in range(-kernel_size//2+1, kernel_size//2+1):
            kernel_vectors.append((i,j))

    for d in tqdm(range(depth)):
        for w in range(width):
            if w+d < width:
                for h in range(height):
                    weight_sum = 0.0
                    weight_cost_sum = 0.0
                    for (i,j) in kernel_vectors:
                        if h+i < height and w+d+j < width and h+i >= 0 and  w+j >= 0:
                            weight_flag = weight(left_image_CIELab,h,w,h+i,w+j) * weight(right_image_CIELab,h,w+d,h+i,w+d+j)
                            cost_flag = np.abs(left_image[h+i,w+j] - right_image[h+i,w+d+j])
                            weight_sum += weight_flag
                            weight_cost_sum += cost_flag * weight_flag
                    left_costvolume[h,w,d] = weight_cost_sum / weight_sum

            if w-d >= 0:
                for h in range(height):
                    weight_sum = 0.0
                    weight_cost_sum = 0.0
                    for (i,j) in kernel_vectors: 
                        if h+i < height and w+j < width and h+i >= 0 and  w-d+j >= 0:
                            weight_flag = weight(left_image_CIELab,h,w-d,h+i,w-d+j) * weight(right_image_CIELab,h,w,h+i,w+j)
                            cost_flag = np.abs(left_image[h+i,w-d+j] - right_image[h+i,w+j])
                            weight_sum += weight_flag
                            weight_cost_sum += cost_flag * weight_flag
                    right_costvolume[h,w,d] = weight_cost_sum / weight_sum

    # left_costvolume = left_costvolume.mean(axis=2)
    # right_costvolume = right_costvolume.mean(axis=2)

    return left_costvolume, right_costvolume


def disparity_map(left_costvolume, right_costvolume, depth, kernel_size):
    
    left_disparity = np.argmin(left_costvolume, axis=2)
    right_disparity = np.argmin(right_costvolume, axis=2)

    print_img = left_disparity.astype(np.uint8) * int(255 / depth)
    cv2.imshow('right_disparity_map',print_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print_img = right_disparity.astype(np.uint8) * int(255 / depth)
    cv2.imshow('left_disparity_map',print_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    return left_disparity, right_disparity
'''