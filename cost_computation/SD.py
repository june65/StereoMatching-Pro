import numpy as np
import cv2
from utils import RGB_to_gray

def SD(left_image, right_image, min_depth, max_depth, disparity_print):

    left_image = RGB_to_gray(left_image)
    right_image = RGB_to_gray(right_image)

    left_CV, right_CV =  cost_volume(left_image, right_image, min_depth, max_depth)
    return disparity_map(left_CV, right_CV, min_depth, max_depth, disparity_print)

def cost_volume(left_image, right_image, min_depth, max_depth):

    height, width = left_image.shape
    left_costvolume = np.full((height, width, max_depth), np.inf).astype(np.float32)
    right_costvolume = np.full((height, width, max_depth), np.inf).astype(np.float32)

    for d in range(min_depth, max_depth):
        for w in range(width):
            if w+d < width:
                left_costvolume[:,w,d] = np.square(left_image[:,w] - right_image[:,w+d])
            if w-d >=0:
                right_costvolume[:,w,d] = np.square(left_image[:,w-d] - right_image[:,w])

    return left_costvolume, right_costvolume

def disparity_map(left_costvolume, right_costvolume, min_depth, max_depth, disparity_print):
    
    left_disparity = np.argmin(left_costvolume, axis=2)
    right_disparity = np.argmin(right_costvolume, axis=2)

    if disparity_print:
        print_img = left_disparity.astype(np.uint8) * int(255 / max_depth)
        cv2.imshow('left_disparity_map',print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print_img = right_disparity.astype(np.uint8) * int(255 / max_depth)
        cv2.imshow('right_disparity_map',print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return left_disparity, right_disparity, left_costvolume


def SD_rgb(left_image, right_image, min_depth, max_depth, disparity_print):

    left_CV, right_CV =  cost_volume_rgb(left_image, right_image, min_depth, max_depth)
    return disparity_map_rgb(left_CV, right_CV, min_depth, max_depth, disparity_print)

def cost_volume_rgb(left_image, right_image, min_depth, max_depth):

    height, width, _ = left_image.shape
    left_costvolume = np.full((height, width, 3, max_depth), np.inf)
    right_costvolume = np.full((height, width, 3, max_depth) ,np.inf)

    for d in range(min_depth, max_depth):
        for w in range(width):
            if w+d < width:
                left_costvolume[:,w,:,d] = np.square(left_image[:,w,:] - right_image[:,w+d,:])
            if w-d >=0:
                right_costvolume[:,w,:,d] = np.square(left_image[:,w-d,:] - right_image[:,w,:])

    return left_costvolume, right_costvolume

def weighted_average(list, threshold=2):
    height, width = list[1].shape
    result_costvolume = np.full((height, width), np.inf)
    for h in range(height):
        for w in range(width):
            a = abs(list[1][h,w]-list[2][h,w])
            b = abs(list[2][h,w]-list[0][h,w])
            c = abs(list[0][h,w]-list[1][h,w])
            if b/a > threshold and c/a > threshold:
                result_costvolume[h,w] = int((list[1][h,w]+list[2][h,w])/2)
            elif a/b > threshold and c/b > threshold:
                result_costvolume[h,w] = int((list[2][h,w]+list[0][h,w])/2)
            elif a/c > threshold and b/c > threshold:
                result_costvolume[h,w] = int((list[0][h,w]+list[1][h,w])/2)
            else:
                result_costvolume[h,w] = int((list[0][h,w]+list[1][h,w]+list[2][h,w])/3)
    return result_costvolume

def disparity_map_rgb(left_costvolume, right_costvolume, min_depth, max_depth, disparity_print):

    left_disparity_list = []
    right_disparity_list = []
    
    for RGB in range(3):
        left_disparity_list.append(np.argmin(left_costvolume[:,:,RGB,:], axis=2))
        right_disparity_list.append(np.argmin(right_costvolume[:,:,RGB,:], axis=2))

    left_disparity = weighted_average(left_disparity_list)
    right_disparity = weighted_average(right_disparity_list)

    if disparity_print:
        print_img = left_disparity.astype(np.uint8) * int(255 / max_depth)
        cv2.imshow('left_disparity_map',print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print_img = right_disparity.astype(np.uint8) * int(255 / max_depth)
        cv2.imshow('right_disparity_map',print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return left_disparity,  right_disparity, left_costvolume