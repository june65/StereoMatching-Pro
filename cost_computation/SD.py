import numpy as np
import cv2

def SD(left_image, right_image, depth):

    left_CV, right_CV =  cost_volume(left_image, right_image, depth)
    return disparity_map(left_CV, right_CV, depth)


def cost_volume(left_image, right_image, depth):

    height, width, _ = left_image.shape
    left_costvolume = np.full((height, width, 3 ,depth), np.inf)
    right_costvolume = np.full((height, width, 3 ,depth) ,np.inf)

    for d in range(depth):
        for w in range(width):
            if w+d < width:
                left_costvolume[:,w,:,d] = np.sqrt(left_image[:,w,:] - right_image[:,w+d,:])
                right_costvolume[:,w,:,d] = np.sqrt(left_image[:,w-d,:] - right_image[:,w,:])

    left_costvolume = left_costvolume.mean(axis=2)
    right_costvolume = right_costvolume.mean(axis=2)

    return left_costvolume, right_costvolume


def disparity_map(left_costvolume, right_costvolume, depth):
    
    left_disparity = np.argmin(left_costvolume, axis=2)
    right_disparity = np.argmin(right_costvolume, axis=2)

    return left_disparity, right_disparity