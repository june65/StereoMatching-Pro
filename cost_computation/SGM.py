import numpy as np
import cv2
from utils import RGB_to_gray, RGB_to_CIELab
from tqdm import tqdm
    
def SGM(left_image, right_image, depth, disparity_print):

    left_image = RGB_to_gray(left_image) 
    right_image = RGB_to_gray(right_image)
    
    left_CV, right_CV =  cost_volume(left_image, right_image, depth)
    aggregate_cost = aggregate_cost_volume(left_CV, right_CV)
    return disparity_map(left_CV, right_CV, aggregate_cost, depth, disparity_print)

def cost_volume(left_image, right_image, depth):

    height, width = left_image.shape
    left_costvolume = np.full((height, width, depth), np.inf)
    right_costvolume = np.full((height, width, depth) ,np.inf)

    for d in range(depth):
        for w in range(width):
            if w+d < width:
                left_costvolume[:,w,d] = np.abs(left_image[:,w] - right_image[:,w+d])
            if w-d >=0:
                right_costvolume[:,w,d] = np.abs(left_image[:,w-d] - right_image[:,w])

    return left_costvolume, right_costvolume

def aggregate_cost_volume(left_costvolume, right_costvolume):

    forward_pass = [(1, 0), (0, 1), (1, 1), (1, -1), (0, 2), (0, 3), (1, 2), (1, -2)]
    backward_pass = [(-1, 0), (0, -1), (-1, -1), (-1, 1), (0, -2), (0, -3), (-1, 2), (-1, -2)]
    p1 = 5
    p2 = 150
    height, width, depth = left_costvolume.shape 
    aggregated_costs =np.zeros((height, width, depth, 16))

    for index, (i, j) in tqdm(enumerate(forward_pass)):
        for h in range(height):
            for w in range(width):
                for d in range(depth):
                    if (h-i>=0 and h-i<height and w-j>=0 and w-j<width):
                        disparity_value =  left_costvolume[h-i, w-j, d]
                        L_value_1 = aggregated_costs[h-i, w-j,d-1,index] if d-1 >= 0 else np.inf
                        L_value_2 = aggregated_costs[h-i, w-j,d+1,index] if d+1 < depth else np.inf 
                        min_value1 = np.min(aggregated_costs[h-i, w-j,:])
                        aggregated_costs[h,w,d,index] += disparity_value + min(aggregated_costs[h-i, w-j,d,index] , L_value_1 + p1 , L_value_2+ p1 , min_value1 + p2) - min_value1

    for index, (i, j) in tqdm(enumerate(backward_pass)):
        for h in  range(height-1,-1,-1):
            for w in range(width-1,-1,-1):
                for d in range(depth):
                    if (h-i>=0 and h-i<height and w-j>=0 and w-j<width):
                        disparity_value =  left_costvolume[h-i, w-j, d]
                        L_value_1 = aggregated_costs[h-i, w-j,d-1,index+8] if d-1 >= 0 else np.inf
                        L_value_2 = aggregated_costs[h-i, w-j,d+1,index+8] if d+1 < depth else np.inf    
                        min_value1 = np.min(aggregated_costs[h-i, w-j,:,index+8])
                        aggregated_costs[h,w,d,index+8] += disparity_value + min(aggregated_costs[h-i, w-j,d,index+4] , L_value_1 + p1 , L_value_2+ p1 , min_value1 + p2) - min_value1

    aggregated_volume = aggregated_costs.mean(axis=3)
        
    return aggregated_volume


def disparity_map(left_costvolume, right_costvolume, aggregated_volume, depth, disparity_print):

    aggregated_volume = np.argmin(aggregated_volume, axis=2)
    left_disparity = np.argmin(left_costvolume, axis=2)
    right_disparity = np.argmin(right_costvolume, axis=2)

    if disparity_print:
        print_img = aggregated_volume.astype(np.uint8) * int(255 / depth)
        cv2.imshow('aggregated_volume',print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return left_disparity, right_disparity