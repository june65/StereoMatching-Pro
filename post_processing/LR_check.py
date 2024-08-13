import numpy as np
import cv2
from tqdm import tqdm
def LR_check(left_disparity, right_disparity, max_depth):
    height, width = left_disparity.shape
    disparity_mask = np.zeros((height, width))
    for w in range(width):
        for h in range(height):
            left_dis = left_disparity[h,w]
            if w+left_dis < width:
                right_dis = right_disparity[h,w+left_dis]
                if right_dis == left_dis:
                    disparity_mask[h,w] = 1

    aggregated_disparity = left_disparity * disparity_mask

    print_img = aggregated_disparity.astype(np.uint8) * int(255 / max_depth)
    cv2.imshow('aggregated_disparity',print_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    return aggregated_disparity