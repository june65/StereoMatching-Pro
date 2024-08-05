import numpy as np
import cv2
from tqdm import tqdm
def LR_check(left_disparity, right_disparity, depth):
    height, width = left_disparity.shape
    aggregated_disparity = np.full((height, width), np.inf)
    for w in range(width-depth):
        for h in range(height):
            if left_disparity[h,w] == right_disparity[h,w+depth]:
                aggregated_disparity[h,w] = left_disparity[h,w]

    print_img = aggregated_disparity.astype(np.uint8) * int(255 / depth)
    cv2.imshow('aggregated_disparity',print_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    return aggregated_disparity