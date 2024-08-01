import numpy as np
import cv2

def AD(left_image, right_image, depth):

    left_CV, right_CV =  cost_volume(left_image, right_image, depth)
    return disparity_map(left_CV, right_CV, depth)


def cost_volume(left_image, right_image, depth):

    height, width, _ = left_image.shape
    left_costvolume = np.full((height, width, 3 ,depth), np.inf)
    right_costvolume = np.full((height, width, 3 ,depth) ,np.inf)

    for d in range(depth):
        for w in range(width):
            if w+d < width:
                left_costvolume[:,w,:,d] = abs(left_image[:,w,:] - right_image[:,w+d,:])
            if w-d >=0:
                right_costvolume[:,w,:,d] = abs(left_image[:,w-d,:] - right_image[:,w,:])
        '''
        print_img = right_costvolume[:, :, :,d].astype(np.uint8)
        cv2.imshow('{right_costvolume depth}:'+ str(d), print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

    left_costvolume = left_costvolume.mean(axis=2)
    right_costvolume = right_costvolume.mean(axis=2)

    return left_costvolume, right_costvolume


def disparity_map(left_costvolume, right_costvolume, depth):
    
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