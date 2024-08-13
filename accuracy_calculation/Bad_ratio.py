import numpy as np

def Bad_ratio(GT_image, test_image, depth, threshold):
    if GT_image.shape != test_image.shape:
        raise ValueError("Different Size")
    
    height, width = GT_image.shape
    crop_GT_image = GT_image[depth:height-depth,depth:width-depth]
    crop_test_image = test_image[depth:height-depth,depth:width-depth]* int(255 / depth)
    
    difference = np.abs(crop_GT_image - crop_test_image)
    bad_pixels = difference > threshold
    bad_pixel_ratio = np.sum(bad_pixels) / GT_image.size

    print(f"Bad_ratio: {bad_pixel_ratio}")

    return bad_pixel_ratio