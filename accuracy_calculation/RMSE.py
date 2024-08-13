import numpy as np

def RMSE(GT_image, test_image, depth):
    if GT_image.shape != test_image.shape:
        raise ValueError("Different Size")
    
    height, width = GT_image.shape
    crop_GT_image = GT_image[depth:height-depth,depth:width-depth] / float(16)
    crop_test_image = test_image[depth:height-depth,depth:width-depth] 

    mse = np.mean((crop_GT_image - crop_test_image) ** 2)
    rmse_value = np.sqrt(mse)

    print(f"RMSE: {rmse_value}")

    return rmse_value
