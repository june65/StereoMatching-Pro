import argparse
from utils import ImageLoader
from cost_computation import SAD, SSD, ASW
from post_processing import LR_check, Mid_filter, Tree_filter
from accuracy_calculation import RMSE, Bad_ratio
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

#Data parameters
parser.add_argument('--dataset', default='tsukuba', help='data name')
#Cost computation parameters
parser.add_argument('--costmethod', default='AD', help='cost computation method')
parser.add_argument('--costwindow', default=33, help='kernel size')
parser.add_argument('--rgbexpand', default=False, help='rgb expansion')
#Post processing parameters
parser.add_argument('--lrcheck', default=False, help='left right consistency check')
parser.add_argument('--treefilter', default=False, help='tree filter')
parser.add_argument('--midfilter', default=0, help='weighted median filter')
#Accuracy calculation parameters
parser.add_argument('--rmse', default=True, help='rmse accuracy')
parser.add_argument('--badratio', default=True, help='bad pixel ratio')
parser.add_argument('--print', default=False, help='print disparity result')

args = parser.parse_args()

datapath = "./data/" + args.dataset+ "/"

def main():
    #Data loading
    imageset = ImageLoader(datapath)
    min_depth = 5
    max_depth = 15
    crop_depth = 17
    window_size = int(args.costwindow)
    mid_window_size = int(args.midfilter)
    right_image, GT_image = imageset[0]
    left_image, _ = imageset[1]

    window_size_list = list(range(1, window_size+1, 2))
    RMSE_list = []
    Bad_ratio_list = []

    for window_size in window_size_list:
        #Cost computation
        if args.costmethod == 'SAD':    
            left_disparity, right_disparity, left_costvolume = SAD(left_image, right_image, min_depth, max_depth, window_size, False)
        if args.costmethod == 'SSD':    
            left_disparity, right_disparity, left_costvolume = SSD(left_image, right_image, min_depth, max_depth, window_size, False)
        if args.costmethod == 'ASW':    
            left_disparity, right_disparity, left_costvolume = ASW(left_image, right_image, min_depth, max_depth, window_size, False, False)
        #Post processing
        if args.lrcheck:
            left_disparity = LR_check(left_disparity, right_disparity, max_depth, False)
        if args.treefilter:
            left_disparity = Tree_filter(left_image, left_disparity, left_costvolume, min_depth, max_depth, window_size, True, args.lrcheck, False)
        if mid_window_size > 0:
            left_disparity = Mid_filter(left_disparity, left_image, min_depth, max_depth, mid_window_size, False)

        #Accuracy calculation
        if args.rmse:
            RMSE_list.append(RMSE(GT_image, left_disparity, crop_depth))
        if args.badratio:
            Bad_ratio_list.append(Bad_ratio(GT_image, left_disparity, crop_depth, 1))

    plt.subplot(1, 2, 1)
    plt.plot(window_size_list, RMSE_list, marker='o')
    plt.title('RMSE by Window Size')
    plt.xlabel('Window Size')
    plt.ylabel('RMSE')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(window_size_list, Bad_ratio_list, marker='o', color='r')
    plt.title('Bad Ratio by Window Size')
    plt.xlabel('Window Size')
    plt.ylabel('Bad Ratio')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()