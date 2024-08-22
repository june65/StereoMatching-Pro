import argparse
from utils import ImageLoader
from cost_computation import AD, SD, SAD, SSD, ASW, SGM, AD_rgb, SD_rgb
from post_processing import LR_check, Mid_filter, Tree_filter, Graph_cut
from accuracy_calculation import RMSE, Bad_ratio
parser = argparse.ArgumentParser()

#Data parameters
parser.add_argument('--dataset', default='tsukuba', help='data name')
#Cost computation parameters
parser.add_argument('--costmethod', default='ASW', help='cost computation method')
parser.add_argument('--costwindow', default=33, help='kernel size')
parser.add_argument('--rgbexpand', default=False, help='rgb expansion')
#Post processing parameters
parser.add_argument('--lrcheck', default=False, help='left right consistency check')
parser.add_argument('--treefilter', default=False, help='tree filter')
parser.add_argument('--midfilter', default=0, help='weighted median filter')
parser.add_argument('--graphcut', default=False, help='graph cut algorithm (0:no graph cut)(1:before cost aggregation)(2:after cost aggregation)')
#Accuracy calculation parameters
parser.add_argument('--rmse', default=True, help='rmse accuracy')
parser.add_argument('--badratio', default=True, help='bad pixel ratio')
parser.add_argument('--print', default=True, help='print disparity result')

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
    disparity_print = args.print

    #Cost computation
    if args.costmethod == 'AD':
        if not args.rgbexpand:
            left_disparity, right_disparity, left_costvolume = AD(left_image, right_image, min_depth, max_depth, disparity_print)
        else:
            left_disparity, right_disparity, left_costvolume = AD_rgb(left_image, right_image, min_depth, max_depth, disparity_print)
    if args.costmethod == 'SD':
        if not args.rgbexpand:
            left_disparity, right_disparity, left_costvolume = SD(left_image, right_image, min_depth, max_depth, disparity_print)
        else:
            left_disparity, right_disparity, left_costvolume = SD_rgb(left_image, right_image, min_depth, max_depth, disparity_print)
    if args.costmethod == 'SAD':    
        left_disparity, right_disparity, left_costvolume = SAD(left_image, right_image, min_depth, max_depth, window_size, disparity_print)
    if args.costmethod == 'SSD':
        left_disparity, right_disparity, left_costvolume = SSD(left_image, right_image, min_depth, max_depth, window_size, disparity_print)
    if args.costmethod == 'ASW':    
        left_disparity, right_disparity, left_costvolume = ASW(left_image, right_image, min_depth, max_depth, window_size, False, disparity_print)
    if args.costmethod == 'SGM':    
        left_disparity, right_disparity = SGM(left_image, right_image, max_depth, disparity_print)

    #Post processing
    if args.graphcut:
        left_disparity = Graph_cut(left_disparity ,left_costvolume, min_depth, max_depth, disparity_print)
    if args.lrcheck:
        left_disparity = LR_check(left_disparity, right_disparity, max_depth, disparity_print)
    if args.treefilter:
        left_disparity = Tree_filter(left_image, left_disparity, left_costvolume, min_depth, max_depth, window_size, True, args.lrcheck, disparity_print)
    if mid_window_size > 0:
        left_disparity = Mid_filter(left_disparity, left_image, min_depth, max_depth, mid_window_size, disparity_print)

    #Accuracy calculation
    if args.rmse:
        RMSE(GT_image, left_disparity, crop_depth)
    if args.badratio:
        Bad_ratio(GT_image, left_disparity, crop_depth, 1)    

if __name__ == "__main__":
    main()