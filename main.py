import argparse
from utils import ImageLoader
from cost_computation import AD, SD, SAD, SSD, ASW, SGM
from post_processing import LR_check, Mid_filter, Tree_filter

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='tsukuba', help='data name')
parser.add_argument('--costmethod', default='ASW', help='cost computation method')
parser.add_argument('--lrcheck', default=False, help='left right consistency check')
parser.add_argument('--treefilter', default=True, help='tree filter')
parser.add_argument('--midfilter', default=False, help='weighted median filter')

args = parser.parse_args()

datapath = "./data/" + args.dataset+ "/"

def main():
    imageset = ImageLoader(datapath)
    depth = 16
    right_image = imageset[0]
    left_image = imageset[1]
    
    if args.costmethod == 'AD':
        left_disparity, right_disparity, left_costvolume = AD(left_image, right_image, depth)
    if args.costmethod == 'SD':
        left_disparity, right_disparity = SD(left_image, right_image, depth)
    if args.costmethod == 'SAD':    
        left_disparity, right_disparity, left_costvolume = SAD(left_image, right_image, depth, 3)
    if args.costmethod == 'SSD':    
        left_disparity, right_disparity = SSD(left_image, right_image, depth, 3)
    if args.costmethod == 'ASW':    
        left_disparity, right_disparity, left_costvolume = ASW(left_image, right_image, depth, 33, specular=False)
    if args.costmethod == 'SGM':    
        left_disparity, right_disparity = SGM(left_image, right_image, depth)

    #Post Processing
    if args.lrcheck:
        LR_check(left_disparity, right_disparity, depth)
    if args.treefilter:
        Tree_filter(left_image, left_disparity, left_costvolume, depth, 3)
    if args.midfilter:
        Mid_filter(left_disparity, left_image, depth, 3)

if __name__ == "__main__":
    main()