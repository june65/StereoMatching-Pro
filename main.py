import argparse
from utils import *
from cost_computation import AD, SD, SAD, SSD

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='tsukuba', help='data name')
parser.add_argument('--costmethod', default='SAD', help='cost computation method')

args = parser.parse_args()

datapath = "./data/" + args.dataset+ "/"

def main():
    imageset = ImageLoader(datapath)
    depth = 16
    right_image = imageset[0]
    left_image = imageset[1]
    
    if args.costmethod == 'AD':
        AD(left_image, right_image, depth)
    if args.costmethod == 'SD':
        SD(left_image, right_image, depth)
    if args.costmethod == 'SAD':    
        SAD(left_image, right_image, depth, 3)
    if args.costmethod == 'SSD':    
        SSD(left_image, right_image, depth, 3)

if __name__ == "__main__":
    main()