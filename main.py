import argparse
from utils import *
from cost_computation import AD, SD, SAD

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='tsukuba', help='data name')
parser.add_argument('--costmethod', default='AD', help='cost computation method')

args = parser.parse_args()

datapath = "./data/" + args.dataset+ "/"

def main():
    imageset = ImageLoader(datapath)
    depth = 16
    right_image = imageset[1]
    left_image = imageset[2]
    
    if args.costmethod == 'AD':
        AD(left_image, right_image, depth)
    if args.costmethod == 'SD':
        SD(left_image, right_image, depth)
    

if __name__ == "__main__":
    main()