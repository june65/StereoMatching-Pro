import argparse
from utils import ImageLoader
from cost_computation import AD, SD, SAD, SSD, ASW, SGM

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='tsukuba', help='data name')
parser.add_argument('--costmethod', default='SGM', help='cost computation method')

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
    if args.costmethod == 'ASW':    
        ASW(left_image, right_image, depth, 33)
    if args.costmethod == 'SGM':    
        SGM(left_image, right_image, depth)

if __name__ == "__main__":
    main()