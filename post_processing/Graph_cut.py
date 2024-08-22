import numpy as np
import cv2
import maxflow
from tqdm import tqdm

def weight_function(cost1):
    return cost1**2

def penalty_function(cost1, cost2, penalty_C):
    return (np.abs(cost1-cost2)**2) * penalty_C

def Graph_cut(disparity, cost_volume, min_depth, max_depth, disparity_print):
    height, width, _ = cost_volume.shape
    penalty_C = 50
    cost_volume = cost_volume.astype(np.uint8)
    depth = max_depth - min_depth + 1
    g = maxflow.Graph[float](height * width * depth, (height * width * depth) * 6)
    nodes = g.add_nodes(height * width * depth)

    def graph_weight(idx, h1, w1, d1, h2, w2, d2, flag):
        
        if d1 != d2:
            weight = weight_function(cost_volume[h1, w1, d1])
            g.add_edge(nodes[idx], nodes[flag], weight, weight)
        else:
            weight = penalty_function(disparity[h1, w1], disparity[h2, w2], penalty_C)
            g.add_edge(nodes[idx], nodes[flag], weight, weight)
        return 0
    
    for d in tqdm(range(min_depth, max_depth)):
        for h in range(height):
            for w in range(width):
                idx = (d-min_depth) * height * width + h * width + w

                if w > 0: 
                    graph_weight(idx, h, w, d, h, w-1, d, idx-1)
                if w < width - 1:
                    graph_weight(idx, h, w, d, h, w+1, d, idx+1)
                if h > 0:
                    graph_weight(idx, h, w, d, h-1, w, d, idx-width)
                if h < height - 1:
                    graph_weight(idx, h, w, d, h+1, w, d, idx+width)
                if d > min_depth:
                    graph_weight(idx, h, w, d, h, w, d-1, idx-(height*width))
                if d < max_depth - 1:
                    graph_weight(idx, h, w, d, h, w, d+1, idx+(height*width))

                if d == min_depth:
                    g.add_tedge(nodes[idx], penalty_C, np.inf)
                if d == max_depth-1:
                    g.add_tedge(nodes[idx], np.inf, penalty_C)

    g.maxflow()

    labels = np.zeros((height, width, max_depth), dtype=np.uint8)
    aggregated_disparity = np.zeros((height, width))
    for d in range(min_depth, max_depth):
        for h in range(height):
            for w in range(width):
                idx = (d-min_depth) * height * width + h * width + w
                labels[h, w, d] = g.get_segment(nodes[idx])
                if labels[h, w, d]:
                    aggregated_disparity[h, w] = d
        '''
        #Graph Node Value Print
        print_img = labels[:,:,d].astype(np.uint8) * int(255)
        cv2.imshow('labels',print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()            
        '''

    if disparity_print:
        print_img = aggregated_disparity.astype(np.uint8) * int(255 / max_depth)
        cv2.imshow('aggregated_disparity',print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return aggregated_disparity
