import numpy as np
import cv2
from tqdm import tqdm
from utils import RGB_to_gray
import heapq

import sys
limit_number = 100000
sys.setrecursionlimit(limit_number)

def prim(n, edges):
    graph = [[] for _ in range(n)]
    for u, v, weight in edges:
        graph[u].append((weight, v))
        graph[v].append((weight, u))
    
    mst = []
    visited = [False] * n
    min_heap = [(0, 0)] 
    parent = [-1] * n 
    
    while min_heap:
        weight, u = heapq.heappop(min_heap)
        if visited[u]:
            continue
        visited[u] = True
        if parent[u] != -1:
            mst.append((parent[u], u, weight))
        
        for next_weight, v in graph[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (next_weight, v))
                parent[v] = u
    return mst

#####BASELINE#####

def create_graph(image, height, width, texture):
    edges = []
    flag = 0.00001
    for h in range(height):
        for w in range(width):
            if h < height - 1:
                if not texture:
                    weight = (image[h, w] - image[h+1, w])**2
                else:
                    weight = 1
                if weight == 0:
                    weight = flag
                edges.append((h * width + w, (h + 1) * width + w, weight))
            if w < width - 1:
                if not texture:
                    weight = (image[h, w] - image[h, w+1])**2
                else:
                    weight = 1
                if weight == 0:
                    weight = flag
                edges.append((h * width + w, h * width + (w + 1), weight))
    return edges, height * width

def construct_MST(image, height, width, texture):
    edges, num_pixels = create_graph(image, height, width, texture)
    MST = prim(num_pixels, edges)
    return MST

def relation_MST(MST, num_pixels):
    parent = -np.ones(num_pixels, dtype=int)
    children = [[] for _ in range(num_pixels)]
    for data in MST:
        parent[data[1]] = data[0]
        children[data[0]].append(data[1])
    return parent, children

def simularity(X1, Y1, X2, Y2):
    sigma = 0.1
    D = np.abs(X1 - X2) + np.abs(Y1 - Y2)
    return np.exp(-(D / sigma))

def child_cost_agg(MST, width, children, child, target):
    X1 = target // width
    Y1 = target % width
    if target in memo:
        pass
    else:
        if len(child) == 0:
            memo[target] = standard_costvolume[X1, Y1]
        else:
            total_cost = 0
            for i in range(len(child)):
                child_i = child[i]
                X2 = child_i // width
                Y2 = child_i % width
                total_cost += child_cost_agg(MST, width, children, children[child_i], child_i) * simularity(X1, Y1, X2, Y2)
            standard_costvolume[target // width, target % width] = total_cost
            memo[target] = standard_costvolume[target // width, target % width]

    return memo[target]

def result_cost_agg(MST, width, parent, children, target):
    target_p = parent[target]
    X1 = target // width
    Y1 = target % width
    X_p = target_p // width
    Y_p = target_p % width
    if target in memo_result:
        pass
    else:
        if target_p == -1:
            memo_result[target] = child_cost_agg(MST, width, children, children[target], target)
        else:
            cost_1 = simularity(X_p, Y_p, X1, Y1) * result_cost_agg(MST, width, parent, children, target_p)
            cost_2 = (1 - simularity(X1, Y1, X_p, Y_p)**2) * child_cost_agg(MST, width, children, children[target], target)
            memo_result[target] = cost_1 + cost_2

    return memo_result[target]

def mst_filter(image, height, width, parent, iter=1):
    num_nodes = height * width
    sigma = 0.1
    filtered_image = image
    for _ in range(iter):
        new_filtered_image = filtered_image
        for i in range(1, num_nodes):
            child_pixel = filtered_image[i // width, i % width]
            parent_pixel = filtered_image[parent[i] // width, parent[i] % width]
            weight = np.exp(-abs(child_pixel - parent_pixel) / sigma)
            new_filtered_image[i // width, i % width] = (1 - weight) * child_pixel + weight * parent_pixel
        filtered_image = new_filtered_image
    return filtered_image

def Tree_filter(image, disparity, costvolume, depth, kernel_size, texture, LR_refine):
    pad_size = kernel_size // 2
    image = RGB_to_gray(image)
    height, width = image.shape

    if texture:
        MST = construct_MST(image, height, width, True)
        parent, children = relation_MST(MST, height*width)
        image = mst_filter(image, height, width, parent, iter=1)
        '''
        print_img = image.astype(np.uint8)
        cv2.imshow('{mst_filter image}:', print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

    if LR_refine:
        for d in range(depth):
            for w in range(width-d-pad_size):
                for h in range(height):
                    if disparity[h,w] > 0:
                        costvolume[h,w,d] = np.abs(d - disparity[h,w])
                    else:
                     costvolume[h,w,d] = 0
            '''
            print_img = costvolume[:, :, d].astype(np.uint8)
            cv2.imshow('{right_costvolume depth}:'+ str(d), print_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''

    global standard_costvolume
    global memo
    global memo_result 
    result_disparity = np.full((height, width, depth), np.inf)

    for d in tqdm(range(depth)):
        MST = construct_MST(image, height, width-d-pad_size, False)
        parent, children = relation_MST(MST, height*(width-d-pad_size))
        standard_costvolume = costvolume[:, :, d]
        '''
        print_img = standard_costvolume.astype(np.uint8)
        cv2.imshow('{right_costvolume depth}:'+ str(d), print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        memo = {}
        memo_result = {}
        for w in range(width-d-pad_size):
            for h in range(height):
                target = w + h*(width-d-pad_size)
                result_disparity[h, w, d] = result_cost_agg(MST, (width-d-pad_size), parent, children, target)
        '''
        print_img = result_disparity[:, :, d].astype(np.uint8)
        cv2.imshow('{right_costvolume depth}:'+ str(d), print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
    
    result_disparity_argmin = np.argmin(result_disparity, axis=2)

    print_img = result_disparity_argmin.astype(np.uint8) * int(255 / depth)
    cv2.imshow('standard_costvolume',print_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    return result_disparity_argmin