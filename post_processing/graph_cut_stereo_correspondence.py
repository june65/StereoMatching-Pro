import os
import random

import numpy as np
import cv2
import networkx as nx
from tqdm import tqdm

from mincut import MinCut

class GraphCutStereoCorrespondance(object):
    def __init__(self, img1, img2, ndisp, color=False, k_size=5, label_step=2):
        """
        :img1:   image np array or image path of left image
        :img2:   image np array or image path of left image
        :color:  boolean, true for three channel BGR images, false for grayscale
        :ndisp:  Conservative upper bound on pixel disparity amount. Can be
                 obtained from Middlebury's calibration files.
        """

        # Assert correct format
        for im in [img1, img2]:
            assert isinstance(im, np.ndarray) or isinstance(im, str)

        # Handle paths
        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        if isinstance(img1, str):
            img1 = cv2.imread(img1, flag)
        if isinstance(img2, str):
            img2 = cv2.imread(img2, flag)

        # Assert correct dimensions
        for im in [img1, img2]:
            if color:
                assert im.ndim==3
            else:
                assert im.ndim==2

        assert img1.shape==img2.shape

        # Create local variables
        self.color = color
        self.ndisp = ndisp
        self.img1 = img1
        self.img2 = img2
        self.k_size = k_size
        self.label_step = label_step
        self.labels = np.arange(0, ndisp, label_step, dtype=np.int16)
        self.shape = self.img1.shape[:2]
        self.predictions = np.random.choice(self.labels, size=self.shape).flatten()
        self.V_thresh = 2 #int(self.ndisp/8)
        self.D_thresh = 100
        self.img_len = self.shape[0]*self.shape[1]

        # Create bordered images for window comparisons
        b = int((self.k_size-1)/2) #buffer

        self.img1b = cv2.cv2.copyMakeBorder(self.img1, b, b, b, b, cv2.BORDER_REFLECT101)
        self.img2b = cv2.cv2.copyMakeBorder(self.img2, b, b, b, b, cv2.BORDER_REFLECT101)

        self.img1b = np.hstack((np.zeros((self.shape[0]+2*b,self.ndisp,3)), self.img1b))
        self.img2b = np.hstack((np.zeros((self.shape[0]+2*b,self.ndisp,3)), self.img2b))

    def V(self, a, b):
        '''potential function. Feel free to experiment'''
        return 2*int(min(9, (a-b)**2))

    def find_neighbors(self, a_idxs, b):
        m,n = self.shape
        neighbors = []
        for a_idx in a_idxs:
            if a_idx>0:
                if (self.predictions[a_idx-1] == b) and (a_idx%n != 0):
                    neighbors.append((a_idx, a_idx-1))
            if a_idx<self.img_len-1:
                if (self.predictions[a_idx+1] == b) and ((a_idx+1)%n != 0):
                    neighbors.append((a_idx, a_idx+1))
            if a_idx>=n:
                if self.predictions[a_idx-n] == b:
                    neighbors.append((a_idx, a_idx-n))
            if a_idx<self.img_len-n:
                if self.predictions[a_idx+n] == b:
                    neighbors.append((a_idx, a_idx+n))
        return neighbors

    def a_b_swap(self, a, b):
        m,n = self.shape

        # Find a,b label locations
        a_idxs = np.where(self.predictions==a)[0]
        b_idxs = np.where(self.predictions==b)[0]
        # Find neighbors
        n_idxs = self.find_neighbors(a_idxs, b)

        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(list(a_idxs)+list(b_idxs)+['a','b'])

        # t-links
        for idx in list(a_idxs)+list(b_idxs):
            r,c = np.unravel_index(idx, self.shape)
            c += self.ndisp

            # D term
            snip1 = self.img1b[r:r+self.k_size, c:c+self.k_size]
            snip2 = self.img2b[r:r+self.k_size, c-a:c+self.k_size-a]
            Da = min(((snip2-snip1)**2).mean(), self.D_thresh)

            snip2 = self.img2b[r:r+self.k_size, c-b:c+self.k_size-b]
            Db = min(((snip2-snip1)**2).mean(), self.D_thresh)

            # Potential term
            Va = 0
            Vb = 0
            c -= self.ndisp
            if idx>0:
                if (self.predictions[idx-1] not in (a,b)) and (idx%n != 0):
                    Va += self.V(a, self.predictions[idx-1])
                    Vb += self.V(b, self.predictions[idx-1])
            if idx<self.img_len-1:
                if (self.predictions[idx+1] not in (a,b)) and ((idx+1)%n != 0):
                    Va += self.V(a, self.predictions[idx+1])
                    Vb += self.V(b, self.predictions[idx+1])
            if idx>=n:
                if self.predictions[idx-n] not in (a,b):
                    Va += self.V(a, self.predictions[idx-n])
                    Vb += self.V(b, self.predictions[idx-n])
            if idx<self.img_len-n:
                if self.predictions[idx+n] not in (a,b):
                    Va += self.V(a, self.predictions[idx+n])
                    Vb += self.V(b, self.predictions[idx+n])

            G.add_edge('a', idx, capacity=int(Da+Va))
            G.add_edge(idx, 'b', capacity=int(Db+Vb))

            G.add_edge(idx, 'a', capacity=int(Da+Va))
            G.add_edge('b', idx, capacity=int(Db+Vb))

        # n-links
        n_cap = int(self.V(a,b))
        for n in n_idxs:
            p,q = n
            G.add_edge(p, q, capacity=n_cap)
            G.add_edge(q, p, capacity=n_cap)

        # Compute min-cut and update labels accordingly
        mincut = MinCut(G, 'a', 'b')
        partitions = mincut.compute()

        partitions[0].remove('a')
        partitions[1].remove('b')

        self.predictions[list(partitions[1])] = a
        self.predictions[list(partitions[0])] = b

        # Record success
        success = False if set(partitions[1]) == set(a_idxs) else True
        return success

    def a_expansion(self, a):
        m,n = self.shape

        a_idxs = np.where(self.predictions==a)[0]
        initial_a_size = len(a_idxs)

        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(list(range(m*n))+['a','a_bar'])

        for i, pred in enumerate(self.predictions):
            # t-links
            r,c = np.unravel_index(i, self.shape)
            c += self.ndisp

            # D term
            snip1 = self.img1b[r:r+self.k_size, c:c+self.k_size]
            snip2 = self.img2b[r:r+self.k_size, c-a:c+self.k_size-a]
            Da = min(((snip2-snip1)**2).mean(),self.D_thresh)

            G.add_edge('a', i, capacity=int(Da))
            G.add_edge(i, 'a', capacity=int(Da)) #NEW4/21
            if pred==a:
                G.add_edge(i, 'a_bar', capacity=10_000_000) #inf capacity
                G.add_edge('a_bar', i, capacity=10_000_000) #NEW4/21
            else:
                snip2 = self.img2b[r:r+self.k_size, c-pred:c+self.k_size-pred]
                Da_bar = min(((snip2-snip1)**2).mean(), self.D_thresh)
                G.add_edge(i, 'a_bar', capacity=int(Da_bar))
                G.add_edge('a_bar', i, capacity=int(Da_bar)) #NEW4/21

            c -= self.ndisp
            # n-links
            # To avoid double counting neighbors, look to the right and down only
            if ((i+1)%n != 0): #not right-most pixel
                pred_n = self.predictions[i+1]
                if pred==pred_n:
                    G.add_edge(i, i+1, capacity=self.V(pred,a))
                    G.add_edge(i+1, i, capacity=self.V(pred,a))
                else:
                    aux = 'n_'+str(i)+'_'+str(i+1)
                    G.add_node(aux)
                    G.add_edge(i, aux, capacity=self.V(pred,a))
                    G.add_edge(aux, i, capacity=self.V(pred,a))
                    G.add_edge(i+1, aux, capacity=self.V(pred_n,a))
                    G.add_edge(aux, i+1, capacity=self.V(pred_n,a))
                    G.add_edge(aux, 'a_bar', capacity=self.V(pred,pred_n))
                    G.add_edge('a_bar', aux, capacity=self.V(pred,pred_n)) #NEW4/21
            if i//n<m-1: #not bottom-most pixel
                pred_n = self.predictions[i+n]
                if pred==pred_n:
                    G.add_edge(i, i+n, capacity=self.V(pred,a))
                    G.add_edge(i+n, i, capacity=self.V(pred,a))
                else:
                    aux = 'aux_'+str(i)+'_'+str(i+n)
                    G.add_node(aux)
                    G.add_edge(i, aux, capacity=self.V(pred,a))
                    G.add_edge(aux, i, capacity=self.V(pred,a))
                    G.add_edge(i+n, aux, capacity=self.V(pred_n,a))
                    G.add_edge(aux, i+n, capacity=self.V(pred_n,a))
                    G.add_edge(aux, 'a_bar', capacity=self.V(pred,pred_n))
                    G.add_edge('a_bar', aux, capacity=self.V(pred,pred_n)) #NEW4/21

        # Compute min-cut and update labels accordingly
        mincut = MinCut(G, 'a', 'a_bar')
        partitions = mincut.compute()


        # Ignore auxillary neighnor nodes and 'a_bar' node, which are strings
        partition = [x for x in partitions[1] if type(x) is int]

        self.predictions[partition] = a

        # Record success
        success = True if initial_a_size < len(partition) else False
        return success

    def calculate_a_expansion(self):
        rand_labels = self.labels.copy()
        random.shuffle(rand_labels)
        for l in tqdm(rand_labels):
            self.a_expansion(l)
        return np.reshape(self.predictions, self.shape)
