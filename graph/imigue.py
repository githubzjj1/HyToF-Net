import sys

sys.path.extend(['../'])
from graph import tools

num_node = 22
self_link = [(i, i) for i in range(num_node)]
outward_ori_index = [
            (1, 0), 
            (1, 2), (2, 3), (3, 4), 
            (1, 5), (5, 6), (6, 7), 
            (0, 8), (8, 10), (0, 9), (9, 11), 
            (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), 
            (4, 17), (4, 18), (4, 19), (4, 20), (4, 21) 
]
outward = [(i, j) for (i, j) in outward_ori_index]
inward = [(j, i) for (i, j) in outward]
neighbor = inward + outward

body_part = [
    
    [0, 1, 8, 9, 10, 11],
    
    [2, 3, 4],
    
    [5, 6, 7],
    
    [12, 13, 14, 15, 16],
    
    [17, 18, 19, 20, 21]
]

body_part_bone = [
    
    [0, 1, 8, 9, 10, 11],
    
    [2, 3, 4],
    
    [5, 6, 7],
    
    [12, 13, 14, 15, 16],
    
    [17, 18, 19, 20, 21]
]
num_part = len(body_part)


class Graph:
    def __init__(self, labeling_mode='spatial', body_part=body_part, body_part_bone=body_part_bone, num_node=num_node):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.part_mask = self.build_part_mask(body_part, num_node)
        self.part_mask_bone = self.build_part_mask(body_part_bone, num_node)
        self.num_part = num_part

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_sum_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
    
    def build_part_mask(self, body_part, num_node):

        mask = tools.build_part_mask(body_part, num_node)

        return mask



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)