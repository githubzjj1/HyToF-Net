import sys

sys.path.extend(['../'])
from graph import tools


num_node = 44
self_link = [(i, i) for i in range(num_node)]
outward_ori_index = [
            
            (7, 9), (9, 32), (32, 33), (32, 34), (32, 35), (32, 36), (32, 37),
            
            (8, 10), (10, 38), (38, 39), (38, 40), (38, 41), (38, 42), (38, 43),
            
            (13, 15), (15, 24), (24, 20), (24, 22),
            
            (14, 16), (16, 25), (25, 21), (25, 23),
            
            (31, 30), (30, 27), (27, 0), (0, 26), (26, 1), (1, 29), (29, 3), (26, 2), (2, 28), (28, 4),
            
            (18, 17), (18, 5), (18, 6), (18, 19), (19, 11), (19, 12),
            
            (5, 7), (6, 8), (11, 13), (12, 14), (18, 0), (0, 17)
]
outward = [(i, j) for (i, j) in outward_ori_index]
inward = [(j, i) for (i, j) in outward]
neighbor = inward + outward


body_part = [
    
    [5, 6, 17, 18, 19, 11, 12],                 
    
    [7, 9, 32, 33, 34, 35, 36, 37],              
    
    [8, 10, 38, 39, 40, 41, 42, 43],             
    
    [13, 15, 24, 20, 22],                        
    
    [14, 16, 25, 21, 23],                        
    
    [31, 30, 27, 0, 26, 1, 29, 3, 2, 28, 4],     
]

num_part = len(body_part)  


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.part_mask = self.build_part_mask(body_part, num_node)
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
    body_mask = Graph('spatial').build_part_mask(body_part, num_node)
    plt.imshow(A, cmap='gray')
    plt.show()
    
    plt.imshow(body_mask)
    plt.savefig('part_body.png')
    plt.show()

    print(A)