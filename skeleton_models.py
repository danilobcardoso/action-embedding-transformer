import numpy as np


ntu_rgbd = {
    'num_nodes': 25,
    'center': 1,
    'links': [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
                (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)],
    'colors': ['#02FF00','#02FF00','#02FF00','#02FF00','#FFFF00','#FFFF00','#FFFF00','#FFFF00','#FF9802', '#FF9802', '#FF9802', '#FF9802', '#02FFFF',
            '#02FFFF','#02FFFF','#02FFFF','#FF00FF','#FF00FF','#FF00FF','#FF00FF', '#02FF00','#FFFF00','#FFFF00','#FF9802','#FF9802'],
    'node_group': [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,1,1,2,2],
    'ss_selection': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
}

ntu_ss_1 = {
    'num_nodes': 1,
    'center': 0,
    'links': [],
    'colors': ['#02FF00'],
    'node_group': [0],
    'ss_selection': [0]
}

ntu_ss_2 = {
    'num_nodes': 5,
    'center': 1,
    'links': [(0,4), (1,4), (2,4), (3,4)],
    'colors': ['#02FF00','#02FF00','#FFFF00','#FF9802','#02FF00'],
    'node_group': [0,0,1,2,0],
    'ss_selection': [0,3,4,8,20]
}

ntu_ss_3 = {
    'num_nodes': 9,
    'center': 1,
    'links': [(0,1), (0,7), (0,6), (1,8), (2,3), (2, 8), (4, 8), (5, 8)],
    'colors': ['#02FF00','#02FF00','#02FF00','#02FF00','#FFFF00','#FF9802','#02FFFF','#FF00FF', '#02FF00'],
    'node_group': [0, 0, 0, 0, 1, 2, 3, 4, 0],
    'ss_selection': [0,1,2,3,4,8,12,16,20]
}

ntu_ss_4 = {
    'num_nodes': 15,
    'center': 1,
    'links': [(0, 1), (0, 10), (0,12), (1, 14), (2, 3), (2, 14), (4, 14), (4, 5), (5, 6), (7, 14), (7, 8), (8, 9), (10, 11), (12, 13)],
    'colors': ['#02FF00','#02FF00','#02FF00','#02FF00','#FFFF00','#FFFF00','#FFFF00','#FF9802','#FF9802','#FF9802','#02FFFF','#02FFFF','#FF00FF','#FF00FF','#02FF00'],
    'node_group': [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 0],
    'ss_selection': [0,1,2,3,4,5,7,8,9,11,12,14,16,18,20]
}

def upsample_columns(model_from, model_to):
    assert model_from['num_nodes'] < model_to['num_nodes']
    return [ model_to['ss_selection'].index(e) for e in model_from['ss_selection']  ]


def partial(sequence, model):
    return sequence[..., model['ss_selection'], :]


def get_kernel_by_group(skeleton_model):
    num_node = skeleton_model['num_nodes']
    links = skeleton_model['links']
    kernel_size = 5
    adj_matrix = np.zeros((kernel_size, num_node, num_node))

    for link in links:
        node1 = link[0]
        node2 = link[1]
        kernel1 = skeleton_model['node_group'][node1]
        kernel2 = skeleton_model['node_group'][node2]
        adj_matrix[kernel1, node1, node2] = 1
        adj_matrix[kernel2, node2, node1] = 1

    for i in range(num_node):
        kernel = skeleton_model['node_group'][i]
        adj_matrix[kernel, i, i] = 1


    norm_coeficient = np.einsum('knm->km', adj_matrix)
    norm_coeficient = 1/norm_coeficient
    norm_coeficient[norm_coeficient == np.inf] = 0

    temp = np.einsum('knm->kmn', adj_matrix)
    for k in range(kernel_size):
        for n in range(num_node):
            temp[k, n, :] = temp[k, n, :] * norm_coeficient[k, n]
    adj_matrix = np.einsum('kmn->knm', temp)
    return adj_matrix