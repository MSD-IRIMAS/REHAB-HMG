import numpy as np
import sys
sys.path.append('../')
from utils.joints import joint_pairs

class Graph:
    """ The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
      
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """
    def __init__(self, num_nodes=18, strategy='uniform'):
        
        self.num_nodes = num_nodes
        self.joint_pairs = joint_pairs
        self.max_hop = 1  
        self.dilation = 1  
        self.get_edge()
        self.hop_dis = self.get_hop_distance(self.num_nodes, self.edge, max_hop=self.max_hop)
        self.get_adjacency(strategy)
    
    def __str__(self):
        return str(self.A)
    
    def get_edge(self):
        self.num_node = 18
        self_link = [(i, i) for i in range(self.num_node)]
        # neighbor_link = self.joint_pairs
        neighbor_link = [(i - 1, j - 1) for (i, j) in joint_pairs]
        self.edge = self_link + neighbor_link
        self.center = 0

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_nodes, self.num_nodes))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_nodes, self.num_nodes))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_nodes, self.num_nodes))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_nodes, self.num_nodes))
                a_close = np.zeros((self.num_nodes, self.num_nodes))
                a_further = np.zeros((self.num_nodes, self.num_nodes))
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise NotImplementedError("This Strategy is not supported")

    def get_hop_distance(self, num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD

# if __name__ == "__main__":
#     graph = Graph(num_nodes=18)
#     print("Adjacency Matrix:")
#     print(graph)



    # def normalize_graph(self, A):
    #     Dl = np.sum(A, axis=0)
    #     Dn = np.zeros_like(A, dtype=float)
    #     np.fill_diagonal(Dn, 1 / np.maximum(Dl, 1e-12))  # Avoid division by zero and ensure float division
    #     AD = np.dot(A, Dn)
    #     return AD

    # def get_adjacency_matrix(self):
    #     A = np.zeros((self.num_nodes, self.num_nodes))
    #     for edge in self.joint_pairs:
    #         start_vertex, end_vertex = edge
    #         A[start_vertex, end_vertex] = 1
    #     return A

    # def get_graph(self, normalize=True):
    #     A = self.get_adjacency_matrix()
    #     if normalize:
    #         A = self.normalize_graph(A)
    #     return A