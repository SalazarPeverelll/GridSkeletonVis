from abc import ABC, abstractmethod

import numpy as np
import os
import time
import math
import json

from umap.umap_ import fuzzy_simplicial_set, make_epochs_per_sample
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from singleVis.kcenter_greedy import kCenterGreedy
from singleVis.intrinsic_dim import IntrinsicDim
from singleVis.backend import get_graph_elements, get_attention
from singleVis.utils import find_neighbor_preserving_rate
from kmapper import KeplerMapper
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from scipy.sparse import csr_matrix
import networkx as nx
from itertools import combinations
import torch
from scipy.stats import entropy
from umap import UMAP
from scipy.special import softmax

class SpatialEdgeConstructorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, data_provider) -> None:
        pass

    @abstractmethod
    def construct(self, *args, **kwargs):
        # return head, tail, weight, feature_vectors
        pass

    @abstractmethod
    def record_time(self, save_dir, file_name, operation, t):
        pass

'''Base class for Spatial Edge Constructor'''
class SpatialEdgeConstructor(SpatialEdgeConstructorAbstractClass):
    '''Construct spatial complex
    '''
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors) -> None:
        """Init parameters for spatial edge constructor

        Parameters
        ----------
        data_provider : data.DataProvider
             data provider
        init_num : int
            init number to calculate c
        s_n_epochs : int
            the number of epochs to fit for one iteration(epoch)
            e.g. n_epochs=5 means each edge will be sampled 5*prob times in one training epoch
        b_n_epochs : int
            the number of epochs to fit boundary samples for one iteration (epoch)
        n_neighbors: int
            local connectivity
        """
        self.data_provider = data_provider
        self.init_num = init_num
        self.s_n_epochs = s_n_epochs
        self.b_n_epochs = b_n_epochs
        self.n_neighbors = n_neighbors


    ################################## mapper start ######################################################

    # def _construct_mapper_complex(self, train_data, filter_function,epoch, model):
    #     """
    #         construct a mapper complex using a filter function
    #         """
    #     # Apply filter function to the data
    #     print(f"Applying filter function: {filter_function.__name__}...")
    #     filter_values = filter_function(train_data, epoch, model)
    #     print(f"Filter function applied, got {len(filter_values)} filter values.")

    #     # Partition filter values into overlapping intervals
    #     print("Partitioning filter values into intervals...")
    #     intervals = self._partition_into_intervals(filter_values)
    #     print(f"Partitioned into {len(intervals)} intervals.")
    #     # print("intervals",intervals)

    #     # For each interval, select data points in that interval, cluster them,
    #     # and create a simplex for each cluster
       
    #     # Initialize an empty graph
    #     G = nx.Graph()
    #     print("Constructing simplices...")
    #     for interval in intervals:
    #         # interval_data = train_data[(filter_values >= interval[0]) & (filter_values < interval[1])]
    #         interval_data_indices = np.where((filter_values >= interval[0]) & (filter_values < interval[1]))[0]

    #         if len(interval_data_indices) > 0:
    #             # Use DBSCAN to cluster data in the current interval
    #             # Note: Depending on your data, you might want to use a different clustering algorithm
    #             interval_data = train_data[interval_data_indices]
    #             db = DBSCAN(eps=0.3, min_samples=2).fit(interval_data)
    #             cluster_labels = db.labels_

    #             # Create a simplex for each cluster
    #             for cluster_id in np.unique(cluster_labels):
    #                 if cluster_id != -1:  # Ignore noise points
    #                     cluster_indices = interval_data_indices[cluster_labels == cluster_id]
    #                     # Add edges to the graph for every pair of points in the cluster
    #                     G.add_edges_from(combinations(cluster_indices, 2))
    #     # Verify if the graph has nodes and edges
    #     if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
    #         raise ValueError("Graph has no nodes or edges.")
                        
    #     mapper_complex = nx.adjacency_matrix(G)
    #     print(f"Finished constructing simplices using {filter_function.__name__}.")

    #     return mapper_complex
    def _construct_mapper_complex(self, train_data, filter_functions, epoch, model):
        """
        construct a mapper complex using a list of filter functions
        """
        for filter_function in filter_functions:
            # Apply filter function to the data
            print(f"Applying filter function: {filter_function.__name__}...")
            filter_values = filter_function(train_data, epoch, model)
            print(f"Filter function applied, got {len(filter_values)} filter values.")

            # Partition filter values into overlapping intervals
            print("Partitioning filter values into intervals...")
            intervals = self._partition_into_intervals(filter_values)
            print(f"Partitioned into {len(intervals)} intervals.")

            # For each interval, select data points in that interval, cluster them,
            # and create a simplex for each cluster

            # Initialize an empty graph
            G = nx.Graph()
            print("Constructing simplices...")
            for interval in intervals:
                interval_data_indices = np.where((filter_values >= interval[0]) & (filter_values < interval[1]))[0]

                if len(interval_data_indices) > 0:
                    # Use DBSCAN to cluster data in the current interval
                    # interval_data = train_data[interval_data_indices]
                    # db = DBSCAN(eps=0.3, min_samples=2).fit(interval_data)
                    # cluster_labels = db.labels_
                    interval_data = np.column_stack([train_data[interval_data_indices], filter_values[interval_data_indices]])
                    db = DBSCAN(eps=0.3, min_samples=2).fit(interval_data)
                    cluster_labels = db.labels_


                    # Create a simplex for each cluster
                    for cluster_id in np.unique(cluster_labels):
                        if cluster_id != -1:  # Ignore noise points
                            cluster_indices = interval_data_indices[cluster_labels == cluster_id]
                            G.add_edges_from(combinations(cluster_indices, 2))

            # Verify if the graph has nodes and edges
            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                raise ValueError("Graph has no nodes or edges.")

            mapper_complex = nx.adjacency_matrix(G)
            print(f"Finished constructing simplices using {filter_function.__name__}.")

        return mapper_complex
    
    def _construct_boundary_wise_complex_mapper(self, train_data, border_centers, filter_function,epoch, model):
        """
        Construct a boundary-wise mapper complex using a filter function.
        For each cluster of data points (derived from the filter function applied to data points in a particular interval),
        construct a vertex in the mapper graph. Connect vertices if their corresponding data clusters intersect.
        """
        # Combine train and border data
        # print(train_data.shape, border_centers.shape)
        fitting_data = np.concatenate((train_data, border_centers), axis=0)
        
        # Apply the filter function
        filter_values = filter_function(fitting_data, epoch, model)
        
        # Partition filter values into overlapping intervals
        print("Partitioning filter values into intervals...")
        intervals = self._partition_into_intervals(filter_values)
        print(f"Partitioned into {len(intervals)} intervals.")

        # For each interval, select data points in that interval, cluster them,
        # and create a simplex for each cluster
       
        # Initialize an empty graph
        G = nx.Graph()
        print("Constructing simplices...")
        for interval in intervals:
            # interval_data = train_data[(filter_values >= interval[0]) & (filter_values < interval[1])]
            interval_data_indices = np.where((filter_values >= interval[0]) & (filter_values < interval[1]))[0]

            if len(interval_data_indices) > 0:
                # Use DBSCAN to cluster data in the current interval
                # Note: Depending on your data, you might want to use a different clustering algorithm
                interval_data = fitting_data[interval_data_indices]
                db = DBSCAN(eps=0.3, min_samples=2).fit(interval_data)
                cluster_labels = db.labels_

                # Create a simplex for each cluster
                for cluster_id in np.unique(cluster_labels):
                    if cluster_id != -1:  # Ignore noise points
                        cluster_indices = interval_data_indices[cluster_labels == cluster_id]
                        # Add edges to the graph for every pair of points in the cluster
                        G.add_edges_from(combinations(cluster_indices, 2))
        # Verify if the graph has nodes and edges
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            raise ValueError("Graph has no nodes or edges.")
                        
        mapper_complex = nx.adjacency_matrix(G)
        print(f"Finished constructing simplices using {filter_function.__name__}.")

        return mapper_complex

    # def _clusters_intersect(self, cluster1, cluster2):
    #     """
    #     Check if two data clusters intersect.
    #     Note: Here we assume that clusters are represented as sets of data points.
    #     Depending on your actual implementation, you might need to adjust this.
    #     """
    #     return not set(cluster1).isdisjoint(cluster2)
    
    def _clusters_intersect(self, cluster1, cluster2):
        """
        Check if two clusters intersect, i.e., have at least one point in common.
        """
        cluster1 = map(tuple, cluster1)
        cluster2 = map(tuple, cluster2)

        return not set(cluster1).isdisjoint(set(cluster2))



    def _partition_into_intervals(self, filter_values, n_intervals=10, overlap=0.1):
        """
        Partition the range of filter_values into overlapping intervals
        """
        filter_min, filter_max = np.min(filter_values), np.max(filter_values)
        interval_size = (filter_max - filter_min) / n_intervals
        overlap_size = interval_size * overlap
    
        intervals = []
        for i in range(n_intervals):
            interval_start = filter_min + i * interval_size
            interval_end = interval_start + interval_size + overlap_size
            intervals.append((interval_start, interval_end))
    
        return intervals
    
    # def density_filter_function(self, data, epsilon=0.5):
    #     """
    #     The function calculates the density of each data point based on a Gaussian kernel
    #     """
    #     densities = np.zeros(data.shape[0])
    
    #     for i, x in enumerate(data):
    #         distances = distance.cdist([x], data, 'euclidean').squeeze()
    #         densities[i] = np.sum(np.exp(-(distances ** 2) / epsilon))
    
    #     # Normalize the densities so that they sum up to 1
    #     densities /= np.sum(densities)

    #     return densities
    #### TODO density_filter_function
    def density_filter_function(self, data, epoch, model, epsilon=0.5):
        """
        The function calculates the density of each data point based on a Gaussian kernel
        """
        # distances = distance.cdist(data, data, 'euclidean')
        # densities = np.sum(np.exp(-(distances ** 2) / epsilon), axis=1)

        # # Normalize the densities so that they sum up to 1
        # densities /= np.sum(densities)
        densities = np.random.rand(data.shape[0])
    
        # Normalize the densities so that they sum up to 1
        densities /= np.sum(densities)

        return densities
    
    def hook(self, activations, module, input, output):
        activations.append(output)

    def activation_filter(self, data, epoch, model):
        activations = []  # Define activations here as local variable
        model_location = os.path.join(self.data_provider.content_path, "Model", "Epoch_{}".format(epoch), "subject_model.pth")
        model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        model.to(self.data_provider.DEVICE)
        model.eval()

        # Define a hook to capture the activations
        def hook(module, input, output):
            activations.append(output.detach())

        # Register the hook to the desired layer of the model
       # Find the last layer of the model dynamically
        target_layer = model.prediction

        if target_layer is not None:
            target_layer.register_forward_hook(hook)
            with torch.no_grad():
                # Convert the numpy.ndarray to a torch.Tensor
                input_tensor = torch.from_numpy(data)
                model(input_tensor)
        else:
            raise ValueError("Unable to find the 'prediction' layer in the model.")

        # Return the collected activations as a high-dimensional representation
        high_dimensional_representation = torch.cat(activations, dim=0)
        return high_dimensional_representation
    
    def decison_boundary_distance_filter(self,data, epoch, model):
        preds = self.data_provider.get_pred(epoch, data)
        preds = preds + 1e-8

        sort_preds = np.sort(preds, axis=1)
        # diff = (sort_preds[:, -1] - sort_preds[:, -2]) / (sort_preds[:, -1] - sort_preds[:, 0])

        # Confidence is the maximum predicted probability
        confidence = np.max(preds, axis=1)

        # Predicted label is the index of the maximum probability
        predicted_label = np.argmax(preds, axis=1)

        # Combine the predicted label and the confidence into a score
        score = predicted_label + (1 - confidence)

        return score
    
    def umap_filter(self, data,epoch, model, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean'):
        umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                      min_dist=min_dist, metric=metric)
        transformed_data = umap_model.fit_transform(data)
        return transformed_data

    ################################## mapper end ######################################################
    
    
    def _construct_fuzzy_complex(self, train_data):
        """
        construct a vietoris-rips complex
        """
        # number of trees in random projection forest
        n_trees = min(64, 5 + int(round((train_data.shape[0]) ** 0.5 / 20.0)))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
        # distance metric
        metric = "euclidean"
        # get nearest neighbors
        
        nnd = NNDescent(
            train_data,
            n_neighbors=self.n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        knn_indices, knn_dists = nnd.neighbor_graph
        random_state = check_random_state(None)
        complex, sigmas, rhos = fuzzy_simplicial_set(
            X=train_data,
            n_neighbors=self.n_neighbors,
            metric=metric,
            random_state=random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )
        return complex, sigmas, rhos, knn_indices
    
    # def _construct_boundary_wise_complex(self, train_data, border_centers):
    #     """compute the boundary wise complex
    #         for each border point, we calculate its k nearest train points
    #         for each train data, we calculate its k nearest border points
    #     """
    #     high_neigh = NearestNeighbors(n_neighbors=self.n_neighbors, radius=0.4)
    #     high_neigh.fit(border_centers)
    #     fitting_data = np.concatenate((train_data, border_centers), axis=0)
    #     knn_dists, knn_indices = high_neigh.kneighbors(fitting_data, n_neighbors=self.n_neighbors, return_distance=True)
    #     knn_indices = knn_indices + len(train_data)

    #     random_state = check_random_state(None)
    #     bw_complex, sigmas, rhos = fuzzy_simplicial_set(
    #         X=fitting_data,
    #         n_neighbors=self.n_neighbors,
    #         metric="euclidean",
    #         random_state=random_state,
    #         knn_indices=knn_indices,
    #         knn_dists=knn_dists,
    #     )
    #     return bw_complex, sigmas, rhos, knn_indices
    
    def _construct_boundary_wise_complex(self, train_data, border_centers):
        """compute the boundary wise complex
            for each border point, we calculate its k nearest train points
            for each train data, we calculate its k nearest border points
        """
        high_neigh = NearestNeighbors(n_neighbors=self.n_neighbors, radius=0.4)
        high_neigh.fit(border_centers)
        fitting_data = np.concatenate((train_data, border_centers), axis=0)
        knn_dists, knn_indices = high_neigh.kneighbors(train_data, n_neighbors=self.n_neighbors, return_distance=True)
        knn_indices = knn_indices + len(train_data)

        high_bound_neigh = NearestNeighbors(n_neighbors=self.n_neighbors, radius=0.4)
        high_bound_neigh.fit(train_data)
        bound_knn_dists, bound_knn_indices = high_bound_neigh.kneighbors(border_centers, n_neighbors=self.n_neighbors, return_distance=True)
        
        knn_dists = np.concatenate((knn_dists, bound_knn_dists), axis=0)
        knn_indices = np.concatenate((knn_indices, bound_knn_indices), axis=0)

        random_state = check_random_state(None)
        bw_complex, sigmas, rhos = fuzzy_simplicial_set(
            X=fitting_data,
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            random_state=random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )

        return bw_complex, sigmas, rhos, knn_indices
    
    def _construct_step_edge_dataset(self, vr_complex, bw_complex):
        """
        construct the mixed edge dataset for one time step
            connect border points and train data(both direction)
        :param vr_complex: Vietoris-Rips complex
        :param bw_complex: boundary-augmented complex
        :param n_epochs: the number of epoch that we iterate each round
        :return: edge dataset
        """
        # get data from graph
        _, vr_head, vr_tail, vr_weight, _ = get_graph_elements(vr_complex, self.s_n_epochs)
        
        # get data from graph
        if self.b_n_epochs == 0:
            return vr_head, vr_tail, vr_weight
        else:
            _, bw_head, bw_tail, bw_weight, _ = get_graph_elements(bw_complex, self.b_n_epochs)
            head = np.concatenate((vr_head, bw_head), axis=0)
            tail = np.concatenate((vr_tail, bw_tail), axis=0)
            weight = np.concatenate((vr_weight, bw_weight), axis=0)
        return head, tail, weight
    
    def construct_edge_dataset(self, mapper_complex, fuzzy_complex, border_complex=None):
        """
        construct the mixed edge dataset for one time step
        """
        # get data from mapper_complex graph
        _, mapper_head, mapper_tail, mapper_weight, _ = get_graph_elements(mapper_complex, self.s_n_epochs)

        # get data from fuzzy_complex graph
        _, fuzzy_head, fuzzy_tail, fuzzy_weight, _ = get_graph_elements(fuzzy_complex, self.s_n_epochs)
        if border_complex !=None:
            _, border_head, border_tail, border_weight, _ = get_graph_elements(border_complex, self.b_n_epochs)
            head = np.concatenate((mapper_head, fuzzy_head,border_head), axis=0)
            tail = np.concatenate((mapper_tail, fuzzy_tail,border_tail), axis=0)
            weight = np.concatenate((mapper_weight, fuzzy_weight,border_weight), axis=0)
        else:
            # concatenate the head, tail and weight from both graphs
            head = np.concatenate((mapper_head, fuzzy_head), axis=0)
            tail = np.concatenate((mapper_tail, fuzzy_tail), axis=0)
            weight = np.concatenate((mapper_weight, fuzzy_weight), axis=0)

        return head, tail, weight
    

    def construct(self):
        return NotImplemented
    
    def record_time(self, save_dir, file_name, operation, t):
        file_path = os.path.join(save_dir, file_name+".json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                ti = json.load(f)
        else:
            ti = dict()
        ti[operation] = t
        with open(file_path, "w") as f:
            json.dump(ti, f)
        
'''
Strategies:
    Random: random select samples
    KC: select coreset using k center greedy algorithm (recommend)
    KC Parallel: parallel selecting samples
    KC Hybrid: additional term for repley connecting epochs
'''

class RandomSpatialEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors) -> None:
        super().__init__(data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors)
    
    def construct(self):
        # dummy input
        edge_to = None
        edge_from = None
        sigmas = None
        rhos = None
        weight = None
        probs = None
        feature_vectors = None
        attention = None
        knn_indices = None
        time_step_nums = list()
        time_step_idxs_list = list()

        train_num = self.data_provider.train_num
        selected_idxs = np.random.choice(np.arange(train_num), size=self.init_num, replace=False)
        selected_idxs_t = np.array(range(len(selected_idxs)))

        # each time step
        for t in range(self.data_provider.s, self.data_provider.e+1, self.data_provider.p):
            # load train data and border centers
            train_data = self.data_provider.train_representation(t).squeeze()

            train_data = train_data[selected_idxs]
            time_step_idxs_list.append(selected_idxs_t.tolist())

            selected_idxs_t = np.random.choice(list(range(len(selected_idxs))), int(0.9*len(selected_idxs)), replace=False)
            selected_idxs = selected_idxs[selected_idxs_t]

            if self.b_n_epochs != 0:
                border_centers = self.data_provider.border_representation(t).squeeze()
                border_centers = border_centers
                complex, sigmas_t1, rhos_t1, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                bw_complex, sigmas_t2, rhos_t2, _ = self._construct_boundary_wise_complex(train_data, border_centers)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, bw_complex)
                sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
                rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
                fitting_data = np.concatenate((train_data, border_centers), axis=0)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
                t_num = len(train_data)
                b_num = len(border_centers)
            else:
                complex, sigmas_t, rhos_t, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, None, self.n_epochs)
                fitting_data = np.copy(train_data)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
                t_num = len(train_data)
                b_num = 0

            if edge_to is None:
                edge_to = edge_to_t
                edge_from = edge_from_t
                weight = weight_t
                probs = weight_t / weight_t.max()
                feature_vectors = fitting_data
                attention = attention_t
                sigmas = sigmas_t
                rhos = rhos_t
                knn_indices = knn_idxs_t
                time_step_nums.append((t_num, b_num))
            else:
                # every round, we need to add len(data) to edge_to(as well as edge_from) index
                increase_idx = len(feature_vectors)
                edge_to = np.concatenate((edge_to, edge_to_t + increase_idx), axis=0)
                edge_from = np.concatenate((edge_from, edge_from_t + increase_idx), axis=0)
                # normalize weight to be in range (0, 1)
                weight = np.concatenate((weight, weight_t), axis=0)
                probs_t = weight_t / weight_t.max()
                probs = np.concatenate((probs, probs_t), axis=0)
                sigmas = np.concatenate((sigmas, sigmas_t), axis=0)
                rhos = np.concatenate((rhos, rhos_t), axis=0)
                feature_vectors = np.concatenate((feature_vectors, fitting_data), axis=0)
                attention = np.concatenate((attention, attention_t), axis=0)
                knn_indices = np.concatenate((knn_indices, knn_idxs_t+increase_idx), axis=0)
                time_step_nums.append((t_num, b_num))

        return edge_to, edge_from, weight, feature_vectors, time_step_nums, time_step_idxs_list, knn_indices , sigmas, rhos, attention
    

class kcSpatialEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, MAX_HAUSDORFF, ALPHA, BETA, init_idxs=None, adding_num=100) -> None:
        super().__init__(data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors)
        self.MAX_HAUSDORFF = MAX_HAUSDORFF
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.init_idxs = init_idxs
        self.adding_num = adding_num
    
    def _get_unit(self, data, init_num, adding_num=100):
        # normalize
        t0 = time.time()
        l = len(data)
        idxs = np.random.choice(np.arange(l), size=init_num, replace=False)
        # _,_ = hausdorff_dist_cus(data, idxs)

        id = IntrinsicDim(data)
        d0 = id.twonn_dimension_fast()
        # d0 = twonn_dimension_fast(data)

        kc = kCenterGreedy(data)
        _ = kc.select_batch_with_budgets(idxs, adding_num)
        c0 = kc.hausdorff()
        t1 = time.time()
        return c0, d0, "{:.1f}".format(t1-t0)
    
    def construct(self):
        """construct spatio-temporal complex and get edges

        Returns
        -------
        _type_
            _description_
        """

        # dummy input
        edge_to = None
        edge_from = None
        sigmas = None
        rhos = None
        weight = None
        probs = None
        feature_vectors = None
        attention = None
        knn_indices = None
        time_step_nums = list()
        time_step_idxs_list = list()

        train_num = self.data_provider.train_num
        if self.init_idxs is None:
            selected_idxs = np.random.choice(np.arange(train_num), size=self.init_num, replace=False)
        else:
            selected_idxs = np.copy(self.init_idxs)

        baseline_data = self.data_provider.train_representation(self.data_provider.e)
        max_x = np.linalg.norm(baseline_data, axis=1).max()
        baseline_data = baseline_data/max_x
        
        c0,d0,_ = self._get_unit(baseline_data, self.init_num, self.adding_num)

        if self.MAX_HAUSDORFF is None:
            self.MAX_HAUSDORFF = c0-0.01

        # each time step
        for t in range(self.data_provider.e, self.data_provider.s - 1, -self.data_provider.p):
            print("=================+++={:d}=+++================".format(t))
            # load train data and border centers
            train_data = self.data_provider.train_representation(t)

            # normalize data by max ||x||_2
            max_x = np.linalg.norm(train_data, axis=1).max()
            train_data = train_data/max_x

            # get normalization parameters for different epochs
            c,d,_ = self._get_unit(train_data, self.init_num,self.adding_num)
            c_c0 = math.pow(c/c0, self.BETA)
            d_d0 = math.pow(d/d0, self.ALPHA)
            print("Finish calculating normaling factor")

            kc = kCenterGreedy(train_data)
            _ = kc.select_batch_with_cn(selected_idxs, self.MAX_HAUSDORFF, c_c0, d_d0, p=0.95)
            selected_idxs = kc.already_selected.astype("int")

            save_dir = os.path.join(self.data_provider.content_path, "selected_idxs")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open(os.path.join(save_dir,"selected_{}.json".format(t)), "w") as f:
                json.dump(selected_idxs.tolist(), f)
            print("select {:d} points".format(len(selected_idxs)))

            time_step_idxs_list.insert(0, np.arange(len(selected_idxs)).tolist())

            train_data = self.data_provider.train_representation(t).squeeze()
            train_data = train_data[selected_idxs]

            if self.b_n_epochs != 0:
                # select highly used border centers...
                border_centers = self.data_provider.border_representation(t)
                t_num = len(selected_idxs)
                b_num = len(border_centers)

                complex, sigmas_t1, rhos_t1, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                bw_complex, sigmas_t2, rhos_t2, _ = self._construct_boundary_wise_complex(train_data, border_centers)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, bw_complex)
                sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
                rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
                fitting_data = np.concatenate((train_data, border_centers), axis=0)
                # pred_model = self.data_provider.prediction_function(t)
                # attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
                attention_t = np.ones(fitting_data.shape)
            else:
                t_num = len(selected_idxs)
                b_num = 0

                complex, sigmas_t, rhos_t, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, None)
                fitting_data = np.copy(train_data)
                # pred_model = self.data_provider.prediction_function(t)
                # attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
                attention_t = np.ones(fitting_data.shape)


            if edge_to is None:
                edge_to = edge_to_t
                edge_from = edge_from_t
                weight = weight_t
                probs = weight_t / weight_t.max()
                feature_vectors = fitting_data
                attention = attention_t
                sigmas = sigmas_t
                rhos = rhos_t
                knn_indices = knn_idxs_t
                # npr = npr_t
                time_step_nums.insert(0, (t_num, b_num))
            else:
                # every round, we need to add len(data) to edge_to(as well as edge_from) index
                increase_idx = len(fitting_data)
                edge_to = np.concatenate((edge_to_t, edge_to + increase_idx), axis=0)
                edge_from = np.concatenate((edge_from_t, edge_from + increase_idx), axis=0)
                # normalize weight to be in range (0, 1)
                weight = np.concatenate((weight_t, weight), axis=0)
                probs_t = weight_t / weight_t.max()
                probs = np.concatenate((probs_t, probs), axis=0)
                sigmas = np.concatenate((sigmas_t, sigmas), axis=0)
                rhos = np.concatenate((rhos_t, rhos), axis=0)
                feature_vectors = np.concatenate((fitting_data, feature_vectors), axis=0)
                attention = np.concatenate((attention_t, attention), axis=0)
                knn_indices = np.concatenate((knn_idxs_t, knn_indices+increase_idx), axis=0)
                # npr = np.concatenate((npr_t, npr), axis=0)
                time_step_nums.insert(0, (t_num, b_num))

        return edge_to, edge_from, weight, feature_vectors, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention



class kcParallelSpatialEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, MAX_HAUSDORFF, ALPHA, BETA) -> None:
        super().__init__(data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors)
        self.MAX_HAUSDORFF = MAX_HAUSDORFF
        self.ALPHA = ALPHA
        self.BETA = BETA
    
    def _get_unit(self, data, adding_num=100):
        t0 = time.time()
        l = len(data)
        idxs = np.random.choice(np.arange(l), size=self.init_num, replace=False)

        id = IntrinsicDim(data)
        d0 = id.twonn_dimension_fast()

        kc = kCenterGreedy(data)
        _ = kc.select_batch_with_budgets(idxs, adding_num)
        c0 = kc.hausdorff()
        t1 = time.time()
        return c0, d0, "{:.1f}".format(t1-t0)
    
    def construct(self):
        """construct spatio-temporal complex and get edges

        Returns
        -------
        _type_
            _description_
        """

        # dummy input
        edge_to = None
        edge_from = None
        sigmas = None
        rhos = None
        weight = None
        probs = None
        feature_vectors = None
        attention = None
        knn_indices = None
        time_step_nums = list()
        time_step_idxs_list = list()# the list of selected idxs

        train_num = self.data_provider.train_num
        init_selected_idxs = np.random.choice(np.arange(train_num), size=self.init_num, replace=False)

        baseline_data = self.data_provider.train_representation(self.data_provider.e)
        baseline_data = baseline_data.reshape(len(baseline_data), -1)
        max_x = np.linalg.norm(baseline_data, axis=1).max()
        baseline_data = baseline_data/max_x
        
        c0,d0,_ = self._get_unit(baseline_data)

        # each time step
        for t in range(self.data_provider.e, self.data_provider.s - 1, -self.data_provider.p):
            print("=================+++={:d}=+++================".format(t))
            # load train data and border centers
            train_data = self.data_provider.train_representation(t)
            train_data = train_data.reshape(len(train_data), -1)

            # normalize data by max ||x||_2
            max_x = np.linalg.norm(train_data, axis=1).max()
            train_data = train_data/max_x

            # get normalization parameters for different epochs
            c,d,_ = self._get_unit(train_data)
            c_c0 = math.pow(c/c0, self.BETA)
            d_d0 = math.pow(d/d0, self.ALPHA)
            print("Finish calculating normaling factor")

            kc = kCenterGreedy(train_data)
            _ = kc.select_batch_with_cn(init_selected_idxs, self.MAX_HAUSDORFF, c_c0, d_d0, p=0.95)
            selected_idxs = kc.already_selected.astype("int")

            save_dir = os.path.join(self.data_provider.content_path, "selected_idxs")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open(os.path.join(save_dir,"selected_{}.json".format(t)), "w") as f:
                json.dump(selected_idxs.tolist(), f)
            print("select {:d} points".format(len(selected_idxs)))

            time_step_idxs_list.insert(0, selected_idxs)

            train_data = self.data_provider.train_representation(t)
            train_data = train_data[selected_idxs]
            
            if self.b_n_epochs != 0:
                # select highly used border centers...
                border_centers = self.data_provider.border_representation(t).squeeze()
                t_num = len(selected_idxs)
                b_num = len(border_centers)

                complex, sigmas_t1, rhos_t1, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                bw_complex, sigmas_t2, rhos_t2, _ = self._construct_boundary_wise_complex(train_data, border_centers)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, bw_complex)
                sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
                rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
                fitting_data = np.concatenate((train_data, border_centers), axis=0)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            else:
                t_num = len(selected_idxs)
                b_num = 0

                complex, sigmas_t, rhos_t, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, None)
                fitting_data = np.copy(train_data)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)

            if edge_to is None:
                edge_to = edge_to_t
                edge_from = edge_from_t
                weight = weight_t
                probs = weight_t / weight_t.max()
                feature_vectors = fitting_data
                attention = attention_t
                sigmas = sigmas_t
                rhos = rhos_t
                knn_indices = knn_idxs_t
                # npr = npr_t
                time_step_nums.insert(0, (t_num, b_num))
            else:
                # every round, we need to add len(data) to edge_to(as well as edge_from) index
                increase_idx = len(fitting_data)
                edge_to = np.concatenate((edge_to_t, edge_to + increase_idx), axis=0)
                edge_from = np.concatenate((edge_from_t, edge_from + increase_idx), axis=0)
                # normalize weight to be in range (0, 1)
                weight = np.concatenate((weight_t, weight), axis=0)
                probs_t = weight_t / weight_t.max()
                probs = np.concatenate((probs_t, probs), axis=0)
                sigmas = np.concatenate((sigmas_t, sigmas), axis=0)
                rhos = np.concatenate((rhos_t, rhos), axis=0)
                feature_vectors = np.concatenate((fitting_data, feature_vectors), axis=0)
                attention = np.concatenate((attention_t, attention), axis=0)
                knn_indices = np.concatenate((knn_idxs_t, knn_indices+increase_idx), axis=0)
                # npr = np.concatenate((npr_t, npr), axis=0)
                time_step_nums.insert(0, (t_num, b_num))

        return edge_to, edge_from, weight, feature_vectors, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention
    

class SingleEpochSpatialEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, iteration, s_n_epochs, b_n_epochs, n_neighbors) -> None:
        super().__init__(data_provider, 100, s_n_epochs, b_n_epochs, n_neighbors)
        self.iteration = iteration
    
    def construct(self):
        # load train data and border centers
        train_data = self.data_provider.train_representation(self.iteration)
        # selected = np.random.choice(len(train_data), int(0.3*len(train_data)), replace=False)
        # train_data = train_data[selected]

        if self.b_n_epochs > 0:
            border_centers = self.data_provider.border_representation(self.iteration).squeeze()
            complex, _, _, _ = self._construct_fuzzy_complex(train_data)
            bw_complex, _, _, _ = self._construct_boundary_wise_complex(train_data, border_centers)
            edge_to, edge_from, weight = self._construct_step_edge_dataset(complex, bw_complex)
            feature_vectors = np.concatenate((train_data, border_centers), axis=0)
            pred_model = self.data_provider.prediction_function(self.iteration)
            attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            # attention = np.zeros(feature_vectors.shape)
        elif self.b_n_epochs == 0:
            complex, _, _, _ = self._construct_fuzzy_complex(train_data)
            edge_to, edge_from, weight = self._construct_step_edge_dataset(complex, None)
            feature_vectors = np.copy(train_data)
            pred_model = self.data_provider.prediction_function(self.iteration)
            attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)            
            # attention = np.zeros(feature_vectors.shape)
        else: 
            raise Exception("Illegal border edges proposion!")
            
        return edge_to, edge_from, weight, feature_vectors, attention
    
    def record_time(self, save_dir, file_name, operation, t):
        file_path = os.path.join(save_dir, file_name+".json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                ti = json.load(f)
        else:
            ti = dict()
        if operation not in ti.keys():
            ti[operation] = dict()
        ti[operation][str(self.iteration)] = t
        with open(file_path, "w") as f:
            json.dump(ti, f)


class SingleEpochSpatialInterpolatedEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, iteration, s_n_epochs, b_n_epochs, n_neighbors) -> None:
        super().__init__(data_provider, 100, s_n_epochs, b_n_epochs, n_neighbors)
        self.iteration = iteration
       
    
    def construct(self):
        # load train data and border centers
        train_data = self.data_provider.train_representation(self.iteration)
        train_data = train_data.reshape(len(train_data), -1)
        # X = train_data
        # num_pairs = 2000
        # num_interpolations_per_pair = 2  # 每一对样本生成2个插值样本，这样总共就会生成40000个插值样本
        # labels = self.data_provider.train_labels(self.iteration)

        # interpolated_X_div = self.generate_interpolated_samples(train_data,labels,self.get_conf,num_pairs)
        # # interpolated_X = np.empty((num_pairs * num_interpolations_per_pair, 512))
        # # for i in range(num_pairs):
        # #     sample1 = X[np.random.randint(0, X.shape[0])]
        # #     sample2 = X[np.random.randint(0, X.shape[0])]
        # #     interpolated_samples = self.interpolate_samples(sample1, sample2, num_interpolations_per_pair)
        # #     interpolated_X[i*num_interpolations_per_pair : (i+1)*num_interpolations_per_pair] = interpolated_samples
 
        # confidence_bins = np.linspace(0.5, 1, 6)[1:-1]  # 置信度区间
        # interpolated_X = np.concatenate([np.array(interpolated_X_div[bin]) for bin in confidence_bins])

        # np.save(os.path.join(self.data_provider.content_path, "Model", "Epoch_{}".format(self.iteration),"interpolated_X.npy"), interpolated_X)
        interpolated_X = np.load(os.path.join(self.data_provider.content_path, "Model", "Epoch_{}".format(self.iteration),"GAN.npy"))
        train_data = np.concatenate((train_data, interpolated_X), axis=0)

        print("train_data111",train_data.shape)
        # selected = np.random.choice(len(train_data), int(0.3*len(train_data)), replace=False)
        # train_data = train_data[selected]

        if self.b_n_epochs > 0:
            border_centers = self.data_provider.border_representation(self.iteration).squeeze()
            complex, _, _, _ = self._construct_fuzzy_complex(train_data)
            bw_complex, _, _, _ = self._construct_boundary_wise_complex(train_data, border_centers)
            edge_to, edge_from, weight = self._construct_step_edge_dataset(complex, bw_complex)
            feature_vectors = np.concatenate((train_data, border_centers), axis=0)
            pred_model = self.data_provider.prediction_function(self.iteration)
            attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            # attention = np.zeros(feature_vectors.shape)
        elif self.b_n_epochs == 0:
            complex, _, _, _ = self._construct_fuzzy_complex(train_data)
            edge_to, edge_from, weight = self._construct_step_edge_dataset(complex, None)
            feature_vectors = np.copy(train_data)
            pred_model = self.data_provider.prediction_function(self.iteration)
            attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)            
            # attention = np.zeros(feature_vectors.shape)
        else: 
            raise Exception("Illegal border edges proposion!")
            
        return edge_to, edge_from, weight, feature_vectors, attention
    
    # def interpolate_samples(self, sample1, sample2, num_interpolations):
    #     t_values = np.linspace(0, 1, num_interpolations+2)[1:-1] # 去掉0和1，保证生成的样本不包括原始样本
    #     interpolated_samples = np.empty((num_interpolations, sample1.shape[0]))

    #     for i, t in enumerate(t_values):
    #         interpolated_samples[i] = t * sample1 + (1 - t) * sample2
        
    #     return interpolated_samples
    def interpolate_samples(self, sample1, sample2, t):
        return t * sample1 + (1 - t) * sample2

    def select_samples_from_different_classes(self, X, labels):
        classes = np.unique(labels)
        selected_samples = []
        for i in range(len(classes)-1):
            for j in range(i+1, len(classes)):
                samples_class_i = X[labels == classes[i]]
                samples_class_j = X[labels == classes[j]]
                sample1 = samples_class_i[np.random.choice(samples_class_i.shape[0])]
                sample2 = samples_class_j[np.random.choice(samples_class_j.shape[0])]
                selected_samples.append((sample1, sample2))
        return selected_samples
    def get_conf(self, epoch, interpolated_X):
        predctions = self.data_provider.get_pred(epoch, interpolated_X)
        scores = np.amax(softmax(predctions, axis=1), axis=1)
        return scores

    def generate_interpolated_samples(self, X, labels, get_conf, num_interpolations_per_bin):
        selected_samples = self.select_samples_from_different_classes(X, labels)

        # confidence_bins = np.linspace(0, 1, 11)[1:-1]  # 置信度区间
        confidence_bins = np.linspace(0.5, 1, 6)[1:-1]  # 置信度区间
    
        interpolated_X = {bin: [] for bin in confidence_bins}  # 储存插值样本的字典，键是置信度区间，值是插值样本

        # 执行循环，直到每个置信度区间都有足够的插值样本
        while min([len(samples) for samples in interpolated_X.values()]) < num_interpolations_per_bin:
            batch_samples = []
            for _ in range(100):
                # 选择两个样本并生成插值样本
                sample1, sample2 = selected_samples[np.random.choice(len(selected_samples))]
                t = np.random.rand()  # 随机选择插值参数t
                interpolated_sample = self.interpolate_samples(sample1, sample2, t)
                batch_samples.append(interpolated_sample)

            # 计算插值样本的置信度并根据置信度将其分配到相应的区间
            confidences = get_conf(self.iteration, np.array(batch_samples))
            for i, confidence in enumerate(confidences):
                for bin in confidence_bins:
                    if confidence < bin:
                        interpolated_X[bin].append(batch_samples[i])
                        # print("interpolated_X",len(interpolated_X[0.6]),len(interpolated_X[0.7]))
                        break

        return interpolated_X
    
    
    def record_time(self, save_dir, file_name, operation, t):
        file_path = os.path.join(save_dir, file_name+".json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                ti = json.load(f)
        else:
            ti = dict()
        if operation not in ti.keys():
            ti[operation] = dict()
        ti[operation][str(self.iteration)] = t
        with open(file_path, "w") as f:
            json.dump(ti, f)

class SingleEpochSpatialEdgeMapperConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, model, iteration, s_n_epochs, b_n_epochs, n_neighbors) -> None:
        super().__init__(data_provider, 100, s_n_epochs, b_n_epochs, n_neighbors)
        self.iteration = iteration
        self.mapper = KeplerMapper(verbose=1)
        self.model = model
    
    def construct(self):
        # load train data and border centers
        train_data = self.data_provider.train_representation(self.iteration)
        #### TODO sampling method
        # selected = np.random.choice(len(train_data), int(0.2*len(train_data)), replace=False)
        selected = self.uncertainty_sampling(self.iteration, train_data, int(0.1*len(train_data)))
        train_data = train_data[selected]

        if self.b_n_epochs > 0:
            fuzzy_complex, _, _, _  = self._construct_fuzzy_complex(train_data)
            border_centers = self.data_provider.border_representation(self.iteration).squeeze()
            complex = self._construct_mapper_complex(train_data, [self.decison_boundary_distance_filter, self.density_filter_function], self.iteration, self.model)
            bw_complex, _, _, _ = self._construct_boundary_wise_complex(train_data, border_centers)
            # bw_complex = self._construct_boundary_wise_complex_mapper(train_data, border_centers, [self.decison_boundary_distance_filter,self.density_filter_function] ,self.iteration, self.model )
            # edge_to, edge_from, weight = self._construct_step_edge_dataset(complex, bw_complex)
            edge_to, edge_from, weight = self.construct_edge_dataset(complex, fuzzy_complex, bw_complex)
            feature_vectors = np.concatenate((train_data, border_centers), axis=0)
            attention = np.zeros(feature_vectors.shape)
        elif self.b_n_epochs == 0:
            # fuzzy_complex, _, _, _  = self._construct_fuzzy_complex(train_data)
            complex = self._construct_mapper_complex(train_data, [self.decison_boundary_distance_filter],self.iteration, self.model)
            # edge_to, edge_from, weight = self.construct_edge_dataset(complex, fuzzy_complex)
            edge_to, edge_from, weight = self._construct_step_edge_dataset(complex,None)
            feature_vectors = np.copy(train_data)          
            # attention = np.zeros(feature_vectors.shape)
            pred_model = self.data_provider.prediction_function(self.iteration)
            attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)      
        else: 
            raise Exception("Illegal border edges proposion!")
            
        return edge_to, edge_from, weight, feature_vectors, attention
    
    def record_time(self, save_dir, file_name, operation, t):
        file_path = os.path.join(save_dir, file_name+".json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                ti = json.load(f)
        else:
            ti = dict()
        if operation not in ti.keys():
            ti[operation] = dict()
        ti[operation][str(self.iteration)] = t
        with open(file_path, "w") as f:
            json.dump(ti, f)

    def uncertainty_sampling(self, epoch, data, num_samples):
        print("num_samples",num_samples)
        # Get the model's predictions for each data point
        predictions = self.data_provider.get_pred(epoch, data) # Make sure your model has 'predict_proba' method to output probabilities

        # Calculate the entropy of the predictions
        prediction_entropies = entropy(predictions.T)

        # Get the indices of the data points with highest uncertainty
        uncertain_indices = np.argsort(prediction_entropies)[-num_samples:]

        # Return the selected data points
        return uncertain_indices

class SingleEpochSpatialEdgeConstructorForGrid(SpatialEdgeConstructor):
    def __init__(self, data_provider, grid_high, iteration, s_n_epochs, b_n_epochs, n_neighbors,only_grid=False) -> None:
        super().__init__(data_provider, 100, s_n_epochs, b_n_epochs, n_neighbors)
        self.iteration = iteration
        self.grid_high = grid_high
        self.only_grid = only_grid
    
    def construct(self):
        # load train data and border centers
        train_data = self.data_provider.train_representation(self.iteration)
        train_data = np.concatenate((train_data, self.grid_high), axis=0)
       
        if self.only_grid == True: 
            train_data = self.grid_high

        print("train_data",train_data.shape, "if only:", self.only_grid)

        # selected = np.random.choice(len(train_data), int(0.9*len(train_data)), replace=False)
        # train_data = train_data[selected]

        if self.b_n_epochs > 0:
            border_centers = self.data_provider.border_representation(self.iteration).squeeze()
            complex, _, _, _ = self._construct_fuzzy_complex(train_data)
            bw_complex, _, _, _ = self._construct_boundary_wise_complex(train_data, border_centers)
            edge_to, edge_from, weight = self._construct_step_edge_dataset(complex, bw_complex)
            feature_vectors = np.concatenate((train_data, border_centers), axis=0)
            # pred_model = self.data_provider.prediction_function(self.iteration)
            # attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            attention = np.zeros(feature_vectors.shape)
        elif self.b_n_epochs == 0:
            complex, _, _, _ = self._construct_fuzzy_complex(train_data)
            edge_to, edge_from, weight = self._construct_step_edge_dataset(complex, None)
            feature_vectors = np.copy(train_data)
            # pred_model = self.data_provider.prediction_function(self.iteration)
            # attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)            
            attention = np.zeros(feature_vectors.shape)
        else: 
            raise Exception("Illegal border edges proposion!")
            
        return edge_to, edge_from, weight, feature_vectors, attention
    
    def record_time(self, save_dir, file_name, operation, t):
        file_path = os.path.join(save_dir, file_name+".json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                ti = json.load(f)
        else:
            ti = dict()
        if operation not in ti.keys():
            ti[operation] = dict()
        ti[operation][str(self.iteration)] = t
        with open(file_path, "w") as f:
            json.dump(ti, f)


class kcHybridSpatialEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, MAX_HAUSDORFF, ALPHA, BETA, init_idxs=None, init_embeddings=None, c0=None, d0=None) -> None:
        super().__init__(data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors)
        self.MAX_HAUSDORFF = MAX_HAUSDORFF
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.init_idxs = init_idxs
        self.init_embeddings = init_embeddings
        self.c0 = c0
        self.d0 = d0
    
    def _get_unit(self, data, adding_num=100):
        t0 = time.time()
        l = len(data)
        idxs = np.random.choice(np.arange(l), size=self.init_num, replace=False)

        id = IntrinsicDim(data)
        d0 = id.twonn_dimension_fast()

        kc = kCenterGreedy(data)
        _ = kc.select_batch_with_budgets(idxs, adding_num)
        c0 = kc.hausdorff()
        t1 = time.time()
        return c0, d0, "{:.1f}".format(t1-t0)
    
    def construct(self):
        """construct spatio-temporal complex and get edges

        Returns
        -------
        _type_
            _description_
        """

        # dummy input
        edge_to = None
        edge_from = None
        sigmas = None
        rhos = None
        weight = None
        probs = None
        feature_vectors = None
        attention = None
        knn_indices = None
        time_step_nums = list()
        time_step_idxs_list = list()
        coefficient = None
        embedded = None

        train_num = self.data_provider.train_num
        # load init_idxs
        if self.init_idxs is None:
            selected_idxs = np.random.choice(np.arange(train_num), size=self.init_num, replace=False)
        else:
            selected_idxs = np.copy(self.init_idxs)
        
        # load c0 d0
        if self.c0 is None or self.d0 is None:
            baseline_data = self.data_provider.train_representation(self.data_provider.e)
            max_x = np.linalg.norm(baseline_data, axis=1).max()
            baseline_data = baseline_data/max_x
            c0,d0,_ = self._get_unit(baseline_data)
            save_dir = os.path.join(self.data_provider.content_path, "selected_idxs")
            os.system("mkdir -p {}".format(save_dir))
            with open(os.path.join(save_dir,"baseline.json"), "w") as f:
                json.dump([float(c0), float(d0)], f)
            print("save c0 and d0 to disk!")
            
        else:
            c0 = self.c0
            d0 = self.d0

        # each time step
        for t in range(self.data_provider.e, self.data_provider.s - 1, -self.data_provider.p):
            print("=================+++={:d}=+++================".format(t))
            # load train data and border centers
            train_data = self.data_provider.train_representation(t).squeeze()

            # normalize data by max ||x||_2
            max_x = np.linalg.norm(train_data, axis=1).max()
            train_data = train_data/max_x

            # get normalization parameters for different epochs
            c,d,_ = self._get_unit(train_data)
            c_c0 = math.pow(c/c0, self.BETA)
            d_d0 = math.pow(d/d0, self.ALPHA)
            print("Finish calculating normaling factor")

            kc = kCenterGreedy(train_data)
            _, hausd = kc.select_batch_with_cn(selected_idxs, self.MAX_HAUSDORFF, c_c0, d_d0, p=0.95, return_min=True)
            selected_idxs = kc.already_selected.astype("int")

            save_dir = os.path.join(self.data_provider.content_path, "selected_idxs")
            os.system("mkdir -p {}".format(save_dir))
            with open(os.path.join(save_dir,"selected_{}.json".format(t)), "w") as f:
                json.dump(selected_idxs.tolist(), f)
            print("select {:d} points".format(len(selected_idxs)))

            time_step_idxs_list.insert(0, selected_idxs)

            train_data = self.data_provider.train_representation(t).squeeze()
            train_data = train_data[selected_idxs]

            if self.b_n_epochs != 0:
                # select highly used border centers...
                border_centers = self.data_provider.border_representation(t).squeeze()
                t_num = len(selected_idxs)
                b_num = len(border_centers)

                complex, sigmas_t1, rhos_t1, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                bw_complex, sigmas_t2, rhos_t2, _ = self._construct_boundary_wise_complex(train_data, border_centers)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, bw_complex)
                sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
                rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
                fitting_data = np.concatenate((train_data, border_centers), axis=0)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            else:
                t_num = len(selected_idxs)
                b_num = 0

                complex, sigmas_t, rhos_t, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, None)
                fitting_data = np.copy(train_data)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            
            if edge_to is None:
                edge_to = edge_to_t
                edge_from = edge_from_t
                weight = weight_t
                probs = weight_t / weight_t.max()
                feature_vectors = fitting_data
                attention = attention_t
                sigmas = sigmas_t
                rhos = rhos_t
                knn_indices = knn_idxs_t
                # npr = npr_t
                time_step_nums.insert(0, (t_num, b_num))

                if self.init_embeddings is None:
                    coefficient = np.zeros(len(feature_vectors))
                    embedded = np.zeros((len(feature_vectors), 2))
                else:
                    coefficient = np.zeros(len(feature_vectors))
                    coefficient[:len(self.init_embeddings)] = 1
                    embedded = np.zeros((len(feature_vectors), 2))
                    embedded[:len(self.init_embeddings)] = self.init_embeddings

            else:
                # every round, we need to add len(data) to edge_to(as well as edge_from) index
                increase_idx = len(fitting_data)
                edge_to = np.concatenate((edge_to_t, edge_to + increase_idx), axis=0)
                edge_from = np.concatenate((edge_from_t, edge_from + increase_idx), axis=0)
                # normalize weight to be in range (0, 1)
                weight = np.concatenate((weight_t, weight), axis=0)
                probs_t = weight_t / weight_t.max()
                probs = np.concatenate((probs_t, probs), axis=0)
                sigmas = np.concatenate((sigmas_t, sigmas), axis=0)
                rhos = np.concatenate((rhos_t, rhos), axis=0)
                feature_vectors = np.concatenate((fitting_data, feature_vectors), axis=0) 
                attention = np.concatenate((attention_t, attention), axis=0)
                knn_indices = np.concatenate((knn_idxs_t, knn_indices+increase_idx), axis=0)
                # npr = np.concatenate((npr_t, npr), axis=0)
                time_step_nums.insert(0, (t_num, b_num))
                coefficient = np.concatenate((np.zeros(len(fitting_data)), coefficient), axis=0)
                embedded = np.concatenate((np.zeros((len(fitting_data), 2)), embedded), axis=0)

                

        return edge_to, edge_from, weight, feature_vectors, embedded, coefficient, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention, (c0, d0)


class kcHybridDenseALSpatialEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, MAX_HAUSDORFF, ALPHA, BETA, iteration, init_idxs=None, init_embeddings=None, c0=None, d0=None) -> None:
        super().__init__(data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors)
        self.MAX_HAUSDORFF = MAX_HAUSDORFF
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.init_idxs = init_idxs
        self.init_embeddings = init_embeddings
        self.c0 = c0
        self.d0 = d0
        self.iteration = iteration
    
    def _get_unit(self, data, adding_num=100):
        t0 = time.time()
        l = len(data)
        idxs = np.random.choice(np.arange(l), size=self.init_num, replace=False)

        id = IntrinsicDim(data)
        d0 = id.twonn_dimension_fast()

        kc = kCenterGreedy(data)
        _ = kc.select_batch_with_budgets(idxs, adding_num)
        c0 = kc.hausdorff()
        t1 = time.time()
        return c0, d0, "{:.1f}".format(t1-t0)
    
    def construct(self):
        """construct spatio-temporal complex and get edges

        Returns
        -------
        _type_
            _description_
        """

        # dummy input
        edge_to = None
        edge_from = None
        sigmas = None
        rhos = None
        weight = None
        probs = None
        feature_vectors = None
        attention = None
        knn_indices = None
        time_step_nums = list()
        time_step_idxs_list = list()
        coefficient = None
        embedded = None

        train_num = self.data_provider.label_num(self.iteration)
        # load init_idxs
        if self.init_idxs is None:
            selected_idxs = np.random.choice(np.arange(train_num), size=self.init_num, replace=False)
        else:
            selected_idxs = np.copy(self.init_idxs)
        
        # load c0 d0
        if self.c0 is None or self.d0 is None:
            baseline_data = self.data_provider.train_representation_lb(self.iteration, self.data_provider.e)
            max_x = np.linalg.norm(baseline_data, axis=1).max()
            baseline_data = baseline_data/max_x
            c0,d0,_ = self._get_unit(baseline_data)
            save_dir = os.path.join(self.data_provider.content_path, "Model", "Iteration_{}".format(self.iteration), "selected_idxs")
            os.system("mkdir -p {}".format(save_dir))
            with open(os.path.join(save_dir,"baseline.json"), "w") as f:
                json.dump([float(c0), float(d0)], f)
            print("save c0 and d0 to disk!")
            
        else:
            c0 = self.c0
            d0 = self.d0

        # each time step
        for t in range(self.data_provider.e, self.data_provider.s - 1, -self.data_provider.p):
            print("=================+++={:d}=+++================".format(t))
            # load train data and border centers
            train_data = self.data_provider.train_representation_lb(self.iteration, t).squeeze()

            # normalize data by max ||x||_2
            max_x = np.linalg.norm(train_data, axis=1).max()
            train_data = train_data/max_x

            # get normalization parameters for different epochs
            c,d,_ = self._get_unit(train_data)
            c_c0 = math.pow(c/c0, self.BETA)
            d_d0 = math.pow(d/d0, self.ALPHA)
            print("Finish calculating normaling factor")

            kc = kCenterGreedy(train_data)
            _, hausd = kc.select_batch_with_cn(selected_idxs, self.MAX_HAUSDORFF, c_c0, d_d0, p=0.95, return_min=True)
            selected_idxs = kc.already_selected.astype("int")

            save_dir = os.path.join(self.data_provider.content_path, "Model", "Iteration_{}".format(self.iteration), "selected_idxs")
            os.system("mkdir -p {}".format(save_dir))
            with open(os.path.join(save_dir,"selected_{}.json".format(t)), "w") as f:
                json.dump(selected_idxs.tolist(), f)
            print("select {:d} points".format(len(selected_idxs)))

            time_step_idxs_list.insert(0, selected_idxs)

            train_data = self.data_provider.train_representation_lb(self.iteration, t).squeeze()
            train_data = train_data[selected_idxs]

            if self.b_n_epochs != 0:
                # select highly used border centers...
                border_centers = self.data_provider.border_representation(self.iteration, t).squeeze()
                t_num = len(selected_idxs)
                b_num = len(border_centers)

                complex, sigmas_t1, rhos_t1, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                bw_complex, sigmas_t2, rhos_t2, _ = self._construct_boundary_wise_complex(train_data, border_centers)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, bw_complex)
                sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
                rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
                fitting_data = np.concatenate((train_data, border_centers), axis=0)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            else:
                t_num = len(selected_idxs)
                b_num = 0

                complex, sigmas_t, rhos_t, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, None)
                fitting_data = np.copy(train_data)
                pred_model = self.data_provider.prediction_function(self.iteration,t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            
            if edge_to is None:
                edge_to = edge_to_t
                edge_from = edge_from_t
                weight = weight_t
                probs = weight_t / weight_t.max()
                feature_vectors = fitting_data
                attention = attention_t
                sigmas = sigmas_t
                rhos = rhos_t
                knn_indices = knn_idxs_t
                # npr = npr_t
                time_step_nums.insert(0, (t_num, b_num))

                if self.init_embeddings is None:
                    coefficient = np.zeros(len(feature_vectors))
                    embedded = np.zeros((len(feature_vectors), 2))
                else:
                    coefficient = np.zeros(len(feature_vectors))
                    coefficient[:len(self.init_embeddings)] = 1
                    embedded = np.zeros((len(feature_vectors), 2))
                    embedded[:len(self.init_embeddings)] = self.init_embeddings

            else:
                # every round, we need to add len(data) to edge_to(as well as edge_from) index
                increase_idx = len(fitting_data)
                edge_to = np.concatenate((edge_to_t, edge_to + increase_idx), axis=0)
                edge_from = np.concatenate((edge_from_t, edge_from + increase_idx), axis=0)
                # normalize weight to be in range (0, 1)
                weight = np.concatenate((weight_t, weight), axis=0)
                probs_t = weight_t / weight_t.max()
                probs = np.concatenate((probs_t, probs), axis=0)
                sigmas = np.concatenate((sigmas_t, sigmas), axis=0)
                rhos = np.concatenate((rhos_t, rhos), axis=0)
                feature_vectors = np.concatenate((fitting_data, feature_vectors), axis=0) 
                attention = np.concatenate((attention_t, attention), axis=0)
                knn_indices = np.concatenate((knn_idxs_t, knn_indices+increase_idx), axis=0)
                # npr = np.concatenate((npr_t, npr), axis=0)
                time_step_nums.insert(0, (t_num, b_num))
                coefficient = np.concatenate((np.zeros(len(fitting_data)), coefficient), axis=0)
                embedded = np.concatenate((np.zeros((len(fitting_data), 2)), embedded), axis=0)

        return edge_to, edge_from, weight, feature_vectors, embedded, coefficient, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention, (c0, d0)


class tfEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, s_n_epochs, b_n_epochs, n_neighbors) -> None:
        super().__init__(data_provider, 100, s_n_epochs, b_n_epochs, n_neighbors)
    # override
    def _construct_step_edge_dataset(self, vr_complex, bw_complex):
        """
        construct the mixed edge dataset for one time step
            connect border points and train data(both direction)
        :param vr_complex: Vietoris-Rips complex
        :param bw_complex: boundary-augmented complex
        :param n_epochs: the number of epoch that we iterate each round
        :return: edge dataset
        """
        # get data from graph
        _, vr_head, vr_tail, vr_weight, _ = get_graph_elements(vr_complex, self.s_n_epochs)
        epochs_per_sample = make_epochs_per_sample(vr_weight, 10)
        vr_head = np.repeat(vr_head, epochs_per_sample.astype("int"))
        vr_tail = np.repeat(vr_tail, epochs_per_sample.astype("int"))
        vr_weight = np.repeat(vr_weight, epochs_per_sample.astype("int"))
        
        # get data from graph
        if self.b_n_epochs == 0:
            return vr_head, vr_tail, vr_weight
        else:
            _, bw_head, bw_tail, bw_weight, _ = get_graph_elements(bw_complex, self.b_n_epochs)
            b_epochs_per_sample = make_epochs_per_sample(bw_weight, self.b_n_epochs)
            bw_head = np.repeat(bw_head, b_epochs_per_sample.astype("int"))
            bw_tail = np.repeat(bw_tail, b_epochs_per_sample.astype("int"))
            bw_weight = np.repeat(bw_weight, epochs_per_sample.astype("int"))
            head = np.concatenate((vr_head, bw_head), axis=0)
            tail = np.concatenate((vr_tail, bw_tail), axis=0)
            weight = np.concatenate((vr_weight, bw_weight), axis=0)
        return head, tail, weight
        
    def construct(self, prev_iteration, iteration):
        '''
        If prev_iteration<epoch_start, then temporal loss will be 0
        '''
        train_data = self.data_provider.train_representation(iteration)
        if prev_iteration > self.data_provider.s:
            prev_data = self.data_provider.train_representation(prev_iteration)
        else:
            prev_data = None
        n_rate = find_neighbor_preserving_rate(prev_data, train_data, self.n_neighbors)
        if self.b_n_epochs > 0:
            border_centers = self.data_provider.border_representation(iteration).squeeze()
            complex, _, _, _ = self._construct_fuzzy_complex(train_data)
            bw_complex, _, _, _ = self._construct_boundary_wise_complex(train_data, border_centers)
            edges_to_exp, edges_from_exp, weights_exp = self._construct_step_edge_dataset(complex, bw_complex)
            feature_vectors = np.concatenate((train_data, border_centers), axis=0)
            # pred_model = self.data_provider.prediction_function(self.iteration)
            # attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            attention = np.zeros(feature_vectors.shape)

        elif self.b_n_epochs == 0:
            complex, _, _, _ = self._construct_fuzzy_complex(train_data)
            edges_to_exp, edges_from_exp, weights_exp = self._construct_step_edge_dataset(complex, None)
            feature_vectors = np.copy(train_data)
            # pred_model = self.data_provider.prediction_function(self.iteration)
            # attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)            
            attention = np.zeros(feature_vectors.shape)
        else: 
            raise Exception("Illegal border edges proposion!")
            
        return edges_to_exp, edges_from_exp, weights_exp, feature_vectors, attention, n_rate