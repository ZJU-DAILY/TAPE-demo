import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL.features import features
from networkx.algorithms.core import core_number
from scipy.cluster.hierarchy import single
from sympy import sequence
from torch.utils.data import Dataset, DataLoader
import random
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
from collections import deque
from pympler import asizeof
from scipy.sparse import lil_matrix
import time
import cProfile
import tracemalloc
from torch_geometric.data import TemporalData
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models.dimenet import triplets
from model import AdapterTemporalGNN
from torch.nn import init
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
import heapq
from sklearn.cluster import KMeans, DBSCAN
import math
import matplotlib.pyplot as plt
import time
import networkx as nx
from MLP_search import *


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = "cpu"

print(f"Using device: {device}")

# Tree
dataset_name = "collegemsg"
# dataset_name = 'mathoverflow'
# dataset_name = 'superuser'
# dataset_name = 'youtube'
# dataset_name = 'wikitalk'
# dataset_name = 'flickr'
# dataset_name = "stackoverflow"
num_vertex = 0
num_edge = 0
num_timestamp = 0
time_edge = {}
num_core_number = 0
vertex_core_numbers = []
time_range_core_number = defaultdict(dict)
temporal_graph = []
time_range_layers = []
time_range_set = set()
max_time_range_layers = []
min_time_range_layers = []
sequence_features1_matrix = torch.empty(0, 0, 0)
model_time_range_layers = []
partition = 4
max_range = 45
max_degree = 0
max_layer_id = 0
k_core_conductance = 0.0
k_core_density = 0.0
k_core_num = 0
root = None

inter_time = 0.0

# GNN
temporal_graph_pyg = TemporalData()
subgraph_k_hop_cache = {}
filtered_temporal_graph_pyg = TemporalData()

node_in_channels = 8
node_out_channels = 16
learning_rate = 0.001
epochs = 200
batch_size = 8
k_hop = 5
positive_hop = 3
edge_dim = 8
test_result_list = []


def read_temporal_graph():
    print("Loading the graph...")
    filename = f"../datasets/{dataset_name}.txt"
    global num_vertex, num_edge, num_timestamp, time_edge, temporal_graph, max_degree, temporal_graph_pyg
    time_edge = defaultdict(set)
    # temporal_graph = defaultdict(default_dict_factory)
    temporal_graph = defaultdict(lambda: defaultdict(list))

    with open(filename, "r") as f:
        first_line = True
        for line in f:
            if first_line:
                num_vertex, num_edge, num_timestamp = map(int, line.strip().split())
                first_line = False
                continue
            v1, v2, t = map(int, line.strip().split())
            if v1 == v2:
                continue
            if v1 > v2:
                v1, v2 = v2, v1
            time_edge[t].add((v1, v2))

            temporal_graph[v1][t].append(v2)
            temporal_graph[v2][t].append(v1)

    total_size = asizeof.asizeof(temporal_graph)
    print(f"temporal_graph 占用的内存大小为 {total_size / (1024 ** 2):.2f} MB")

    edge_to_timestamps = {}
    for t, edges in time_edge.items():
        for src, dst in edges:

            edge_to_timestamps.setdefault((src, dst), []).append(t)
            edge_to_timestamps.setdefault((dst, src), []).append(t)

    edge_index = []
    edge_attr = []

    for (src, dst), timestamps in edge_to_timestamps.items():
        edge_index.append([src, dst])
        edge_attr.append(timestamps)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    max_timestamps = max(len(ts) for ts in edge_attr)
    edge_attr = [ts + [0] * (max_timestamps - len(ts)) for ts in edge_attr]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    temporal_graph_pyg = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_vertex,
    )

    temporal_graph_pyg = temporal_graph_pyg.to(device)


def read_core_number():
    print("Loading the core number...")
    global num_core_number, vertex_core_numbers
    vertex_core_numbers = [{} for _ in range(num_vertex)]
    core_number_filename = f"../datasets/{dataset_name}-core_number.txt"
    with open(core_number_filename, "r") as f:
        first_line = True
        for line in f:
            if first_line:
                num_core_number = int(line.strip())
                first_line = False
                continue
            range_part, core_numbers_part = line.split(" ", 1)
            range_start, range_end = map(int, range_part.strip("[]").split(","))
            is_node_range = False
            if (range_start, range_end) in time_range_set:
                is_node_range = True
            for pair in core_numbers_part.split():
                vertex, core_number = map(int, pair.split(":"))
                if range_start == range_end:
                    vertex_core_numbers[vertex][range_start] = core_number
                if is_node_range:
                    time_range_core_number[(range_start, range_end)][
                        vertex
                    ] = core_number


def get_timerange_layers():
    global time_range_layers, min_time_range_layers, max_time_range_layers
    temp_range = num_timestamp
    while temp_range > max_range:
        temp_range = temp_range // partition
    layer_id = 0
    while temp_range < num_timestamp:
        range_start = 0
        time_range_layers.append([])
        temp_max_range = 0
        temp_min_range = num_timestamp
        while range_start < num_timestamp:
            range_end = range_start + temp_range - 1
            range_end = min(range_end, num_timestamp - 1)
            if range_end + temp_range > num_timestamp - 1:
                range_end = num_timestamp - 1
            time_range_layers[layer_id].append((range_start, range_end))
            time_range_set.add((range_start, range_end))
            if range_end - range_start + 1 > temp_max_range:
                temp_max_range = range_end - range_start + 1
            if range_end - range_start + 1 < temp_min_range:
                temp_min_range = range_end - range_start + 1
            range_start = range_end + 1
        max_time_range_layers.append(temp_max_range)
        min_time_range_layers.append(temp_min_range)
        temp_range = temp_range * partition
        layer_id = layer_id + 1
    time_range_layers.reverse()
    max_time_range_layers.reverse()
    min_time_range_layers.reverse()
    max_layer_id = layer_id


def construct_feature_matrix():
    print("Constructing the feature matrix...")
    global sequence_features1_matrix, indices_vertex_of_matrix
    indices = []
    values = []

    for v in range(num_vertex):
        # print(v)
        for t, neighbors in temporal_graph[v].items():
            core_number = vertex_core_numbers[v].get(t, 0)
            neighbor_count = len(neighbors)
            if core_number > 0:

                indices.append([v, t, 0])
                values.append(core_number)
            if neighbor_count > 0:
                indices.append([v, t, 1])
                values.append(neighbor_count)

    indices = torch.tensor(indices).T
    values = torch.tensor(values, dtype=torch.float32)

    sorted_order = torch.argsort(indices[0])
    sorted_indices = indices[:, sorted_order]
    sorted_values = values[sorted_order]

    sequence_features1_matrix = torch.sparse_coo_tensor(
        sorted_indices,
        sorted_values,
        size=(num_vertex, num_timestamp, 2),
        device=device,
    )

    sequence_features1_matrix = sequence_features1_matrix.coalesce()

    indices_size = indices.element_size() * indices.numel()
    values_size = values.element_size() * values.numel()
    total_size = indices_size + values_size
    print(f"feature matrix 占用的内存大小为 {total_size / (1024 ** 2):.2f} MB")


# @profile
def init_vertex_features(t_start, t_end, vertex_set, feature_dim, anchor):
    # vertex_indices = torch.tensor(list(vertex_set), device=device).sort().values
    vertex_indices = vertex_set

    indices = sequence_features1_matrix.indices()  # (n, nnz)
    values = sequence_features1_matrix.values()  # (nnz,)

    start_idx = torch.searchsorted(indices[0], vertex_indices, side="left")
    end_idx = torch.searchsorted(indices[0], vertex_indices, side="right")

    range_lengths = end_idx - start_idx
    total_indices = range_lengths.sum()

    range_offsets = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.long),
            range_lengths.cumsum(dim=0)[:-1],
        ]
    )
    flat_indices = torch.arange(
        total_indices, device=device
    ) - range_offsets.repeat_interleave(range_lengths)

    mask_indices = start_idx.repeat_interleave(range_lengths) + flat_indices

    vertex_mask = torch.zeros(indices.shape[1], dtype=torch.bool, device=device)
    vertex_mask[mask_indices] = True

    filtered_indices = indices[:, vertex_mask]
    filtered_values = values[vertex_mask]

    time_mask = (filtered_indices[1] >= t_start) & (filtered_indices[1] <= t_end)
    final_indices = filtered_indices[:, time_mask]
    final_values = filtered_values[time_mask]

    vertex_map = torch.zeros(vertex_indices.max() + 1, dtype=torch.long, device=device)
    vertex_map[vertex_indices] = torch.arange(len(vertex_indices), device=device)

    final_indices[0] = vertex_map[final_indices[0]]

    final_indices[1] -= t_start

    result_size = (
        len(vertex_indices),
        t_end - t_start + 1,
        sequence_features1_matrix.size(2),
    )
    result_sparse_tensor = torch.sparse_coo_tensor(
        final_indices, final_values, size=result_size
    )
    degree_tensor = result_sparse_tensor.to_dense()[:, :, 1]
    degree_tensor.to(device)

    degree_tensor = degree_tensor.unsqueeze(1)
    if degree_tensor.shape[2] < feature_dim - 1:
        degree_tensor = F.interpolate(
            degree_tensor, size=feature_dim - 2, mode="linear", align_corners=True
        )
    else:
        degree_tensor = F.adaptive_avg_pool1d(
            degree_tensor, output_size=feature_dim - 2
        )
    degree_tensor = degree_tensor.squeeze(1)

    core_number_values = torch.tensor(
        [time_range_core_number[(t_start, t_end)].get(v.item(), 0) for v in vertex_set],
        dtype=torch.float32,
        device=device,
    )

    vertex_features_matrix = torch.cat(
        [core_number_values.unsqueeze(1), degree_tensor], dim=1
    )

    matrix_max = torch.max(vertex_features_matrix)
    matrix_min = torch.min(vertex_features_matrix)
    vertex_features_matrix = (vertex_features_matrix - matrix_min) / (
        matrix_max - matrix_min + 1e-6
    )
    query_feature = torch.zeros(len(vertex_set), 1, device=device)
    query_feature[vertex_map[anchor]][0] = 1
    vertex_features_matrix = torch.cat([query_feature, vertex_features_matrix], dim=1)

    return vertex_features_matrix


def get_candidate_neighbors(
    center_vertex, k, t_start, t_end, filtered_temporal_graph, total_edge_weight
):
    global subgraph_k_hop_cache
    visited = set()
    visited.add(center_vertex)
    query_core_number = time_range_core_number[(t_start, t_end)].get(center_vertex, 0)
    queue = deque([(center_vertex, query_core_number, 0)])
    subgraph_result = set()
    core_number_condition = query_core_number * 0.5
    current_hop = 0
    total_core_number = 0
    tau = 0.5
    best_avg_core_number = 0
    while queue:
        top_vertex, neighbor_core_number, hop = queue.popleft()
        if hop > k:
            average_core_number = total_core_number / (len(subgraph_result) ** tau)
            # print(f"Average core number: {average_core_number}")
            break
        if hop > current_hop:
            average_core_number = total_core_number / (len(subgraph_result) ** tau)
            # print(f"Average core number: {average_core_number}")
            current_hop = hop
            if average_core_number > best_avg_core_number:
                best_avg_core_number = average_core_number
            else:
                break
        subgraph_result.add(top_vertex)
        if len(subgraph_result) > 8000:
            break
        total_core_number += neighbor_core_number
        for neighbor, edge_count, neighbor_core_number in filtered_temporal_graph[
            top_vertex
        ]:
            # if neighbor not in visited and neighbor_core_number >= core_number_condition:
            if neighbor not in visited:
                queue.append((neighbor, neighbor_core_number, hop + 1))
                visited.add(neighbor)
    # print(f"subgraph k-hop: {current_hop}")
    return subgraph_result


# @profile
def extract_subgraph(anchor, t_start, t_end, k, feature_dim):
    # generate filtered subgraph
    filtered_subgraph = {}
    vertex_connect_scores = {}
    total_edge_weight = 0
    for vertex in range(num_vertex):
        neighbor_time_edge_count = defaultdict(int)
        total_time_edge_count = 0
        for t, neighbors in temporal_graph[vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    neighbor_time_edge_count[neighbor] += 1
                    total_time_edge_count += 1
        neighbors_list = []
        for neighbor, count in neighbor_time_edge_count.items():
            core_number = time_range_core_number[(t_start, t_end)].get(neighbor, 0)
            neighbors_list.append((neighbor, count, core_number))
            if vertex < neighbor:
                total_edge_weight += count
        filtered_subgraph[vertex] = neighbors_list
        vertex_core_number = time_range_core_number[(t_start, t_end)].get(vertex, 0)
        vertex_connect_scores[vertex] = (
            vertex_core_number * total_time_edge_count / len(neighbors_list)
            if len(neighbors_list) != 0
            else 0
        )

    candidate_neighbors = get_candidate_neighbors(
        anchor, k, t_start, t_end, filtered_subgraph, total_edge_weight
    )

    candidate_neighbors = torch.tensor(sorted(candidate_neighbors), device=device)

    vertex_map = torch.full((num_vertex,), -1, dtype=torch.long)
    vertex_map[candidate_neighbors] = torch.arange(len(candidate_neighbors))
    vertex_map = vertex_map.to(device)

    mask = (vertex_map[temporal_graph_pyg.edge_index[0]] != -1) & (
        vertex_map[temporal_graph_pyg.edge_index[1]] != -1
    )
    mask = mask.to(device)

    # print(f"edge_index device: {temporal_graph_pyg.edge_index.device}")
    # print(f"mask device: {mask.device}")

    sub_edge_index = temporal_graph_pyg.edge_index[:, mask]

    sub_edge_index = vertex_map[sub_edge_index]

    sub_edge_attr = temporal_graph_pyg.edge_attr[mask]
    time_mask = (sub_edge_attr >= t_start) & (sub_edge_attr <= t_end)
    valid_edges = time_mask.any(dim=1)
    sub_edge_attr[~time_mask] = -1
    sub_edge_attr = sub_edge_attr[valid_edges]
    sub_edge_index = sub_edge_index[:, valid_edges]

    mask = sub_edge_attr != -1
    indices = mask.nonzero(as_tuple=True)
    origin_edge_attr = torch.zeros(
        sub_edge_attr.shape[0], t_end - t_start + 1, device=device
    )
    origin_edge_attr[indices[0], (sub_edge_attr[indices] - t_start).long()] = 1
    target_dim = edge_dim
    origin_edge_attr = F.adaptive_avg_pool1d(origin_edge_attr.unsqueeze(1), target_dim)
    sub_edge_attr = origin_edge_attr.squeeze(1)
    sub_edge_attr = torch.abs(torch.fft.fft(sub_edge_attr, dim=1))

    subgraph_pyg = Data(
        x=init_vertex_features(
            t_start, t_end, candidate_neighbors, feature_dim, anchor
        ),
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        num_nodes=len(candidate_neighbors),
        device=device,
    )
    return subgraph_pyg, vertex_map, candidate_neighbors


def temporal_test_GNN(distances, vertex_map, query_vertex, t_start, t_end, tau):
    # test with GNN using threshold
    visited = set()
    visited.add(query_vertex)
    result = set()
    queue = []
    heapq.heappush(queue, (0, query_vertex))

    mask = distances != 0
    temp_distances = distances[mask]
    distances = (distances - temp_distances.min()) / (
        distances.max() - temp_distances.min() + 1e-6
    )
    result_distance = []
    threshold = distances.mean().item()
    # threshold = distances.max().item()
    less_num = torch.sum(distances < threshold).item()

    while queue:
        if len(result) > 500:
            break
        distance, top_vertex = heapq.heappop(queue)
        result.add(top_vertex)
        result_distance.append(distance)
        if distance > threshold:
            break
        if len(result) > 1:
            alpha = np.cos((np.pi / 2) * (len(result) / (less_num**tau)))
            threshold = alpha * threshold + (1 - alpha) * (
                sum(result_distance) / (len(result_distance) - 1)
            )

        for t, neighbors in temporal_graph[top_vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if vertex_map[neighbor] != -1:
                            heapq.heappush(
                                queue,
                                (distances[vertex_map[neighbor]].item(), neighbor),
                            )
    return result


def optimize_neighbor_search(temporal_graph, t_start, t_end, center_vertices):
    neighbor_k_hop = set()
    visited = set()
    for center_vertex in center_vertices:
        queue = [(center_vertex, 0)]
        visited = {center_vertex}
        while queue:
            curr_vertex, curr_hop = queue.pop(0)
            neighbor_k_hop.add(curr_vertex)

            if curr_hop < 4 and curr_vertex in temporal_graph:
                for t, neighbors in temporal_graph[curr_vertex].items():
                    if t_start <= t <= t_end:
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append((neighbor, curr_hop + 1))

    return neighbor_k_hop


def get_candidate_neighbors_new(
    center_vertex,
    k,
    t_start,
    t_end,
    filtered_temporal_graph,
    total_edge_weight,
    time_range_core_number,
    subgraph_k_hop_cache,
):

    visited = set()
    visited.add(center_vertex)
    query_core_number = time_range_core_number[(t_start, t_end)].get(center_vertex, 0)
    queue = deque([(center_vertex, query_core_number, 0)])
    subgraph_result = set()
    core_number_condition = query_core_number * 0.2
    current_hop = 0
    total_core_number = 0
    tau = 0.5
    best_avg_core_number = 0
    while queue:
        top_vertex, neighbor_core_number, hop = queue.popleft()
        if hop > k:
            average_core_number = total_core_number / (len(subgraph_result) ** tau)
            # print(f"Average core number: {average_core_number}")
            break
        if hop > current_hop:
            average_core_number = total_core_number / (len(subgraph_result) ** tau)
            # print(f"Average core number: {average_core_number}")
            current_hop = hop
            if average_core_number > best_avg_core_number:
                best_avg_core_number = average_core_number
            else:
                break
        subgraph_result.add(top_vertex)
        if len(subgraph_result) > 2000:
            break
        total_core_number += neighbor_core_number
        if top_vertex in filtered_temporal_graph:
            for neighbor, edge_count, neighbor_core_number in filtered_temporal_graph[
                top_vertex
            ]:
                # if neighbor not in visited and neighbor_core_number >= core_number_condition:
                if neighbor not in visited:
                    queue.append((neighbor, neighbor_core_number, hop + 1))
                    visited.add(neighbor)
    return subgraph_result


def get_samples(
    center_vertex,
    k,
    t_start,
    t_end,
    filtered_temporal_graph,
    vertex_connect_scores,
    total_edge_weight,
    time_range_core_number,
    subgraph_k_hop_cache,
):

    candidates_neighbors = get_candidate_neighbors_new(
        center_vertex,
        k,
        t_start,
        t_end,
        filtered_temporal_graph,
        total_edge_weight,
        time_range_core_number,
        subgraph_k_hop_cache,
    )
    # print(f"K-hop Neighbors: {len(candidates_neighbors)}")

    positive_neighbors_list = []
    positive_neighbors = set()
    hard_negative_neighbors = set()
    visited = set()
    visited.add(center_vertex)
    queue = []
    edge_sum = 0
    heapq.heappush(queue, (0, center_vertex))
    query_vertex_core_number = time_range_core_number[(t_start, t_end)].get(
        center_vertex, 0
    )
    while queue:
        if edge_sum > len(candidates_neighbors) * 1.5:
            break
        _, top_vertex = heapq.heappop(queue)
        v_core_number = time_range_core_number[(t_start, t_end)].get(top_vertex, 0)
        if v_core_number >= query_vertex_core_number:
            hard_negative_neighbors.add(top_vertex)
        positive_neighbors_list.append(top_vertex)
        if top_vertex in filtered_temporal_graph:
            for neighbor, edge_count, neighbor_core_number in filtered_temporal_graph[
                top_vertex
            ]:
                if neighbor not in visited and neighbor in candidates_neighbors:
                    # edge_sum+=edge_count
                    heapq.heappush(queue, (-vertex_connect_scores[neighbor], neighbor))
                    visited.add(neighbor)

    while len(positive_neighbors) < len(hard_negative_neighbors) * 0.15:
        if len(positive_neighbors_list) > 0:
            left_vertex = positive_neighbors_list.pop(0)
        else:
            break
        if left_vertex != center_vertex:
            positive_neighbors.add(left_vertex)
    hard_negative_neighbors = (
        hard_negative_neighbors - positive_neighbors - {center_vertex}
    )
    subgraph_k_hop_cache[(center_vertex, (t_start, t_end))] = sorted(
        candidates_neighbors
    )

    return positive_neighbors, hard_negative_neighbors, candidates_neighbors


def generate_triplets_index(
    center_vertices,
    k_hop,
    t_start,
    t_end,
    num_vertex,
    temporal_graph,
    time_range_core_number,
    time_range_link_samples_cache,
    subgraph_k_hop_cache,
):
    triplets = []
    idx = 0

    filtered_subgraph = {}
    vertex_connect_scores = {}
    total_edge_weight = 0

    subgraph_nodes = set()
    for center_vertex in center_vertices:
        queue = [(center_vertex, 0)]
        visited = {center_vertex}
        while queue:
            curr_vertex, curr_hop = queue.pop(0)
            subgraph_nodes.add(curr_vertex)

            if curr_hop < k_hop and curr_vertex in temporal_graph:
                for t, neighbors in temporal_graph[curr_vertex].items():
                    if t_start <= t <= t_end:
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append((neighbor, curr_hop + 1))

    for vertex in subgraph_nodes:
        neighbor_time_edge_count = {}
        total_time_edge_count = 0

        if vertex in temporal_graph:
            for t, neighbors in temporal_graph[vertex].items():
                if t_start <= t <= t_end:
                    for neighbor in neighbors:
                        if neighbor in subgraph_nodes:
                            neighbor_time_edge_count[neighbor] = (
                                neighbor_time_edge_count.get(neighbor, 0) + 1
                            )
                            total_time_edge_count += 1

        neighbors_list = []
        for neighbor, count in neighbor_time_edge_count.items():
            core_number = time_range_core_number.get((t_start, t_end), {}).get(
                neighbor, 0
            )
            neighbors_list.append((neighbor, count, core_number))
            if vertex < neighbor:
                total_edge_weight += count

        filtered_subgraph[vertex] = neighbors_list
        vertex_core_number = time_range_core_number.get((t_start, t_end), {}).get(
            vertex, 0
        )
        vertex_connect_scores[vertex] = (
            vertex_core_number * total_time_edge_count / len(neighbors_list)
            if len(neighbors_list) != 0
            else 0
        )

    for anchor in center_vertices:
        idx = idx + 1

        positive_samples, hard_negative_samples, k_hop_samples = get_samples(
            anchor,
            k_hop,
            t_start,
            t_end,
            filtered_subgraph,
            vertex_connect_scores,
            total_edge_weight,
            time_range_core_number,
            subgraph_k_hop_cache,
        )
        if len(positive_samples) == 0:
            continue

        easy_negative_samples = random.choices(
            list(k_hop_samples - positive_samples - hard_negative_samples - {anchor}),
            k=min(
                int(len(positive_samples) * 0.8),
                len(
                    k_hop_samples - positive_samples - hard_negative_samples - {anchor}
                ),
            ),
        )
        hard_negative_samples = random.choices(
            list(hard_negative_samples),
            k=min(
                int(len(positive_samples) - len(easy_negative_samples)),
                len(hard_negative_samples),
            ),
        )
        negative_samples = hard_negative_samples + easy_negative_samples

        if len(positive_samples) == 0 or len(negative_samples) == 0:
            continue

        triplets.append((anchor, list(positive_samples), list(negative_samples)))

        link_samples = generate_time_range_link_samples(
            k_hop_samples, filtered_subgraph, num_vertex
        )
        time_range_link_samples_cache[(t_start, t_end)][anchor] = link_samples

    return triplets


class MultiSampleQuadrupletDataset:
    def __init__(self, quadruplets):
        self.quadruplets = quadruplets

    def __len__(self):
        return len(self.quadruplets)

    def __getitem__(self, idx):
        anchor, positives, negatives, time_range = self.quadruplets[idx]
        return anchor, list(positives), list(negatives), time_range


def quadruplet_collate_fn(batch):
    anchors = torch.tensor([item[0] for item in batch], dtype=torch.long)
    positives = [item[1] for item in batch]
    negatives = [item[2] for item in batch]
    time_ranges = [item[3] for item in batch]
    return anchors, positives, negatives, time_ranges


def compute_time_range_k_core(time_range_start, time_range_end):
    G = nx.Graph()
    for v in range(num_vertex):
        for t, neighbors in temporal_graph[v].items():
            if time_range_start <= t <= time_range_end:
                for neighbor in neighbors:
                    G.add_edge(v, neighbor)
    core_numbers = nx.core_number(G)
    return core_numbers


def generate_time_range_link_samples(
    k_hop_samples, filtered_temporal_graph, num_vertex
):
    link_samples_dict = defaultdict(list)
    min_neighbors = 3
    for vertex in k_hop_samples:
        if vertex in filtered_temporal_graph:

            if len(filtered_temporal_graph[vertex]) < min_neighbors:
                continue

            neighbor_edges = {}
            for neighbor, edge_count, _ in filtered_temporal_graph[vertex]:
                neighbor_edges[neighbor] = edge_count

            if not neighbor_edges:
                continue

            sorted_neighbors = sorted(neighbor_edges.items(), key=lambda x: x[1])

            if (
                len(sorted_neighbors) >= 2
                and sorted_neighbors[-1][1] > sorted_neighbors[0][1]
            ):
                neg_neighbor = sorted_neighbors[0][0]
                pos_neighbor = sorted_neighbors[-1][0]
                link_samples_dict[vertex].append((pos_neighbor, neg_neighbor))

    if len(link_samples_dict) > 100:
        selected_vertices = random.sample(list(link_samples_dict.keys()), 100)
        selected_samples = defaultdict(list)
        for vertex in selected_vertices:
            selected_samples[vertex] = link_samples_dict[vertex]
        return selected_samples
    return link_samples_dict


def margin_triplet_loss(anchor, positives, negatives, margin=1):

    D_ap = torch.sum((anchor.unsqueeze(0) - positives) ** 2, dim=1)

    D_an = torch.sum((anchor.unsqueeze(0) - negatives) ** 2, dim=1)

    D_ap_expanded = D_ap.unsqueeze(1)
    D_an_expanded = D_an.unsqueeze(0)

    losses = torch.clamp(D_ap_expanded - D_an_expanded + margin, min=0.0)

    loss = losses.mean()
    return loss


# @profile
def query_test():
    global k_core_conductance, k_core_density
    tau = 1

    print(f"the current dataset name is :{dataset_name} and the query tau is :{tau}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AdapterTemporalGNN(
        node_in_channels, node_out_channels, edge_dim=edge_dim
    ).to(device)
    model.load_state_dict(torch.load(f"./model_L1_{dataset_name}.pth"), strict=False)
    model.to(device)

    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "adapter" in name or "gating_params" in name:
            param.requires_grad = True

    optimizer = optim.Adam(
        [
            p
            for n, p in model.named_parameters()
            if ("adapter" in n or "gating_params" in n) and p.requires_grad
        ],
        lr=learning_rate,
    )
    scaler = GradScaler()

    core_index()

    test_time_range_list = []
    while len(test_time_range_list) < 10:
        t_start = random.randint(
            0, num_timestamp - 101
        )  # Ensure there's enough space for a window of at least 100
        t_end = random.randint(
            t_start + 100, min(num_timestamp - 1, t_start + 100)
        )  # Ensure t_end is at least 100 units after t_start
        if (t_start, t_end) not in test_time_range_list:
            test_time_range_list.append((t_start, t_end))

    valid_cnt = 0

    time_range_link_samples_cache = defaultdict(dict)
    epochs = 10

    for t_start, t_end in test_time_range_list:
        print(f"Test time range: [{t_start}, {t_end}]")

        result_dict = {}
        result_dict = compute_time_range_k_core(t_start, t_end)
        # core_number_dict  = result_dict
        time_range_core_number[(t_start, t_end)] = result_dict
        core_number_dict = time_range_core_number[(t_start, t_end)]
        query_vertex_list = set()

        nodes_with_core_gte_5 = [
            node for node, core in core_number_dict.items() if core >= 5
        ]

        remaining_nodes = set(nodes_with_core_gte_5) - query_vertex_list

        while len(query_vertex_list) < 10 and remaining_nodes:
            query_vertex = random.choice(list(remaining_nodes))
            query_vertex_list.add(query_vertex)
            remaining_nodes.remove(query_vertex)

        for query_vertex in query_vertex_list:
            # print(valid_cnt)
            center_vertices = set()
            center_vertices.add(query_vertex)

            neighbor_k_hop = list(
                optimize_neighbor_search(
                    temporal_graph, t_start, t_end, center_vertices
                )
            )

            result_dict = {}
            index_batch_size = 50000  # Adjust based on your GPU memory capacity
            if len(neighbor_k_hop) < 100:
                continue
            # Process vertices in batches
            for i in range(0, len(neighbor_k_hop), index_batch_size):
                batch_vertices = neighbor_k_hop[i : i + index_batch_size]
                batch_indices = torch.tensor(batch_vertices, device=device)
                with torch.no_grad():
                    batch_output = query_index(t_start, t_end, batch_indices)

                    for j, vertex in enumerate(batch_vertices):
                        result_dict[int(vertex)] = int(round(batch_output[j].item()))

                    # Optional: Free memory explicitly
                torch.cuda.empty_cache()
            time_range_core_number[(t_start, t_end)] = result_dict
            triplets = generate_triplets_index(
                center_vertices,
                k_hop,
                t_start,
                t_end,
                num_vertex,
                temporal_graph,
                time_range_core_number,
                time_range_link_samples_cache,
                subgraph_k_hop_cache,
            )

            quadruplet = [(t[0], t[1], t[2], (t_start, t_end)) for t in triplets]
            if not quadruplet:
                print("Warning: Quadruplet list is empty, skipping this iteration.")
                continue

            query_dataset = MultiSampleQuadrupletDataset(quadruplet)

            query_loader = DataLoader(
                query_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                collate_fn=quadruplet_collate_fn,
            )

            feature_dim = node_in_channels

            subgraph, vertex_map, neighbors_k_hop = extract_subgraph(
                query_vertex, t_start, t_end, k_hop, feature_dim
            )

            model.train()

            for epoch in range(epochs):
                epoch_loss = 0.0

                for batch_idx, batch in enumerate(query_loader):
                    batch_start_time = time.time()

                    anchors, positives, negatives, time_ranges = batch
                    anchors = torch.tensor(anchors, device=device)

                    positives = [torch.tensor(pos, device=device) for pos in positives]
                    negatives = [torch.tensor(neg, device=device) for neg in negatives]

                    optimizer.zero_grad()

                    subgraphs = []
                    vertex_maps = []
                    time_window_length = 0
                    query_coreness = 0
                    time_window_position = 0
                    neighbor_coreness = 0

                    for anchor, time_range in zip(anchors, time_ranges):
                        # Add subgraph creation logic here
                        subgraphs.append(subgraph)
                        vertex_maps.append(vertex_map)

                        time_window_length = t_end - t_start

                        query_coreness = time_range_core_number[
                            (time_range[0], time_range[1])
                        ].get(anchor.item(), 0)

                        global_start_time, global_end_time = 0, num_timestamp - 1
                        time_window_position = (time_range[0] - global_start_time) / (
                            global_end_time - global_start_time
                        )

                        neighbor_coreness = 0
                        if len(neighbors_k_hop) > 0:
                            neighbor_coreness = sum(
                                [
                                    time_range_core_number[
                                        (time_range[0], time_range[1])
                                    ].get(neighbor, 0)
                                    for neighbor in neighbors_k_hop
                                ]
                            ) / len(neighbors_k_hop)

                    batched_subgraphs = Batch.from_data_list(subgraphs).to(device)

                    embeddings = model(
                        batched_subgraphs,
                        query_coreness,
                        time_window_length,
                        time_window_position,
                        neighbor_coreness,
                    )
                    batch_indices = batched_subgraphs.batch
                    del batched_subgraphs

                    batch_loss = 0.0
                    for i, (
                        anchor,
                        pos_samples,
                        neg_samples,
                        vertex_map,
                        time_range,
                    ) in enumerate(
                        zip(anchors, positives, negatives, vertex_maps, time_ranges)
                    ):
                        node_indices = (batch_indices == i).nonzero(as_tuple=True)[0]
                        anchor_idx = node_indices[vertex_map[anchor.long()]]

                        pos_samples = pos_samples.to(device)
                        neg_samples = neg_samples.to(device)
                        pos_indices = node_indices[vertex_map[pos_samples.long()]]
                        neg_indices = node_indices[vertex_map[neg_samples.long()]]

                        if len(pos_indices) == 0 or len(neg_indices) == 0:
                            continue

                        with autocast():
                            loss = margin_triplet_loss(
                                embeddings[anchor_idx],
                                embeddings[pos_indices],
                                embeddings[neg_indices],
                            )
                        batch_loss += loss

                    if len(vertex_maps) > 0:
                        batch_loss = batch_loss / len(vertex_maps)

                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler.scale(batch_loss).backward()
                    else:
                        batch_loss.backward()

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    epoch_loss += batch_loss.item()
            model.eval()
            embeddings = model(subgraph)
            query_vertex_embedding = embeddings[vertex_map[query_vertex]].unsqueeze(0)
            neighbors_embeddings = embeddings[vertex_map[neighbors_k_hop]]

            distances = F.pairwise_distance(
                query_vertex_embedding, neighbors_embeddings
            )
            result = temporal_test_GNN(
                distances, vertex_map, query_vertex, t_start, t_end, tau
            )
            print(
                f"the query vertex is {query_vertex} and the result size is {len(result)}"
            )
            valid_cnt += 1
            torch.cuda.empty_cache()


def main():

    read_temporal_graph()
    get_timerange_layers()
    read_core_number()
    construct_feature_matrix()
    query_test()


if __name__ == "__main__":
    main()
