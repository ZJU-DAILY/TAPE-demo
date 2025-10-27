from collections import deque
from collections import defaultdict
import networkx as nx
import torch
import torch.nn.functional as F
import random
from Tree import *
from MLP_models import *
from index import *


def get_timerange_layers(num_timestamp, max_range, partition):
    print("Calculating time range layers...")
    time_range_set = set()
    time_range_layers = []
    max_time_range_layers = []

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
        temp_range = temp_range * partition
        layer_id = layer_id + 1

    time_range_layers.reverse()
    max_time_range_layers.reverse()

    max_layer_id = layer_id
    print(f"Number of layers: {max_layer_id}")

    return time_range_layers, max_time_range_layers, time_range_set, max_layer_id


def read_core_number(dataset_name, num_vertex, time_range_set):

    print("Loading the core number...")
    vertex_core_numbers = [{} for _ in range(num_vertex)]
    time_range_core_number = {
        time_range: {} for time_range in time_range_set
    }  # Initialize dictionary for time ranges
    core_number_filename = f"../datasets/{dataset_name}-core_number.txt"

    with open(core_number_filename, "r") as f:
        num_core_number = int(
            f.readline().strip()
        )  # Read the first line as num_core_number directly

        for line in f:
            range_part, core_numbers_part = line.split(" ", 1)
            range_start, range_end = map(int, range_part.strip("[]").split(","))
            is_node_range = (range_start, range_end) in time_range_set

            for pair in core_numbers_part.split():
                vertex, core_number = map(int, pair.split(":"))
                if range_start == range_end:
                    vertex_core_numbers[vertex][range_start] = core_number
                if is_node_range:
                    time_range_core_number[(range_start, range_end)][
                        vertex
                    ] = core_number

    return vertex_core_numbers, time_range_core_number


def construct_feature_matrix(
    num_vertex, num_timestamp, temporal_graph, vertex_core_numbers, device
):
    print("Constructing the feature matrix...")
    sequence_features1_matrix = torch.empty(0, 0, 0)
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

    return sequence_features1_matrix.to(device)


def init_vertex_features(
    t_start,
    t_end,
    vertex_set,
    feature_dim,
    anchor,
    sequence_features1_matrix,
    time_range_core_number,
    device,
):

    vertex_indices = vertex_set

    indices = sequence_features1_matrix.indices()
    values = sequence_features1_matrix.values()

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
    core_number_values = (core_number_values - core_number_values.min()) / (
        core_number_values.max() - core_number_values.min() + 1e-6
    )

    vertex_features_matrix = degree_tensor
    matrix_max = torch.max(vertex_features_matrix)
    matrix_min = torch.min(vertex_features_matrix)
    vertex_features_matrix = (vertex_features_matrix - matrix_min) / (
        matrix_max - matrix_min + 1e-6
    )
    vertex_features_matrix = torch.cat(
        [core_number_values.unsqueeze(1), vertex_features_matrix], dim=1
    )

    query_feature = torch.zeros(len(vertex_set), 1, device=device)
    if anchor != -1:
        query_feature[vertex_map[anchor]][0] = 1
    vertex_features_matrix = torch.cat([query_feature, vertex_features_matrix], dim=1)
    return vertex_features_matrix


def init_vertex_features_index(
    t_start,
    t_end,
    vertex_set,
    feature_dim,
    anchor,
    sequence_features1_matrix,
    time_range_core_number,
    max_layer_id,
    max_time_range_layers,
    device,
    partition,
    num_timestamp,
    root,
):

    vertex_indices = vertex_set

    indices = sequence_features1_matrix.indices()
    values = sequence_features1_matrix.values()

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

    # core_number_dict = model_out_put_for_any_range_vertex_set(vertex_set, t_start, t_end, max_layer_id, max_time_range_layers, device, sequence_features1_matrix, partition, num_timestamp, root)

    # Create a tensor from the dictionary values, using 0 for missing vertices
    # core_number_values = torch.tensor([core_number_dict.get(v.item(), 0) for v in vertex_set],
    #                                 dtype=torch.float32, device=device)

    core_number_values = (core_number_values - core_number_values.min()) / (
        core_number_values.max() - core_number_values.min() + 1e-6
    )

    vertex_features_matrix = degree_tensor
    matrix_max = torch.max(vertex_features_matrix)
    matrix_min = torch.min(vertex_features_matrix)
    vertex_features_matrix = (vertex_features_matrix - matrix_min) / (
        matrix_max - matrix_min + 1e-6
    )
    vertex_features_matrix = torch.cat(
        [core_number_values.unsqueeze(1), vertex_features_matrix], dim=1
    )

    query_feature = torch.zeros(len(vertex_set), 1, device=device)
    if anchor != -1:
        query_feature[vertex_map[anchor]][0] = 1
    vertex_features_matrix = torch.cat([query_feature, vertex_features_matrix], dim=1)
    return vertex_features_matrix


def get_candidate_neighbors(
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
        if len(subgraph_result) > 1000:
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


def compute_modularity(subgraph, filtered_temporal_graph, total_edge_weight):

    subgraph = set(subgraph)
    internal_weights = 0
    total_weights = 0

    for vertex in subgraph:
        if vertex in filtered_temporal_graph:
            for neighbor, edge_count, _ in filtered_temporal_graph[vertex]:

                total_weights += edge_count

                if neighbor in subgraph and vertex < neighbor:
                    internal_weights += edge_count

    if total_weights == 0:
        return 0.0

    # density modularity
    modularity = (1.0 / len(subgraph)) * (
        internal_weights - (total_weights**2) / (4 * total_edge_weight)
    )
    # classic modularity
    # modularity = (1.0 / (2 * total_edge_weight)) * (internal_weights - (total_weights ** 2) / (2 * total_edge_weight))
    return modularity


class MultiSampleQuadrupletDataset:
    def __init__(self, quadruplets):
        self.quadruplets = quadruplets

    def __len__(self):
        return len(self.quadruplets)

    def __getitem__(self, idx):
        anchor, positives, negatives, time_range = self.quadruplets[idx]
        return anchor, list(positives), list(negatives), time_range


def optimize_neighbor_search(temporal_graph, t_start, t_end, center_vertices):
    neighbor_k_hop = set()
    visited = set()
    for center_vertex in center_vertices:
        queue = [(center_vertex, 0)]
        visited = {center_vertex}
        while queue:
            curr_vertex, curr_hop = queue.pop(0)
            neighbor_k_hop.add(curr_vertex)

            if curr_hop < 5 and curr_vertex in temporal_graph:
                for t, neighbors in temporal_graph[curr_vertex].items():
                    if t_start <= t <= t_end:
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append((neighbor, curr_hop + 1))

    return neighbor_k_hop
