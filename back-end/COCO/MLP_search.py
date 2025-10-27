import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
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
import multiprocessing
from multiprocessing import Pool, Manager
import time
import cProfile
import tracemalloc


from MLP_models import (
    MLP,
    MLPNonleaf,
    preprocess_sequences,
    SequenceDataset,
    SequenceDatasetNonleaf,
)
from Tree import TreeNode


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

None  # print(f"Using device: {device}")


# dataset_name = 'mathoverflow'
# dataset_name = 'wikitalk'
dataset_name = "collegemsg"
# dataset_name = 'superuser'
# dataset_name = 'youtube'
# dataset_name = 'dblp'
# dataset_name = 'flickr'

num_vertex = 0
num_edge = 0
num_timestamp = 0
time_edge = {}
num_core_number = 0
vertex_core_numbers = []
time_range_core_number = defaultdict(dict)
temporal_graph = []
temporal_graph_test = []
time_range_layers_MLP_MLP = []
time_range_set = set()
max_time_range_layers_MLP_MLP = []
min_time_range_layers_MLP_MLP = []
sequence_features1_matrix = torch.empty(0, 0, 0)
indices_vertex_of_matrix = torch.empty(0, 0)
model_time_range_layers_MLP_MLP = []
partition = 4
max_range = 45
max_degree = 0
max_layer_id = 0
root = None

inter_time = 0.0


def read_temporal_graph_MLP():
    None  # print("Loading the graph...")
    filename = f"../datasets/{dataset_name}.txt"
    global num_vertex, num_edge, num_timestamp, time_edge, temporal_graph, max_degree
    time_edge = defaultdict(list)
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
            time_edge[t].append((v1, v2))

            temporal_graph[v1][t].append(v2)
            temporal_graph[v2][t].append(v1)
    # temporal_graph_np = defaultdict(lambda: defaultdict(object))
    # for vertex in temporal_graph:
    #     for t in temporal_graph[vertex]:
    #         if len(temporal_graph[vertex][t]) > 0:
    #             temporal_graph_np[vertex][t] = np.array(temporal_graph[vertex][t], dtype=np.int32)

    total_size = asizeof.asizeof(temporal_graph)
    None


def read_core_number_MLP():
    None  # print("Loading the core number...")
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


def get_timerange_layers_MLP():
    global time_range_layers_MLP_MLP, min_time_range_layers_MLP_MLP, max_time_range_layers_MLP_MLP
    temp_range = num_timestamp
    while temp_range > max_range:
        temp_range = temp_range // partition
    layer_id = 0
    while temp_range < num_timestamp:
        range_start = 0
        time_range_layers_MLP_MLP.append([])
        temp_max_range = 0
        temp_min_range = num_timestamp
        while range_start < num_timestamp:
            range_end = range_start + temp_range - 1
            range_end = min(range_end, num_timestamp - 1)
            if range_end + temp_range > num_timestamp - 1:
                range_end = num_timestamp - 1
            time_range_layers_MLP_MLP[layer_id].append((range_start, range_end))
            time_range_set.add((range_start, range_end))
            if range_end - range_start + 1 > temp_max_range:
                temp_max_range = range_end - range_start + 1
            if range_end - range_start + 1 < temp_min_range:
                temp_min_range = range_end - range_start + 1
            range_start = range_end + 1
        max_time_range_layers_MLP_MLP.append(temp_max_range)
        min_time_range_layers_MLP_MLP.append(temp_min_range)
        temp_range = temp_range * partition
        layer_id = layer_id + 1
    time_range_layers_MLP_MLP.reverse()
    max_time_range_layers_MLP_MLP.reverse()
    min_time_range_layers_MLP_MLP.reverse()


def build_tree_MLP():
    node_stack = []
    root_node = TreeNode((0, num_timestamp - 1), 0)
    global max_layer_id
    None  # print(f'max_layer_id: {max_layer_id}')
    None  # print(f'time_range_layers_MLP_MLP len:{len(time_range_layers_MLP_MLP)}')
    node_stack.append(root_node)
    while len(node_stack) > 0:
        current_node = node_stack.pop()
        current_node.vertex_core_number = time_range_core_number[
            (current_node.time_start, current_node.time_end)
        ]
        current_node.vertex_degree = defaultdict(int)
        if current_node.layer_id != 0:
            for v in range(num_vertex):
                neighbors_set = set()
                for t, neighbors in temporal_graph[v].items():
                    if current_node.time_start <= t <= current_node.time_end:
                        neighbors_set.update(neighbors)
                if len(neighbors_set) > 0:
                    current_node.vertex_degree[v] = len(neighbors_set)

        if current_node.layer_id < len(time_range_layers_MLP_MLP) - 1:
            for i in range(len(time_range_layers_MLP_MLP[current_node.layer_id + 1])):
                temp_time_start = time_range_layers_MLP_MLP[current_node.layer_id + 1][
                    i
                ][0]
                temp_time_end = time_range_layers_MLP_MLP[current_node.layer_id + 1][i][
                    1
                ]
                if (
                    temp_time_start >= current_node.time_start
                    and temp_time_end <= current_node.time_end
                ):
                    child_node = TreeNode(
                        (temp_time_start, temp_time_end), current_node.layer_id + 1
                    )
                    current_node.add_child(child_node)
                    node_stack.append(child_node)
    max_layer_id = len(time_range_layers_MLP_MLP) - 1
    None  # print(f'max_layer_id: {max_layer_id}')
    return root_node


def tree_query_MLP(time_start, time_end):
    if time_start < 0 or time_end >= num_timestamp or time_start > time_end:
        return None

    node = root
    while node.layer_id < max_layer_id:
        move_to_next = False
        for child in node.children:
            if child.time_start <= time_start and child.time_end >= time_end:
                node = child
                move_to_next = True
                break
        if not move_to_next:
            break
    return node


def get_node_path_MLP(time_start, time_end):
    node = root
    if time_start < 0 or time_end >= num_timestamp or time_start > time_end:
        return None
    path = [node]
    while node.layer_id < max_layer_id:
        move_to_next = False
        for child in node.children:
            if child.time_start <= time_start and child.time_end >= time_end:
                node = child
                path.append(node)
                move_to_next = True
                break
        if not move_to_next:
            break
    return path


def model_output_for_path_MLP(time_start, time_end, vertex_set, sequence_features):
    if time_start < 0 or time_end >= num_timestamp or time_start > time_end:
        return torch.zeros(len(vertex_set), 1, device=device)
    sequence_features = sequence_features.to(device)
    node = root
    path = [node]
    while node.layer_id < max_layer_id:
        move_to_next = False
        for child in node.children:
            if child.time_start <= time_start and child.time_end >= time_end:
                node = child
                path.append(node)
                move_to_next = True
                break
        if not move_to_next:
            break
    if len(path) == 1:
        return torch.zeros(len(vertex_set), 1, device=device)

    path.pop()
    output = torch.zeros(len(vertex_set), 1, dtype=torch.float32, device=device)
    # output = output.to(device)
    sequence_input1 = torch.zeros(
        len(vertex_set), max_time_range_layers_MLP_MLP[0], 2, device=device
    )
    sequence_input1[:, 0 : sequence_features.shape[1], :] = sequence_features
    for node in path:
        max_length1 = max_time_range_layers_MLP_MLP[node.layer_id + 1] * 2
        max_length2 = partition

        sequence_input1 = sequence_input1[:, :max_length1, :]
        sequence_input2 = torch.zeros(len(vertex_set), max_length2, 2, device=device)

        single_value = output

        model = node.model

        with torch.no_grad():
            output = model(sequence_input1, sequence_input2, single_value)
            if output.dim() == 0:
                output = output.reshape(len(vertex_set), 1)
    return output


def load_models_MLP(depth_id=0):

    global model_time_range_layers_MLP_MLP
    model_time_range_layers_MLP_MLP = [
        [] for _ in range(len(time_range_layers_MLP_MLP))
    ]
    None  # print("Loading models...")
    # load models in a tree
    for layer_id in range(0, depth_id + 1):
        for i in range(len(time_range_layers_MLP_MLP[layer_id])):
            None  # print(f"{i+1}/{len(time_range_layers_MLP_MLP[layer_id])}")
            time_start = time_range_layers_MLP_MLP[layer_id][i][0]
            time_end = time_range_layers_MLP_MLP[layer_id][i][1]
            model = None
            if layer_id != max_layer_id:
                model = MLPNonleaf(
                    2, max_time_range_layers_MLP_MLP[layer_id + 1] * 2, partition, 64
                ).to(device)
                model.load_state_dict(
                    torch.load(f"models/{dataset_name}/model_{layer_id}_{i}.pth")
                )
                model.eval()
            else:
                model = MLP(2, max_time_range_layers_MLP_MLP[max_layer_id], 64).to(
                    device
                )
                model.load_state_dict(
                    torch.load(f"models/{dataset_name}/model_{layer_id}_{i}.pth")
                )
                model.eval()
            node = tree_query_MLP(time_start, time_end)
            if node is None:
                None  # print("Error: node not found.")
            else:
                node.set_model(model)


# @profile
def model_out_put_for_any_range_vertex_set_MLP(vertex_set, time_start, time_end):

    global inter_time
    node = tree_query_MLP(time_start, time_end)
    if node is None:
        None  # print("Error: node not found.")
        return torch.zeros(len(vertex_set))
    model = node.model
    # model.eval()
    if node.layer_id == max_layer_id:
        max_length = max_time_range_layers_MLP_MLP[node.layer_id]

        vertex_indices = torch.tensor(vertex_set, device=device).sort().values

        indices = sequence_features1_matrix.indices()  # (n, nnz)
        values = sequence_features1_matrix.values()  # (nnz,)

        start_idx = torch.searchsorted(indices[0], vertex_indices, side="left")
        end_idx = torch.searchsorted(indices[0], vertex_indices, side="right")

        # mask_indices = torch.cat([torch.arange(start, end, device=device) for start, end in zip(start_idx, end_idx)])

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

        time_mask = (filtered_indices[1] >= time_start) & (
            filtered_indices[1] <= time_end
        )
        final_indices = filtered_indices[:, time_mask]
        final_values = filtered_values[time_mask]

        vertex_map = torch.zeros(
            vertex_indices.max() + 1, dtype=torch.long, device=device
        )
        vertex_map[vertex_indices] = torch.arange(len(vertex_indices), device=device)

        final_indices[0] = vertex_map[final_indices[0]]

        final_indices[1] -= time_start

        result_size = (
            len(vertex_indices),
            max_length,
            sequence_features1_matrix.size(2),
        )
        result_sparse_tensor = torch.sparse_coo_tensor(
            final_indices, final_values, size=result_size, device=device
        )

        sequence_features = result_sparse_tensor.to_dense()

        single_value = model_output_for_path_MLP(
            time_start, time_end, vertex_set, sequence_features
        )

        sequence_features = sequence_features.to(device)
        single_value = single_value.to(device)
        with torch.no_grad():
            output = model(sequence_features, single_value)

        return output
    else:

        covered_nodes = []
        sequence_features2 = torch.zeros(len(vertex_set), partition, 2, device=device)

        for child_node in node.children:
            if child_node.time_start >= time_start and child_node.time_end <= time_end:
                covered_nodes.append(child_node)

        for idx, v in enumerate(vertex_set):
            for idx2, temp_node in enumerate(covered_nodes):
                core_number = temp_node.vertex_core_number.get(v, 0)
                num_neighbor = temp_node.vertex_degree[v]
                if idx == 4:
                    idx -= 1
                if idx2 == 4:
                    idx2 -= 1

                sequence_features2[idx, idx2, 0] = core_number
                sequence_features2[idx, idx2, 1] = num_neighbor

        max_length = max_time_range_layers_MLP_MLP[node.layer_id + 1] * 2

        vertex_indices = torch.tensor(vertex_set, device=device)
        vertex_indices = torch.sort(vertex_indices).values

        indices = sequence_features1_matrix.indices()  # (n, nnz)
        values = sequence_features1_matrix.values()  # (nnz,)

        start_idx = torch.searchsorted(indices[0], vertex_indices, side="left")
        end_idx = torch.searchsorted(indices[0], vertex_indices, side="right")

        # mask_indices = torch.cat([torch.arange(start, end, device=device) for start, end in zip(start_idx, end_idx)])

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

        vertex_map = torch.zeros(
            vertex_indices.max() + 1, dtype=torch.long, device=device
        )
        vertex_map[vertex_indices] = torch.arange(len(vertex_indices), device=device)

        if len(covered_nodes) == 0:

            time_mask = (filtered_indices[1] >= time_start) & (
                filtered_indices[1] <= time_end
            )
            final_indices = filtered_indices[:, time_mask]
            final_values = filtered_values[time_mask]

            final_indices[0] = vertex_map[final_indices[0]]

            final_indices[1] -= time_start

            result_size = (
                len(vertex_indices),
                max_length,
                sequence_features1_matrix.size(2),
            )
            result_sparse_tensor = torch.sparse_coo_tensor(
                final_indices, final_values, size=result_size, device=device
            )

            sequence_features1 = result_sparse_tensor.to_dense()

            single_value = model_output_for_path_MLP(
                time_start, time_end, vertex_set, sequence_features1
            )
        else:

            time_mask = (filtered_indices[1] >= time_start) & (
                filtered_indices[1] <= time_end
            ) & (filtered_indices[1]) < covered_nodes[0].time_start & (
                filtered_indices[1] > covered_nodes[-1].time_end
            )
            final_indices = filtered_indices[:, time_mask]
            final_values = filtered_values[time_mask]
            for i in range(len(final_indices[0])):
                rank = vertex_map.get(final_indices[0][i].item(), 0)
                final_indices[0][i] = rank
            final_indices[1] -= time_start
            result_size = (
                len(vertex_indices),
                max_length,
                sequence_features1_matrix.size(2),
            )
            result_sparse_tensor = torch.sparse_coo_tensor(
                final_indices, final_values, size=result_size, device=device
            )
            sequence_features1 = result_sparse_tensor.to_dense()

            time_mask2 = (filtered_indices[1] >= time_start) & (
                filtered_indices[1] <= time_end
            )
            final_indices2 = filtered_indices[:, time_mask2]
            final_values2 = filtered_values[time_mask2]

            final_indices2[0] = vertex_map[final_indices2[0]]

            final_indices2[1] -= time_start
            result_size2 = (
                len(vertex_indices),
                time_end - time_start + 1,
                sequence_features1_matrix.size(2),
            )
            result_sparse_tensor2 = torch.sparse_coo_tensor(
                final_indices2, final_values2, size=result_size2, device=device
            )
            sequence_features1_extra = result_sparse_tensor2.to_dense()

            single_value = model_output_for_path_MLP(
                time_start, time_end, vertex_set, sequence_features1_extra
            )
            # single_value = torch.zeros(len(vertex_set), 1, device=device)

        sequence_features1 = sequence_features1.to(device)
        sequence_features2 = sequence_features2.to(device)
        single_value = single_value.to(device)
        with torch.no_grad():
            output = model(sequence_features1, sequence_features2, single_value)
        return output


def construct_feature_matrix_MLP():
    global sequence_features1_matrix, indices_vertex_of_matrix

    indices_vertex_of_matrix = torch.zeros(
        num_vertex, 2, dtype=torch.int64, device=device
    )
    indices = []
    values = []

    idx = 0
    for v in range(num_vertex):
        start_idx = idx
        for t, neighbors in temporal_graph[v].items():
            core_number = vertex_core_numbers[v].get(t, 0)
            neighbor_count = len(neighbors)
            if core_number > 0:

                indices.append([v, t, 0])
                values.append(core_number)
                idx = idx + 1
            if neighbor_count > 0:
                indices.append([v, t, 1])
                values.append(neighbor_count)
                idx = idx + 1
        if start_idx != idx:
            indices_vertex_of_matrix[v][0] = start_idx
            indices_vertex_of_matrix[v][1] = idx - 1
        else:
            indices_vertex_of_matrix[v][0] = -1
            indices_vertex_of_matrix[v][1] = -1

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
    None


# @profile
def query_test():
    global temporal_graph_test, inter_time
    None  # print("Query test...")
    query_num = 2000
    total_query_time = 0.0
    total_prediction_time = 0.0
    valid_query_num = 0
    total_result_num = 0.0
    for i in range(query_num):
        query_time_range_start = random.randint(0, num_timestamp - 1)
        query_time_range_end = random.randint(
            query_time_range_start, min(num_timestamp - 1, query_time_range_start + 100)
        )
        query_vertex = random.randint(0, num_vertex - 1)
        query_result = []

        query_time_range_start = 10
        query_time_range_end = 100

        query_vertex_core_number = model_out_put_for_any_range_vertex_set_MLP(
            [query_vertex], query_time_range_start, query_time_range_end
        )

        if query_vertex_core_number.item() < 1:
            None  # print("The query vertex is not a core vertex.")

            continue
        valid_query_num = valid_query_num + 1
        None  # print(f"Query Vertex: {query_vertex}, Core Number: {query_vertex_core_number}")
        None  # print(f"Query {i + 1}: {query_time_range_start} - {query_time_range_end}")

        # precompute
        vertex_visited = np.zeros(num_vertex, dtype=bool)
        vertex_queue = deque()
        # vertex_queue.append(query_vertex)
        vertex_visited[query_vertex] = True
        candidate_vertices = [query_vertex]

        start_time = time.time()
        current_node = tree_query_MLP(query_time_range_start, query_time_range_end)
        vertex_queue.append((query_vertex, 0))
        hop_num = 4

        while len(vertex_queue) > 0:
            (temp_vertex, temp_hop) = vertex_queue.popleft()
            temp_vertex = int(temp_vertex)
            if temp_hop > hop_num:
                continue
            vertex_visited[temp_vertex] = True

            candidate_vertices.append(temp_vertex)

            # neighbor_set = set()
            for t, neighbors in temporal_graph[temp_vertex].items():
                if query_time_range_start <= t <= query_time_range_end:
                    # neighbor_set.update(neighbors)
                    for neighbor in neighbors:
                        neighbor_core_number = current_node.vertex_degree.get(
                            neighbor, 0
                        )
                        if (
                            not vertex_visited[neighbor]
                            and neighbor_core_number >= query_vertex_core_number
                        ):
                            vertex_queue.append((neighbor, temp_hop + 1))
                            vertex_visited[neighbor] = True

        end_time = time.time()
        None  # print(len(candidate_vertices))
        # None #print(candidate_vertices_core_numbers)
        total_result_num += len(candidate_vertices)
        total_query_time += end_time - start_time

    None  # print("Query test finished.")
    if valid_query_num != 0:
        average_query_time = total_query_time / valid_query_num
        average_inter_time = inter_time / valid_query_num
        None  # print(f"the vaild num:{valid_query_num}")
        None  # print(f"Average query time: {total_query_time / valid_query_num:.4f} s")
        None  # print(f"Average Inter time: {inter_time / valid_query_num:.4f} s")
        None  # print(average_inter_time / average_query_time)
        None  # print(f"Average result number: {total_result_num / valid_query_num:.4f}")
    else:
        None  # print(f"Total query time: {total_query_time} s")
    # None #print(f"Average prediction time: {total_prediction_time / valid_query_num:.4f} s")


def core_index():
    # # trace memory usage
    # tracemalloc.start()

    global root
    read_temporal_graph_MLP()
    get_timerange_layers_MLP()
    read_core_number_MLP()
    root = build_tree_MLP()
    None  # print(f'max_layer_id: {max_layer_id}')
    load_models_MLP(max_layer_id)
    construct_feature_matrix_MLP()

    # query_test()


# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     core_index()


def query_index(query_time_range_start, query_time_range_end, query_vertex):
    global temporal_graph_test, inter_time

    query_vertex_list = query_vertex.tolist()  # Convert tensor to list of vertex IDs

    query_vertex_core_number = model_out_put_for_any_range_vertex_set_MLP(
        query_vertex_list, query_time_range_start, query_time_range_end
    )

    return query_vertex_core_number
