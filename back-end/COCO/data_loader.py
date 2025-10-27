import torch
from collections import defaultdict
from pympler import asizeof
from torch_geometric.data import Data


def read_temporal_graph(dataset_name, device):
    print("Loading the graph...")
    # Maps each timestamp to a set of unique edges active at that time.
    time_edge = defaultdict(set)
    # Adjacency list representation: source_node -> timestamp -> [neighbors].
    temporal_graph = defaultdict(lambda: defaultdict(list))
    num_vertex = 0
    num_edge = 0
    num_timestamp = 0

    filename = f"../datasets/{dataset_name}.txt"
    with open(filename, "r") as f:
        # The first line of the file is expected to contain metadata.
        is_first_line = True
        for line in f:
            if is_first_line:
                num_vertex, num_edge, num_timestamp = map(int, line.strip().split())
                is_first_line = False
                continue

            v1, v2, t = map(int, line.strip().split())
            # Ignore self-loops.
            if v1 == v2:
                continue
            # Ensure consistent edge representation (smaller_node, larger_node)
            # to handle the graph as undirected.
            if v1 > v2:
                v1, v2 = v2, v1

            time_edge[t].add((v1, v2))

            # Build the adjacency list for the undirected graph.
            temporal_graph[v1][t].append(v2)
            temporal_graph[v2][t].append(v1)

    # Calculate and report the memory usage of the dictionary-based graph.
    total_size = asizeof.asizeof(temporal_graph)
    print(f"Memory usage of temporal_graph: {total_size / (1024 ** 2):.2f} MB")

    # --- Convert to PyTorch Geometric Data format ---

    # Invert the mapping to group all timestamps for each unique static edge.
    edge_to_timestamps = {}
    for t, edges in time_edge.items():
        for src, dst in edges:
            edge_to_timestamps.setdefault((src, dst), set()).add(t)
            edge_to_timestamps.setdefault((dst, src), set()).add(t)

    # --- Construct edge_index and sparse edge_attr ---

    # A list of unique static edges, forming the edge_index.
    edge_index_list = []
    # Coordinate format (COO) indices for the sparse tensor of timestamps.
    edge_attr_indices = []  # Shape: [2, num_non_zero_timestamps]
    # Values (the actual timestamps) for the sparse tensor.
    edge_attr_values = []

    # Iterate through each unique edge and its associated timestamps.
    for (src, dst), timestamps in edge_to_timestamps.items():
        # The index of the current edge corresponds to its row in the sparse matrix.
        current_edge_idx = len(edge_index_list)
        edge_index_list.append([src, dst])

        # Create an entry in the sparse tensor for each timestamp.
        for t_idx, t in enumerate(sorted(list(timestamps))):
            # [row, col] = [edge's index, timestamp's positional index]
            edge_attr_indices.append([current_edge_idx, t_idx])
            # The value is the actual timestamp.
            edge_attr_values.append(float(t))

    # --- Convert lists to PyTorch tensors ---

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr_indices = (
        torch.tensor(edge_attr_indices, dtype=torch.long).t().contiguous()
    )
    edge_attr_values = torch.tensor(edge_attr_values, dtype=torch.float32)

    # The second dimension of the sparse tensor size is the maximum number of
    # timestamps associated with any single edge.
    max_timestamps_per_edge = (
        max(len(ts) for ts in edge_to_timestamps.values()) if edge_to_timestamps else 0
    )

    # Create a sparse tensor to hold the timestamps for each edge.
    # Shape: [num_edges, max_timestamps_per_edge]
    edge_attr = torch.sparse_coo_tensor(
        indices=edge_attr_indices,
        values=edge_attr_values,
        size=(len(edge_index_list), max_timestamps_per_edge),
        device=device,
    ).coalesce()  # Sums duplicate indices, though none should exist here.

    # --- Create the final PyG Data object ---

    temporal_graph_pyg = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,  # Sparse tensor containing actual timestamps.
        num_nodes=num_vertex,
    )
    temporal_graph_pyg = temporal_graph_pyg.to(device)

    return num_vertex, num_timestamp, temporal_graph, temporal_graph_pyg
