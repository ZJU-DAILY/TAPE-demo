import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from collections import defaultdict, deque
import random
import heapq
import multiprocessing

from utils import *
from index import *


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
    """Generates positive and hard-negative samples for a given center vertex.

    This function first identifies a set of candidate neighbors within a k-hop
    radius. It then performs a priority queue-based search to select positive
    and hard-negative samples from these candidates.

    Args:
      center_vertex: The anchor node for which to generate samples.
      k: The number of hops to define the neighborhood.
      t_start: The start of the time range for the graph snapshot.
      t_end: The end of the time range for the graph snapshot.
      filtered_temporal_graph: A pre-filtered subgraph for the given time range.
      vertex_connect_scores: Precomputed connectivity scores for each vertex.
      total_edge_weight: The total weight of edges in the filtered graph.
      time_range_core_number: A dictionary mapping time ranges to core numbers of
        vertices.
      subgraph_k_hop_cache: A cache for storing k-hop neighborhoods to avoid
        recomputation.

    Returns:
      A tuple containing:
        - positive_neighbors: A set of nodes selected as positive samples.
        - hard_negative_neighbors: A set of nodes selected as hard-negative
          samples.
        - candidates_neighbors: A set of all nodes within the k-hop
          neighborhood.
    """
    candidates_neighbors = get_candidate_neighbors(
        center_vertex,
        k,
        t_start,
        t_end,
        filtered_temporal_graph,
        total_edge_weight,
        time_range_core_number,
        subgraph_k_hop_cache,
    )

    positive_neighbors_list = []
    positive_neighbors = set()
    hard_negative_neighbors = set()
    visited = {center_vertex}

    # Use a priority queue to explore neighbors, prioritizing by connection score.
    # We use negative scores because heapq is a min-heap.
    priority_queue = [(-vertex_connect_scores.get(center_vertex, 0), center_vertex)]
    query_vertex_core_number = time_range_core_number.get((t_start, t_end), {}).get(
        center_vertex, 0
    )

    # Explore the graph starting from the center vertex.
    while priority_queue:
        # Limit the exploration to a multiple of the candidate set size for efficiency.
        if sum(1 for _ in priority_queue) > len(candidates_neighbors) * 3:
            break

        _, top_vertex = heapq.heappop(priority_queue)
        v_core_number = time_range_core_number.get((t_start, t_end), {}).get(
            top_vertex, 0
        )

        # A hard negative is a node with a core number greater than or equal to the query vertex's.
        if v_core_number >= query_vertex_core_number:
            hard_negative_neighbors.add(top_vertex)

        positive_neighbors_list.append(top_vertex)

        if top_vertex in filtered_temporal_graph:
            for neighbor, _, _ in filtered_temporal_graph[top_vertex]:
                if neighbor not in visited and neighbor in candidates_neighbors:
                    heapq.heappush(
                        priority_queue,
                        (-vertex_connect_scores.get(neighbor, 0), neighbor),
                    )
                    visited.add(neighbor)

    # Balance the number of positive samples based on the number of hard negatives.
    while len(positive_neighbors) < len(hard_negative_neighbors) * 0.3:
        if not positive_neighbors_list:
            break
        left_vertex = positive_neighbors_list.pop(0)
        if left_vertex != center_vertex:
            positive_neighbors.add(left_vertex)

    # Ensure no overlap between sample sets.
    hard_negative_neighbors -= positive_neighbors
    hard_negative_neighbors.discard(center_vertex)

    # Cache the k-hop neighborhood for future use.
    subgraph_k_hop_cache[(center_vertex, (t_start, t_end))] = sorted(
        list(candidates_neighbors)
    )

    return positive_neighbors, hard_negative_neighbors, candidates_neighbors


def generate_time_range_link_samples(
    k_hop_samples, filtered_temporal_graph, num_vertex
):
    """Generates positive and negative link samples within a k-hop neighborhood.

    For each vertex in the provided sample set, this function identifies a
    positive link (to the neighbor with the highest edge count) and a negative
    link (to the neighbor with the lowest edge count).

    Args:
      k_hop_samples: A set of vertices forming the neighborhood.
      filtered_temporal_graph: The graph structure for the specific time range.
      num_vertex: The total number of vertices in the graph.

    Returns:
      A dictionary mapping a vertex to a list of (positive_neighbor,
      negative_neighbor) tuples. Returns a random subsample if the result is
      too large.
    """
    link_samples_dict = defaultdict(list)
    min_neighbors = 3

    for vertex in k_hop_samples:
        if (
            vertex in filtered_temporal_graph
            and len(filtered_temporal_graph[vertex]) >= min_neighbors
        ):
            neighbor_edges = {
                neighbor: edge_count
                for neighbor, edge_count, _ in filtered_temporal_graph[vertex]
            }
            if not neighbor_edges:
                continue

            # Sort neighbors by edge count to find the weakest and strongest links.
            sorted_neighbors = sorted(neighbor_edges.items(), key=lambda item: item[1])

            # Ensure there is a variation in edge weights to create meaningful samples.
            if (
                len(sorted_neighbors) >= 2
                and sorted_neighbors[-1][1] > sorted_neighbors[0][1]
            ):
                neg_neighbor = sorted_neighbors[0][
                    0
                ]  # The neighbor with the fewest interactions.
                pos_neighbor = sorted_neighbors[-1][
                    0
                ]  # The neighbor with the most interactions.
                link_samples_dict[vertex].append((pos_neighbor, neg_neighbor))

    # If we generate a very large number of samples, take a random subset.
    if len(link_samples_dict) > 100:
        selected_vertices = random.sample(list(link_samples_dict.keys()), 100)
        return {vertex: link_samples_dict[vertex] for vertex in selected_vertices}
    return link_samples_dict


def generate_triplets(
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
    """Generates training triplets (anchor, positive, negative) for a time range.

    This function orchestrates the sample generation process. It first filters
    the global temporal graph to a specific time window. Then, for each anchor
    vertex, it generates positive, hard-negative, and easy-negative samples to
    form triplets for contrastive learning.

    Args:
      center_vertices: A list of anchor vertices to generate triplets for.
      k_hop: The number of hops to define the neighborhood.
      t_start: The start of the time range.
      t_end: The end of the time range.
      num_vertex: The total number of vertices in the graph.
      temporal_graph: The complete graph with temporal information.
      time_range_core_number: Precomputed core numbers for vertices in different
        time ranges.
      time_range_link_samples_cache: A cache for storing generated link samples.
      subgraph_k_hop_cache: A cache for k-hop neighborhoods.

    Returns:
      A list of triplets, where each triplet is a tuple: (anchor_vertex,
      list_of_positives, list_of_negatives).
    """
    triplets = []

    # Step 1: Create a subgraph filtered by the specified time range [t_start, t_end].
    filtered_subgraph = {}
    vertex_connect_scores = {}
    total_edge_weight = 0
    for vertex in range(num_vertex):
        neighbor_time_edge_count = defaultdict(int)
        total_time_edge_count = 0
        if vertex in temporal_graph:
            for t, neighbors in temporal_graph[vertex].items():
                if t_start <= t <= t_end:
                    for neighbor in neighbors:
                        neighbor_time_edge_count[neighbor] += 1
                        total_time_edge_count += 1

        neighbors_list = []
        for neighbor, count in neighbor_time_edge_count.items():
            core_number = time_range_core_number.get((t_start, t_end), {}).get(
                neighbor, 0
            )
            neighbors_list.append((neighbor, count, core_number))
            if vertex < neighbor:  # Avoid double counting edges.
                total_edge_weight += count

        filtered_subgraph[vertex] = neighbors_list
        vertex_core_number = time_range_core_number.get((t_start, t_end), {}).get(
            vertex, 0
        )

        # Calculate a connectivity score for each vertex.
        if len(neighbors_list) > 0:
            vertex_connect_scores[vertex] = (
                vertex_core_number * total_time_edge_count / len(neighbors_list)
            )
        else:
            vertex_connect_scores[vertex] = 0

    # Step 2: Generate samples for each anchor vertex.
    for anchor in center_vertices:
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

        if not positive_samples:
            continue

        # Create a pool of potential easy negatives.
        easy_negative_pool = (
            k_hop_samples - positive_samples - hard_negative_samples - {anchor}
        )

        # Sample easy negatives, ensuring not to exceed the pool size.
        num_easy_negatives = min(
            int(len(positive_samples) * 0.8), len(easy_negative_pool)
        )
        easy_negative_samples = random.sample(
            list(easy_negative_pool), k=num_easy_negatives
        )

        # Sample hard negatives, ensuring not to exceed the pool size.
        num_hard_negatives = min(
            len(positive_samples) - len(easy_negative_samples),
            len(hard_negative_samples),
        )
        hard_negative_samples = random.sample(
            list(hard_negative_samples), k=num_hard_negatives
        )

        negative_samples = hard_negative_samples + easy_negative_samples

        if not positive_samples or not negative_samples:
            continue

        triplets.append((anchor, list(positive_samples), list(negative_samples)))

        # Step 3: Generate and cache link prediction samples for the neighborhood.
        link_samples = generate_time_range_link_samples(
            k_hop_samples, filtered_subgraph, num_vertex
        )
        time_range_link_samples_cache[(t_start, t_end)][anchor] = link_samples

    return triplets


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


def extract_subgraph_for_time_range(
    anchors,
    t_start,
    t_end,
    feature_dim,
    temporal_graph_pyg,
    num_vertex,
    edge_dim,
    device,
    vertex_features,
    time_range_core_number,
    subgraph_k_hop_cache,
):

    neighbors = set()

    for anchor in anchors:
        neighbors |= set(subgraph_k_hop_cache[(anchor, (t_start, t_end))])
    neighbors = torch.tensor(sorted(neighbors), device=device)

    vertex_map = torch.full((num_vertex,), -1, dtype=torch.long, device=device)
    vertex_map[neighbors] = torch.arange(len(neighbors), device=device)

    sub_edge_index, _, edge_mask = subgraph(
        subset=neighbors,
        edge_index=temporal_graph_pyg.edge_index,
        return_edge_mask=True,
        relabel_nodes=False,
    )

    edge_attr = temporal_graph_pyg.edge_attr
    edge_indices = edge_attr.indices()
    edge_values = edge_attr.values()

    time_mask = (edge_values >= t_start) & (edge_values <= t_end)
    valid_time_indices = edge_indices[:, time_mask]
    valid_time_values = edge_values[time_mask]

    edge_in_subgraph_mask = edge_mask[valid_time_indices[0]]
    valid_time_indices = valid_time_indices[:, edge_in_subgraph_mask]
    valid_time_values = valid_time_values[edge_in_subgraph_mask]

    unique_edges = torch.unique(valid_time_indices[0])
    edge_id_map = torch.full((edge_mask.size(0),), -1, dtype=torch.long, device=device)
    edge_id_map[unique_edges] = torch.arange(len(unique_edges), device=device)
    sub_edge_index = temporal_graph_pyg.edge_index[:, unique_edges]

    batch_size = 5000
    num_batches = (len(unique_edges) + batch_size - 1) // batch_size
    sub_edge_attr_list = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(unique_edges))

        batch_dense_edge_attr = torch.zeros(
            end_idx - start_idx, t_end - t_start + 1, device=device
        )

        batch_mask = (valid_time_indices[0] >= start_idx) & (
            valid_time_indices[0] < end_idx
        )
        batch_indices = valid_time_indices[:, batch_mask]
        batch_values = valid_time_values[batch_mask]

        batch_indices[0] = batch_indices[0] - start_idx
        batch_times = (batch_values - t_start).long()

        batch_dense_edge_attr[batch_indices[0], batch_times] = 1

        target_dim = edge_dim * 2 - 2
        if batch_dense_edge_attr.shape[1] < target_dim:
            batch_dense_edge_attr = F.interpolate(
                batch_dense_edge_attr.unsqueeze(1),
                size=target_dim,
                mode="linear",
                align_corners=True,
            )
        else:
            batch_dense_edge_attr = F.adaptive_avg_pool1d(
                batch_dense_edge_attr.unsqueeze(1), target_dim
            )
        batch_dense_edge_attr = batch_dense_edge_attr.squeeze(1)
        batch_fft = torch.abs(torch.fft.fft(batch_dense_edge_attr, dim=1))[:, :edge_dim]

        sub_edge_attr_list.append(batch_fft)

        del batch_dense_edge_attr
        torch.cuda.empty_cache()

    sub_edge_attr = torch.cat(sub_edge_attr_list, dim=0)

    subgraph_pyg = Data(
        x=init_vertex_features(
            t_start,
            t_end,
            neighbors,
            feature_dim,
            -1,
            vertex_features,
            time_range_core_number,
            device,
        ),
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        device=device,
    )

    return subgraph_pyg, vertex_map


def extract_subgraph_for_anchor(
    anchor,
    t_start,
    t_end,
    subgraph_pyg_cache,
    subgraph_k_hop_cache,
    subgraph_vertex_map_cache,
    num_vertex,
    device,
):
    subgraph_pyg = subgraph_pyg_cache[(t_start, t_end)].to(device)
    neighbors_k_hop = subgraph_k_hop_cache[(int(anchor), (t_start, t_end))]
    neighbors_k_hop = torch.tensor(neighbors_k_hop, device=device)
    sub_edge_index, sub_edge_attr = subgraph(
        subset=neighbors_k_hop,
        edge_index=subgraph_pyg.edge_index,
        edge_attr=subgraph_pyg.edge_attr,
        relabel_nodes=True,
    )

    vertex_map = torch.full((num_vertex,), -1, dtype=torch.long, device=device)
    vertex_map[neighbors_k_hop] = torch.arange(len(neighbors_k_hop), device=device)

    old_vertex_map = subgraph_vertex_map_cache[(t_start, t_end)].to(device)
    feature_matrix = subgraph_pyg.x[old_vertex_map[neighbors_k_hop], :]
    feature_matrix[vertex_map[int(anchor)]][0] = 1

    subgraph_pyg = Data(
        x=feature_matrix,
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
    )
    vertex_map = vertex_map.to(device)

    return subgraph_pyg, vertex_map


def extract_subgraph_multiple_query(
    query_vertex,
    t_start,
    t_end,
    k,
    feature_dim,
    temporal_graph,
    temporal_graph_pyg,
    num_vertex,
    edge_dim,
    sequence_features1_matrix,
    time_range_core_number,
    max_layer_id,
    max_time_range_layers,
    device,
    partition,
    num_timestamp,
    root,
):

    if not isinstance(query_vertex, torch.Tensor):
        query_vertex = torch.tensor(query_vertex, device=device)

    visited = set(query_vertex.cpu().numpy())

    queue = deque([(v.item(), 0) for v in query_vertex])
    neighbors_k_hop = set()

    while queue:
        top_vertex, depth = queue.popleft()
        if depth > k:
            continue
        neighbors_k_hop.add(top_vertex)
        for t, neighbors in temporal_graph[top_vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
                        visited.add(neighbor)

    neighbors_k_hop = torch.tensor(sorted(neighbors_k_hop), device=device)

    vertex_map = torch.full((num_vertex,), -1, dtype=torch.long)
    vertex_map[neighbors_k_hop] = torch.arange(len(neighbors_k_hop))
    vertex_map = vertex_map.to(device)

    mask = (vertex_map[temporal_graph_pyg.edge_index[0]] != -1) & (
        vertex_map[temporal_graph_pyg.edge_index[1]] != -1
    )
    mask = mask.to(device)

    sub_edge_index = temporal_graph_pyg.edge_index[:, mask]

    sub_edge_index = vertex_map[sub_edge_index]

    sub_edge_attr = temporal_graph_pyg.edge_attr.coalesce().to_dense()[mask]
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
        x=init_vertex_features_index(
            t_start,
            t_end,
            neighbors_k_hop,
            feature_dim,
            -1,
            sequence_features1_matrix,
            time_range_core_number,
            max_layer_id,
            max_time_range_layers,
            device,
            partition,
            num_timestamp,
            root,
        ),
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        device=device,
    )

    return subgraph_pyg, vertex_map, neighbors_k_hop


def extract_subgraph(
    anchor,
    t_start,
    t_end,
    k,
    feature_dim,
    temporal_graph,
    temporal_graph_pyg,
    num_vertex,
    edge_dim,
    sequence_features1_matrix,
    time_range_core_number,
    max_layer_id,
    max_time_range_layers,
    device,
    partition,
    num_timestamp,
    root,
):
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
        anchor,
        k,
        t_start,
        t_end,
        filtered_subgraph,
        total_edge_weight,
        time_range_core_number,
        None,
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

    edge_attr = temporal_graph_pyg.edge_attr.to_dense()
    sub_edge_attr = edge_attr[mask]
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
        x=init_vertex_features_index(
            t_start,
            t_end,
            candidate_neighbors,
            feature_dim,
            -1,
            sequence_features1_matrix,
            time_range_core_number,
            max_layer_id,
            max_time_range_layers,
            device,
            partition,
            num_timestamp,
            root,
        ),
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        num_nodes=len(candidate_neighbors),
        device=device,
    )
    return subgraph_pyg, vertex_map, candidate_neighbors
