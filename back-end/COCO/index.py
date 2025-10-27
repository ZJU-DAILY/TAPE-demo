import torch


class HM_Index:
    def __init__(
        self,
        sequence_features1_matrix,
        partition,
        num_timestamp,
        root,
        max_layer_id,
        max_time_range_layers,
        device,
    ):
        self.sequence_features1_matrix = sequence_features1_matrix.clone()
        self.partition = partition
        self.num_timestamp = num_timestamp
        self.root = root
        self.max_layer_id = max_layer_id
        self.max_time_range_layers = max_time_range_layers
        self.device = device

    def predict(self, total_vertex_indices, t_start, t_end):
        return model_out_put_for_any_range_vertex_set(
            total_vertex_indices,
            t_start,
            t_end,
            max_layer_id=self.max_layer_id,
            max_time_range_layers=self.max_time_range_layers,
            device=self.device,
            sequence_features1_matrix=self.sequence_features1_matrix.clone(),
            partition=self.partition,
            num_timestamp=self.num_timestamp,
            root=self.root,
        )


def tree_query(time_start, time_end, num_timestamp, root, max_layer_id):
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


def model_output_for_path(
    time_start,
    time_end,
    vertex_set,
    sequence_features,
    num_timestamp,
    root,
    max_layer_id,
    device,
    max_time_range_layers,
    partition,
):
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
        len(vertex_set), max_time_range_layers[0], 2, device=device
    )
    sequence_input1[:, 0 : sequence_features.shape[1], :] = sequence_features
    for node in path:
        max_length1 = max_time_range_layers[node.layer_id + 1] * 2
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


def model_out_put_for_any_range_vertex_set(
    vertex_set,
    time_start,
    time_end,
    max_layer_id,
    max_time_range_layers,
    device,
    sequence_features1_matrix,
    partition,
    num_timestamp,
    root,
):

    node = tree_query(time_start, time_end, num_timestamp, root, max_layer_id)
    if node is None:
        print("Error: node not found.")
        return torch.zeros(len(vertex_set), 1, device=device)
    model = node.model
    # model.eval()
    if node.layer_id == max_layer_id:
        max_length = max_time_range_layers[node.layer_id]

        vertex_indices = torch.tensor(vertex_set, device=device)
        vertex_indices = torch.sort(vertex_indices).values

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

        single_value = model_output_for_path(
            time_start,
            time_end,
            vertex_set,
            sequence_features,
            num_timestamp,
            root,
            max_layer_id,
            device,
            max_time_range_layers,
            partition,
        )

        sequence_features = sequence_features.to(device)
        single_value = single_value.to(device)
        result_dict = {}
        with torch.no_grad():
            output = model(sequence_features, single_value)
            for i in range(len(vertex_set)):
                result_dict[int(vertex_set[i])] = int(round(output[i].item()))
        return result_dict
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
                sequence_features2[idx, idx2, 0] = core_number
                sequence_features2[idx, idx2, 1] = num_neighbor

        max_length = max_time_range_layers[node.layer_id + 1] * 2

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
            single_value = model_output_for_path(
                time_start,
                time_end,
                vertex_set,
                sequence_features1,
                num_timestamp,
                root,
                max_layer_id,
                device,
                max_time_range_layers,
                partition,
            )

        else:

            time_mask = (filtered_indices[1] >= time_start) & (
                filtered_indices[1] <= time_end
            ) & (filtered_indices[1]) < covered_nodes[0].time_start & (
                filtered_indices[1] > covered_nodes[-1].time_end
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

            single_value = model_output_for_path(
                time_start,
                time_end,
                vertex_set,
                sequence_features1_extra,
                num_timestamp,
                root,
                max_layer_id,
                device,
                max_time_range_layers,
                partition,
            )

        sequence_features1 = sequence_features1.to(device)
        sequence_features2 = sequence_features2.to(device)
        single_value = single_value.to(device)
        result_dict = {}
        with torch.no_grad():
            output = model(sequence_features1, sequence_features2, single_value)
            for i in range(len(vertex_set)):
                result_dict[int(vertex_set[i])] = int(round(output[i].item()))

        return result_dict
