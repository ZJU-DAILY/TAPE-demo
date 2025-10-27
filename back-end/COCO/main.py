import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
import time
from torch_geometric.data import TemporalData
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split

from model import *
from tqdm import tqdm

from data_loader import *
from utils import *
from loss import *
from extract_subgraph import *
from train import *
from index import *
from MLP_search import *


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset_name = "collegemsg"
    # dataset_name = 'dewiki'
    # dataset_name = 'mathoverflow'
    # dataset_name = 'flickr'
    # dataset_name = 'wikitalk'
    # dataset_name = 'youtube'
    # dataset_name = 'superuser'
    # dataset_name = 'stackoverflow'
    # dataset_name = 'dblp'

    num_vertex = 0
    num_timestamp = 0
    vertex_core_numbers = []
    time_range_core_number = defaultdict(dict)
    temporal_graph = []
    time_range_layers = []
    time_range_set = set()
    max_time_range_layers = []
    sequence_features1_matrix = torch.empty(0, 0, 0)
    partition = 4
    max_range = 45
    max_layer_id = 0

    # GNN
    temporal_graph_pyg = TemporalData()
    subgraph_k_hop_cache = {}
    subgraph_pyg_cache = {}
    subgraph_vertex_map_cache = {}
    time_range_link_samples_cache = defaultdict(
        dict
    )  # {(t_start, t_end): {anchor: {vertex: [(pos, neg), ...]}}}

    node_in_channels = 8
    node_out_channels = 16
    edge_dim = 8
    learning_rate = 0.001
    epochs = 200
    batch_size = 8
    k_hop = 5
    alpha = 0.1
    num_time_range_samples = 5
    num_anchor_samples = 100
    test_result_list = []

    num_vertex, num_timestamp, temporal_graph, temporal_graph_pyg = read_temporal_graph(
        dataset_name, device
    )

    time_range_layers, max_time_range_layers, time_range_set, max_layer_id = (
        get_timerange_layers(num_timestamp, max_range, partition)
    )

    windows_number_set = set()

    for sublist in time_range_layers:
        for tup in sublist:
            for number in tup:
                windows_number_set.add(number)

    vertex_core_numbers, time_range_core_number = read_core_number(
        dataset_name, num_vertex, time_range_set
    )

    sequence_features1_matrix = construct_feature_matrix(
        num_vertex, num_timestamp, temporal_graph, vertex_core_numbers, device
    )

    model = AdapterTemporalGNN(
        node_in_channels, node_out_channels, edge_dim=edge_dim
    ).to(device)

    ################  search with trained model  ####################################

    train_time_range_list = []
    while len(train_time_range_list) < num_time_range_samples:
        t_layer = random.randint(0, len(time_range_layers) - 1)
        if t_layer == 0:
            continue
        t_idx = random.randint(0, len(time_range_layers[t_layer]) - 1)
        t_start, t_end = (
            time_range_layers[t_layer][t_idx][0],
            time_range_layers[t_layer][t_idx][1],
        )
        if (t_start, t_end) not in train_time_range_list:
            train_time_range_list.append((t_start, t_end))

    quadruplet = []

    for i, time_range in enumerate(train_time_range_list):
        t_start = time_range[0]
        t_end = time_range[1]
        center_vertices = set()
        select_limit = 50
        select_cnt = 0

        if (t_start, t_end) not in time_range_core_number or not time_range_core_number[
            (t_start, t_end)
        ]:
            print(f"train跳过空的时间范围 [{t_start}, {t_end}]")
            continue

        while len(center_vertices) < num_anchor_samples:
            if select_cnt >= select_limit:
                break
            temp_vertex = random.choice(
                list(time_range_core_number[(t_start, t_end)].keys())
            )
            temp_core_number = time_range_core_number[(t_start, t_end)][temp_vertex]
            min_core_numer = 5
            if t_end - t_start > 300:
                min_core_numer = 10
            if temp_core_number >= min_core_numer:
                if temp_vertex not in center_vertices:
                    center_vertices.add(temp_vertex)
                    select_cnt = 0
                else:
                    select_cnt += 1
            else:
                select_cnt += 1
        if len(center_vertices) == 0:
            continue

        triplets = generate_triplets(
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
        for triplet in triplets:
            quadruplet.append((triplet[0], triplet[1], triplet[2], (t_start, t_end)))

        temp_subgraph_pyg, vertex_map = extract_subgraph_for_time_range(
            center_vertices,
            t_start,
            t_end,
            node_in_channels,
            temporal_graph_pyg,
            num_vertex,
            edge_dim,
            device,
            sequence_features1_matrix,
            time_range_core_number,
            subgraph_k_hop_cache,
        )

        temp_subgraph_pyg = temp_subgraph_pyg.to("cpu")

        vertex_map = vertex_map.to("cpu")
        subgraph_pyg_cache[(t_start, t_end)] = temp_subgraph_pyg
        subgraph_vertex_map_cache[(t_start, t_end)] = vertex_map
        torch.cuda.empty_cache()
        # print("done3")

    test_time_range_list = []
    max_attempts = 1000
    attempts = 0
    while (
        len(test_time_range_list) < num_time_range_samples and max_attempts > attempts
    ):
        t_layer = random.randint(0, len(time_range_layers) - 1)
        attempts += 1
        if t_layer == 0:
            continue
        t_idx = random.randint(0, len(time_range_layers[t_layer]) - 1)
        t_start, t_end = (
            time_range_layers[t_layer][t_idx][0],
            time_range_layers[t_layer][t_idx][1],
        )
        if (t_start, t_end) not in test_time_range_list and (
            t_start,
            t_end,
        ) not in train_time_range_list:
            if (
                t_start,
                t_end,
            ) not in time_range_core_number or not time_range_core_number[
                (t_start, t_end)
            ]:
                print(f"跳过选择空的时间范围 [{t_start}, {t_end}]")
                continue
            test_time_range_list.append((t_start, t_end))

    test_quadruplet = []
    for i, time_range in enumerate(test_time_range_list):
        t_start = time_range[0]
        t_end = time_range[1]
        if (
            t_start,
            t_end,
        ) not in time_range_core_number and not time_range_core_number[
            (t_start, t_end)
        ]:
            print(f"跳过空的时间范围 [{t_start}, {t_end}]")
            continue
        center_vertices = set()
        select_limit = 50
        select_cnt = 0
        min_core_numer = 2
        if t_end - t_start > 300:
            min_core_numer = 10
        while len(center_vertices) < num_anchor_samples:
            temp_vertex = random.choice(
                list(time_range_core_number[(t_start, t_end)].keys())
            )
            temp_core_number = time_range_core_number[(t_start, t_end)][temp_vertex]
            if select_cnt >= select_limit:
                break
            if temp_core_number >= min_core_numer:
                if temp_vertex not in center_vertices:
                    center_vertices.add(temp_vertex)
                    select_cnt = 0
                else:
                    select_cnt += 1
            else:
                select_cnt += 1
        if len(center_vertices) == 0:
            continue
        triplets = generate_triplets(
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
        for triplet in triplets:
            test_quadruplet.append(
                (triplet[0], triplet[1], triplet[2], (t_start, t_end))
            )
        # generate pyg data for the subgraph of the current time range
        temp_subgraph_pyg, vertex_map = extract_subgraph_for_time_range(
            center_vertices,
            t_start,
            t_end,
            node_in_channels,
            temporal_graph_pyg,
            num_vertex,
            edge_dim,
            device,
            sequence_features1_matrix,
            time_range_core_number,
            subgraph_k_hop_cache,
        )

        temp_subgraph_pyg = temp_subgraph_pyg.to("cpu")
        vertex_map = vertex_map.to("cpu")
        subgraph_pyg_cache[(t_start, t_end)] = temp_subgraph_pyg
        subgraph_vertex_map_cache[(t_start, t_end)] = vertex_map
        torch.cuda.empty_cache()

    train_quadruplet = quadruplet
    if test_quadruplet is not None:
        try:
            val_quadruplet, test_quadruplet = train_test_split(
                test_quadruplet, test_size=0.5, random_state=42
            )
        except Exception as e:
            print(f"发生异常: {e}")
            val_quadruplet, test_quadruplet = None, None

    test_dataset = MultiSampleQuadrupletDataset(test_quadruplet)

    train_dataset = MultiSampleQuadrupletDataset(train_quadruplet)
    val_dataset = MultiSampleQuadrupletDataset(val_quadruplet)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=quadruplet_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=quadruplet_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=quadruplet_collate_fn,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    feature_dim = node_in_channels

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0

    total_train_time = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            train_time = time.time()
            anchors, positives, negatives, time_ranges = batch
            anchors = torch.tensor(anchors, device=device).clone()
            positives = [torch.tensor(pos, device=device) for pos in positives]
            negatives = [torch.tensor(neg, device=device) for neg in negatives]

            optimizer.zero_grad()

            subgraphs = []
            vertex_maps = []
            for anchor, time_range in zip(anchors, time_ranges):
                subgraph_pyg, vertex_map = extract_subgraph_for_anchor(
                    anchor.item(),
                    time_range[0],
                    time_range[1],
                    subgraph_pyg_cache,
                    subgraph_k_hop_cache,
                    subgraph_vertex_map_cache,
                    num_vertex,
                    device,
                )
                if len(vertex_map) == 0:
                    continue

                subgraphs.append(subgraph_pyg)
                vertex_maps.append(vertex_map)

            batched_subgraphs = Batch.from_data_list(subgraphs).to(device)

            try:
                embeddings = model(batched_subgraphs)
            except RuntimeError as e:
                print(f"Error in batch: {e}")
                continue

            torch.cuda.synchronize()

            batch_indices = batched_subgraphs.batch

            # release the memory of subgraphs
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
                pos_samples = pos_samples.to(device=vertex_map.device).long()
                neg_samples = neg_samples.to(device=vertex_map.device).long()
                pos_mapped_indices = vertex_map[pos_samples]
                neg_mapped_indices = vertex_map[neg_samples]
                pos_indices = node_indices[pos_mapped_indices.to(node_indices.device)]
                neg_indices = node_indices[neg_mapped_indices.to(node_indices.device)]

                if len(pos_indices) == 0 or len(neg_indices) == 0:
                    continue

                anchor_emb = embeddings[anchor_idx]
                positive_emb = embeddings[pos_indices.to(embeddings.device)]
                negative_emb = embeddings[neg_indices.to(embeddings.device)]

                with autocast():
                    loss = margin_triplet_loss(anchor_emb, positive_emb, negative_emb)
                    link_loss_value = compute_link_loss(
                        embeddings,
                        vertex_map,
                        node_indices,
                        time_range[0],
                        time_range[1],
                        anchor.item(),
                        time_range_link_samples_cache,
                        margin=0.2,
                    )
                    loss += alpha * link_loss_value
                    # loss += alpha * link_loss(subgraph_emb, subgraph_pyg)
                    # loss += alpha * link_loss(vertex_map, subgraph_emb, time_range[0], time_range[1])
                batch_loss += loss
                torch.cuda.empty_cache()

            if len(vertex_maps) > 0:
                batch_loss = batch_loss / len(vertex_maps)

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += batch_loss.item()
            progress_bar.set_postfix(
                loss=loss.item(), avg_loss=epoch_loss / (batch_idx + 1)
            )
            torch.cuda.empty_cache()
            total_train_time += time.time() - train_time

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}",
            end=" ",
        )

        val_loss = validate_model(
            model,
            val_loader,
            device,
            alpha,
            subgraph_pyg_cache,
            subgraph_k_hop_cache,
            subgraph_vertex_map_cache,
            num_vertex,
            time_range_link_samples_cache,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), f"model_L1_{dataset_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")

                model.load_state_dict(torch.load(f"model_L1_{dataset_name}.pth"))
                break

    if test_loader is not None:
        avg_sim_pos, avg_sim_neg = test_model(
            model,
            test_loader,
            device,
            subgraph_pyg_cache,
            subgraph_k_hop_cache,
            subgraph_vertex_map_cache,
            num_vertex,
        )
        test_result_list.append((avg_sim_pos, avg_sim_neg))
    print(f"the total train time for {dataset_name} is {total_train_time}")

    ################# search with trained model ###################


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
