import torch


def margin_triplet_loss(anchor, positives, negatives, margin=1):

    D_ap = torch.sum((anchor.unsqueeze(0) - positives) ** 2, dim=1)

    D_an = torch.sum((anchor.unsqueeze(0) - negatives) ** 2, dim=1)

    D_ap_expanded = D_ap.unsqueeze(1)
    D_an_expanded = D_an.unsqueeze(0)

    losses = torch.clamp(D_ap_expanded - D_an_expanded + margin, min=0.0)

    loss = losses.mean()
    return loss


def compute_link_loss(
    embeddings,
    vertex_map,
    node_indices,
    t_start,
    t_end,
    anchor,
    time_range_link_samples_cache,
    margin=0.2,
):

    link_samples_dict = time_range_link_samples_cache[(t_start, t_end)].get(anchor, {})
    if not link_samples_dict:
        return torch.tensor(0.0, device=embeddings.device)

    vertices = []
    positives = []
    negatives = []

    for vertex, samples in link_samples_dict.items():
        if vertex_map[vertex].item() != -1:
            for pos, neg in samples:
                if vertex_map[pos].item() != -1 and vertex_map[neg].item() != -1:
                    vertices.append(vertex)
                    positives.append(pos)
                    negatives.append(neg)

    if not vertices:
        return torch.tensor(0.0, device=embeddings.device)

    vertex_indices = node_indices[vertex_map[vertices].to(node_indices.device)]
    pos_indices = node_indices[vertex_map[positives].to(node_indices.device)]
    neg_indices = node_indices[vertex_map[negatives].to(node_indices.device)]

    vertex_embs = embeddings[vertex_indices.to(embeddings.device)]
    pos_embs = embeddings[pos_indices.to(embeddings.device)]
    neg_embs = embeddings[neg_indices.to(embeddings.device)]

    d_pos = torch.sum((vertex_embs - pos_embs) ** 2, dim=1)
    d_neg = torch.sum((vertex_embs - neg_embs) ** 2, dim=1)

    losses = torch.clamp(d_pos - d_neg + margin, min=0.0)

    return losses.mean()


def quadruplet_collate_fn(batch):
    anchors = torch.tensor([item[0] for item in batch], dtype=torch.long)
    positives = [item[1] for item in batch]
    negatives = [item[2] for item in batch]
    time_ranges = [item[3] for item in batch]
    return anchors, positives, negatives, time_ranges
