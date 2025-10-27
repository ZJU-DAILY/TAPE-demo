import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class MLP(nn.Module):
    def __init__(self, input_dim, max_seq_length, hidden_dim):
        super(MLP, self).__init__()

        self.sequence_fc = nn.Linear(input_dim * max_seq_length, hidden_dim)
        self.additional_fc = nn.Linear(1, hidden_dim)

        self.relu = nn.ReLU()

        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, additional_feature):

        first_col = x[:, :, 0]
        second_col = x[:, :, 1]
        sorted_indices = torch.argsort(-first_col * 1e5 - second_col, dim=1)
        b, n, c = x.shape
        x = torch.gather(x, 1, sorted_indices.unsqueeze(2).expand(b, n, c))

        x = x.view(x.size(0), -1)
        x = self.sequence_fc(x)
        x = self.relu(x)

        if additional_feature.dim() == 1:
            additional_feature = additional_feature.unsqueeze(1)
        additional_feature = self.additional_fc(additional_feature)
        additional_feature = self.relu(additional_feature)

        combined = torch.cat((x, additional_feature), dim=1)
        combined = self.fusion_fc(combined)
        combined = self.relu(combined)

        output = self.output_fc(combined)
        return output.squeeze()


class MLPNonleaf(nn.Module):
    def __init__(self, input_dim, max_seq_length1, max_seq_length2, hidden_dim):
        super(MLPNonleaf, self).__init__()

        self.sequence_fc1 = nn.Linear(input_dim * max_seq_length1, hidden_dim)
        self.sequence_fc2 = nn.Linear(input_dim * max_seq_length2, hidden_dim)

        self.single_feature_fc = nn.Linear(1, hidden_dim)

        self.relu = nn.ReLU()

        self.fusion_fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 1)

    # @profile
    def forward(self, sequence1, sequence2, single_feature):

        first_col = sequence1[:, :, 0]
        second_col = sequence1[:, :, 1]
        sorted_indices = torch.argsort(-first_col * 1e5 - second_col, dim=1)
        b, n, c = sequence1.shape
        sequence1 = torch.gather(
            sequence1, 1, sorted_indices.unsqueeze(2).expand(b, n, c)
        )

        x1 = sequence1.view(sequence1.size(0), -1)
        x1 = self.sequence_fc1(x1)
        x1 = self.relu(x1)

        first_col = sequence2[:, :, 0]
        second_col = sequence2[:, :, 1]
        sorted_indices = torch.argsort(-first_col * 1e5 - second_col, dim=1)
        b, n, c = sequence2.shape
        sequence2 = torch.gather(
            sequence2, 1, sorted_indices.unsqueeze(2).expand(b, n, c)
        )

        x2 = sequence2.view(sequence2.size(0), -1)
        x2 = self.sequence_fc2(x2)
        x2 = self.relu(x2)

        if single_feature.dim() == 1:
            single_feature = single_feature.unsqueeze(1)
        x3 = self.single_feature_fc(single_feature)
        x3 = self.relu(x3)

        combined = torch.cat((x1, x2, x3), dim=1)
        combined = self.fusion_fc(combined)
        combined = self.relu(combined)

        output = self.output_fc(combined)
        return output.squeeze()


def preprocess_sequences(sequences, max_length):
    sequences = [
        torch.tensor(seq[:max_length], dtype=torch.float32) for seq in sequences
    ]
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    if sequences_padded.size(1) < max_length:
        pad_size = max_length - sequences_padded.size(1)
        padding = torch.zeros(
            (sequences_padded.size(0), pad_size, sequences_padded.size(2)),
            dtype=torch.float32,
        )
        sequences_padded = torch.cat((sequences_padded, padding), dim=1)

    return sequences_padded


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, additional_features, max_seq_length):
        self.sequences = preprocess_sequences(sequences, max_seq_length)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.additional_features = torch.tensor(
            additional_features, dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.additional_features[idx], self.labels[idx]


class SequenceDatasetNonleaf(Dataset):
    def __init__(
        self,
        sequences,
        sequences_range,
        labels,
        additional_features,
        max_seq_length1,
        max_seq_length2,
    ):
        self.sequences = preprocess_sequences(sequences, max_seq_length1)
        self.sequences_range = preprocess_sequences(
            sequences_range, max_seq_length2
        )  # New
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.additional_features = torch.tensor(
            additional_features, dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.sequences_range[idx],
            self.additional_features[idx],
            self.labels[idx],
        )
