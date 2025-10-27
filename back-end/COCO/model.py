import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch.nn import Parameter, ModuleList
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import GATConv, BatchNorm
from torch_scatter import scatter_mean, scatter_sum
from torch_scatter import scatter_softmax

from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from typing import Union, Tuple, Optional

from torch_geometric.utils.sparse import set_sparse_value


class AdapterTemporalGNN(nn.Module):
    def __init__(
        self,
        node_in_channels,
        node_out_channels,
        edge_dim,
        adapter_dim=16,
        use_adapter3=False,
    ):
        super(AdapterTemporalGNN, self).__init__()

        # ---------------------

        # Pre-aggregation adapter
        self.pre_adapter1 = Adapter(
            in_channels=node_in_channels, adapter_dim=adapter_dim, edge_dim=edge_dim
        )

        self.conv1 = GATWithEdgeChannelAttention(
            in_channels=node_in_channels,
            out_channels=64,
            heads=4,
            concat=True,
            edge_dim=edge_dim,
        )
        self.bn1 = nn.BatchNorm1d(64 * 4)

        # Post-aggregation adapter
        self.post_adapter1 = Adapter(
            in_channels=64 * 4, adapter_dim=adapter_dim, edge_dim=edge_dim
        )

        # ---------------------

        # ---------------------
        # Pre-aggregation adapter
        self.pre_adapter2 = Adapter(
            in_channels=64 * 4, adapter_dim=adapter_dim, edge_dim=edge_dim
        )

        self.conv2 = GATWithEdgeChannelAttention(
            in_channels=64 * 4,
            out_channels=32,
            heads=4,
            concat=True,
            edge_dim=edge_dim,
        )
        self.bn2 = nn.BatchNorm1d(32 * 4)

        # Post-aggregation adapter
        self.post_adapter2 = Adapter(
            in_channels=32 * 4, adapter_dim=adapter_dim, edge_dim=edge_dim
        )

        # ---------------------

        # ---------------------
        # Pre-aggregation adapter
        self.pre_adapter3 = Adapter(
            in_channels=32 * 4,  # Input channels from the previous layer
            adapter_dim=adapter_dim,
            edge_dim=edge_dim,
        )

        self.conv3 = GATWithEdgeChannelAttention(
            in_channels=32 * 4,
            out_channels=node_out_channels,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
        )

        # Post-aggregation adapter
        self.post_adapter3 = Adapter(
            in_channels=node_out_channels, adapter_dim=adapter_dim, edge_dim=edge_dim
        )

        # Gating parameters for all adapters
        self.gating_params = nn.ParameterDict(
            {
                "pre_gating1": nn.Parameter(torch.tensor(0.1)),
                "post_gating1": nn.Parameter(torch.tensor(0.1)),
                "pre_gating2": nn.Parameter(torch.tensor(0.1)),
                "post_gating2": nn.Parameter(torch.tensor(0.1)),
                "pre_gating3": nn.Parameter(torch.tensor(0.1)),
                "post_gating3": nn.Parameter(torch.tensor(0.1)),
            }
        )

    def forward(
        self,
        data,
        query_coreness=None,
        time_window_position=None,
        neighbor_coreness=None,
        time_window_length=None,
    ):
        data = data.to(self.gating_params["pre_gating1"].device)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Pre-aggregation adapter
        delta_pre1 = self.pre_adapter1(
            x,
            edge_index,
            edge_attr,
            query_coreness,
            time_window_position,
            neighbor_coreness,
            time_window_length,
        )
        x = x + self.gating_params["pre_gating1"].to(x.device) * (delta_pre1 - x)

        # GNN layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)

        # Post-aggregation adapter
        delta_post1 = self.post_adapter1(
            x,
            edge_index,
            edge_attr,
            query_coreness,
            time_window_position,
            neighbor_coreness,
            time_window_length,
        )
        x = x + self.gating_params["post_gating1"].to(x.device) * (delta_post1 - x)

        x = F.dropout(x, p=0.5, training=self.training)

        # Pre-aggregation adapter
        delta_pre2 = self.pre_adapter2(
            x,
            edge_index,
            edge_attr,
            query_coreness,
            time_window_position,
            neighbor_coreness,
            time_window_length,
        )
        x = x + self.gating_params["pre_gating2"].to(x.device) * (delta_pre2 - x)

        # GNN layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x)

        # Post-aggregation adapter
        delta_post2 = self.post_adapter2(
            x,
            edge_index,
            edge_attr,
            query_coreness,
            time_window_position,
            neighbor_coreness,
            time_window_length,
        )
        x = x + self.gating_params["post_gating2"].to(x.device) * (delta_post2 - x)

        x = F.dropout(x, p=0.5, training=self.training)

        # Pre-aggregation adapter
        delta_pre3 = self.pre_adapter3(
            x,
            edge_index,
            edge_attr,
            query_coreness,
            time_window_position,
            neighbor_coreness,
            time_window_length,
        )
        x = x + self.gating_params["pre_gating3"].to(x.device) * (delta_pre3 - x)

        # GNN layer
        x = self.conv3(x, edge_index, edge_attr)

        # Post-aggregation adapter
        delta_post3 = self.post_adapter3(
            x,
            edge_index,
            edge_attr,
            query_coreness,
            time_window_position,
            neighbor_coreness,
            time_window_length,
        )
        x = x + self.gating_params["post_gating3"].to(x.device) * (delta_post3 - x)

        return x


class Adapter(nn.Module):
    """
    上下文感知Adapter
    """

    def __init__(self, in_channels, adapter_dim, edge_dim):
        super(Adapter, self).__init__()

        self.down = nn.Linear(in_channels, adapter_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(adapter_dim, in_channels)

        self.time_proj = nn.Sequential(nn.Linear(edge_dim, adapter_dim), nn.ReLU())

        self.context_proj = nn.Sequential(nn.Linear(4, adapter_dim), nn.ReLU())

        self.efficient_fusion = HierarchicalTimeAttention(adapter_dim)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        query_coreness=None,
        time_window_length=None,
        time_window_position=None,
        neighbor_coreness=None,
    ):

        node_feat = self.down(x)
        node_feat = self.activation(node_feat)

        time_feat = self.time_proj(edge_attr)

        if all(
            v is not None
            for v in [
                query_coreness,
                time_window_length,
                time_window_position,
                neighbor_coreness,
            ]
        ):

            if not isinstance(query_coreness, torch.Tensor):
                query_coreness = torch.tensor(query_coreness, device=edge_attr.device)
            if not isinstance(time_window_length, torch.Tensor):
                time_window_length = torch.tensor(
                    time_window_length, device=edge_attr.device
                )
            if not isinstance(time_window_position, torch.Tensor):
                time_window_position = torch.tensor(
                    time_window_position, device=edge_attr.device
                )
            if not isinstance(neighbor_coreness, torch.Tensor):
                neighbor_coreness = torch.tensor(
                    neighbor_coreness, device=edge_attr.device
                )

            context_input = torch.stack(
                [
                    query_coreness,
                    time_window_length,
                    time_window_position,
                    neighbor_coreness,
                ],
                dim=-1,
            ).float()

            if context_input.dim() == 1:
                context_input = context_input.unsqueeze(0)

            context_feat = self.context_proj(context_input)

            if context_feat.size(0) == 1:
                context_feat = context_feat.expand(edge_attr.size(0), -1)
            else:

                context_feat = context_feat.repeat_interleave(
                    edge_attr.size(0) // context_feat.size(0), dim=0
                )
        else:

            context_feat = torch.zeros(
                edge_attr.size(0), self.down.out_features, device=edge_attr.device
            )

        fused = self.efficient_fusion(
            node_feat,  # [num_nodes, adapter_dim]
            time_feat,  # [num_edges, adapter_dim]
            context_feat,  # [num_edges, adapter_dim]
            edge_index,  # [2, num_edges]
        )

        out = self.up(fused)
        return x + out


class HierarchicalTimeAttention(nn.Module):
    """
    层次化时间注意力，先对时间特征进行聚类，再在聚类级别计算注意力
    """

    def __init__(self, adapter_dim, num_clusters=8):
        super(HierarchicalTimeAttention, self).__init__()
        self.num_clusters = num_clusters
        self.query = nn.Linear(adapter_dim, adapter_dim)
        self.key = nn.Linear(adapter_dim, adapter_dim)
        self.value = nn.Linear(adapter_dim, adapter_dim)
        self.cluster_embeddings = nn.Parameter(torch.randn(num_clusters, adapter_dim))
        self.activation = nn.ReLU()
        self.scaling = adapter_dim**-0.5
        self.out_proj = nn.Linear(adapter_dim, adapter_dim)

    def forward(self, node_feat, time_feat, context_feat, edge_index):
        q = self.query(node_feat)
        k = self.key(time_feat)
        v = self.value(time_feat)

        cluster_similarities = (
            torch.matmul(time_feat, self.cluster_embeddings.to(time_feat.device).t())
            * self.scaling
        )
        cluster_weights = F.softmax(cluster_similarities, dim=-1)

        cluster_assignments = torch.argmax(cluster_weights, dim=-1)

        src_idx = edge_index[0]
        q_i = q[src_idx]

        outputs = []
        for i in range(self.num_clusters):

            cluster_mask = cluster_assignments == i
            if not cluster_mask.any():
                continue

            masked_k = k[cluster_mask]
            masked_v = v[cluster_mask]
            masked_src_idx = src_idx[cluster_mask]
            masked_q_i = q_i[cluster_mask]

            attn = (masked_q_i * masked_k).sum(dim=-1) * self.scaling
            weights = scatter_softmax(attn, masked_src_idx, dim=0)
            weighted_values = masked_v * weights.unsqueeze(-1)

            cluster_output = scatter_mean(
                weighted_values, masked_src_idx, dim=0, dim_size=node_feat.size(0)
            )
            outputs.append(cluster_output)

        combined_output = sum(outputs) / len(outputs)
        return self.activation(self.out_proj(combined_output))


class GATWithEdgeChannelAttention(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        if isinstance(in_channels, int):
            self.lin = Linear(
                in_channels,
                heads * out_channels,
                bias=False,
                weight_initializer="glorot",
            )
        else:
            self.lin_src = Linear(
                in_channels[0], heads * out_channels, False, weight_initializer="glorot"
            )
            self.lin_dst = Linear(
                in_channels[1], heads * out_channels, False, weight_initializer="glorot"
            )

        if edge_dim is not None:
            self.edge_channel_attention = nn.Sequential(
                Linear(edge_dim, edge_dim // 2),
                nn.ReLU(),
                Linear(edge_dim // 2, edge_dim),
                nn.Sigmoid(),
            )

            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))
        else:
            self.edge_channel_attention = None

            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias:
            self.bias = Parameter(
                torch.empty(heads * out_channels if concat else out_channels)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lin"):
            self.lin.reset_parameters()
        else:
            self.lin_src.reset_parameters()
            self.lin_dst.reset_parameters()

        if self.edge_channel_attention is not None:
            for layer in self.edge_channel_attention:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

        glorot(self.att)
        zeros(self.bias)

    def forward(
        self, x, edge_index, edge_attr=None, size=None, return_attention_weights=None
    ):
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x_src = x_dst = self.lin(x).view(-1, H, C)
        else:
            x_src, x_dst = x[0], x[1]
            x_src = self.lin_src(x_src).view(-1, H, C)
            x_dst = self.lin_dst(x_dst).view(-1, H, C) if x_dst is not None else None

        x = (x_src, x_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index,
                    edge_attr,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes,
                )

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            alpha = self._alpha
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            else:
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        """
        计算消息和注意力权重
        x_i: 目标节点特征
        x_j: 源节点特征
        edge_attr: 边特征
        """

        if edge_attr is not None and self.edge_channel_attention is not None:

            channel_weights = self.edge_channel_attention(edge_attr)

            edge_attr_weighted = edge_attr * channel_weights

            alpha = torch.cat(
                [x_i, x_j, edge_attr_weighted.unsqueeze(1).expand(-1, self.heads, -1)],
                dim=-1,
            )
        else:

            alpha = torch.cat([x_i, x_j], dim=-1)

        alpha = (alpha * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha

        # 4. dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )
