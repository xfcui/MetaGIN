import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.parameter as nnp
import torch.nn.functional as nnf

from torch_scatter import scatter
from torch_geometric.utils import degree, softmax


DROPOUT = 0.1
EMBED_ATOM = 172            # zero for masked
EMBED_BOND = [33, 3, 4]     # zero for masked
EMBED_DIST = 8              # zero for self-loop
DIM_HEAD, WIDTH_GRAPH, WIDTH_NODE, WIDTH_EDGE = 16, (2, 3), (2, 2), (1, 2)


# ReZero: https://arxiv.org/abs/2003.04887v2
# LayerScale: https://arxiv.org/abs/2103.17239v2
class ScaleLayer(nn.Module):
    def __init__(self, width, scale_init):
        super().__init__()
        self.scale = nnp.Parameter(pt.zeros(width) + np.log(scale_init))

    def forward(self, x):
        return pt.exp(self.scale) * x

# PNA: https://arxiv.org/abs/2004.05718
# Graphormer: https://arxiv.org/abs/2106.05234v5
class DegreeLayer(nn.Module):
    def __init__(self, width, degree_init):
        super().__init__()
        self.degree = nnp.Parameter(pt.zeros(width) + degree_init)

    def forward(self, x, deg):
        return pt.pow(deg.unsqueeze(-1), self.degree) * x


# GLU: https://arxiv.org/abs/1612.08083v3
# GLU-variants: https://arxiv.org/abs/2002.05202v1
class GatedLinearBlock(nn.Module):
    def __init__(self, width, num_head, width_norm=1, width_act=1):
        super().__init__()

        self.pre   = nn.Sequential(nn.Conv1d(width, width*width_norm, 1),
                         nn.GroupNorm(num_head, width*width_norm, affine=False))
        self.gate  = nn.Sequential(nn.Conv1d(width*width_norm, width*width_act, 1, bias=False, groups=num_head),
                         nn.ReLU(), nn.Dropout(DROPOUT))
        self.value = nn.Conv1d(width*width_norm, width*width_act, 1, bias=False, groups=num_head)
        self.post  = nn.Conv1d(width*width_act, width, 1)

    def forward(self, x, bias=0):
        xx = self.pre(x.unsqueeze(-1))
        xx = self.gate(xx + bias) * self.value(xx)
        xx = self.post(xx).squeeze(-1)
        return xx

# MetaFormer: https://arxiv.org/abs/2210.13452v1
class MetaFormerBlock(nn.Module):
    def __init__(self, width, num_head, width_norm=1, width_act=1, use_residual=True, block_name=None):
        super().__init__()

        self.scale_pre  = ScaleLayer(width, 1)
        self.scale_post = ScaleLayer(width, 1) if use_residual else None
        self.block      = GatedLinearBlock(width, num_head, width_norm, width_act)
        if block_name is not None:
            print('##params[%s]:' % block_name, np.sum([np.prod(p.shape) for p in self.parameters()]), use_residual)

    def forward(self, x, res):
        xx = self.scale_pre(x) + res
        if self.scale_post is None:
            xx = self.block(xx)
        else:
            xx = self.scale_post(xx) + self.block(xx)
        return xx


# VoVNet: https://arxiv.org/abs/1904.09730v1
# GNN-AK: https://openreview.net/forum?id=Mspk_WYKoEH
class ConvBlock(nn.Module):
    def __init__(self, width, num_head, width_norm, width_act, bond_size, degree_init=-0.01):  # sum_init
        super().__init__()

        # ops per node
        self.pre0  = nn.Conv1d(width, width*width_norm, 1)
        self.pre1  = nn.Conv1d(width, width*width_norm, 1)
        self.norm  = nn.GroupNorm(num_head, width*width_norm, affine=False)
        # ops per edge
        self.local_encoder = nn.EmbeddingBag(bond_size, width*width_norm, scale_grad_by_freq=True, padding_idx=0)
        self.gate  = nn.Sequential(nn.Conv1d(width*width_norm, width*width_act, 1, bias=False, groups=num_head),
                         nn.ReLU(), nn.Dropout(DROPOUT))
        self.value = nn.Conv1d(width*width_norm, width*width_act, 1, bias=False, groups=num_head)
        # ops per node
        self.post  = nn.Conv1d(width*width_act, width, 1)
        self.deg   = DegreeLayer(width, degree_init)

    def forward(self, x, deg, edge_idx, edge_attr):
        xx = self.norm(self.pre0(x.unsqueeze(-1))[edge_idx[0]] + self.pre1(x.unsqueeze(-1))[edge_idx[1]])
        xx = self.gate(xx + self.local_encoder(edge_attr).unsqueeze(-1)) * self.value(xx)
        xx = scatter(xx, edge_idx[1], dim=0, dim_size=len(x), reduce='sum')
        xx = self.deg(self.post(xx).squeeze(-1), deg)
        return xx

class ConvKernel(nn.Module):
    def __init__(self, width, num_head, hop, kernel, width_norm=1, width_act=1):
        super().__init__()
        self.width = width
        self.hop = hop
        self.kernel = kernel

        self.msg = nn.ModuleList()
        for layer in range(hop * kernel):
            hop_idx = layer % hop
            self.msg.append(ConvBlock(width, num_head, width_norm, width_act, EMBED_BOND[hop_idx]))
        self.mix = nn.ModuleList()
        for layer in range(kernel-1):
            self.mix.append(MetaFormerBlock(width, num_head, *WIDTH_NODE))
        print('##params[conv]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, x_res, edge):
        x_kernel = x_hop = x; x_out = 0
        for kernel in range(self.kernel):
            for hop in range(self.hop):
                layer = kernel * self.hop + hop
                edge_index, edge_attr, node_deg = edge[hop]

                x_hop = self.msg[layer](x_hop, node_deg, edge_index, edge_attr)
                x_out = x_out + x_hop
            if kernel < len(self.mix):
                x_kernel = x_hop = self.mix[kernel](x_kernel, x_out); x_out = 0
        return x_out + x_res


# GIN-virtual: https://arxiv.org/abs/2103.09430
class VirtKernel(nn.Module):
    def __init__(self, width, num_head):
        super().__init__()
        self.width = width

        self.virt  = GatedLinearBlock(width, num_head, *WIDTH_GRAPH)
        print('##params[virt]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, x_res, virt_res, batch, batch_size):
        xx = virt_res = scatter(x, batch, dim=0, dim_size=len(batch_size), reduce='sum') + virt_res
        xx = self.virt(xx)[batch]
        return xx + x_res, virt_res


class HeadBlock(nn.Module):
    def __init__(self, width, num_head, degree_init=-0.99):  # mean_init
        super().__init__()
        self.width = width

        self.virt = MetaFormerBlock(width, num_head, *WIDTH_GRAPH, False)

        self.node = GatedLinearBlock(width, num_head, *WIDTH_NODE)
        self.node_degree = DegreeLayer(width, degree_init)

        self.norm = nn.GroupNorm(1, width, affine=False)
        self.head = nn.Linear(width, 1)
        print('##params[head]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, virt_res, batch, batch_size):
        x0 = scatter(x, batch, dim=0, dim_size=len(batch_size), reduce='sum')
        x0 = self.virt(virt_res, x0)

        x1 = self.node(x)
        x1 = scatter(x1, batch, dim=0, dim_size=len(batch_size), reduce='sum')
        x1 = self.node_degree(x1, batch_size)

        xn = self.norm(x0 + x1)
        xx = (self.head(xn) + 1.0) * 5.5
        return xx, xn


# GIN: https://openreview.net/forum?id=ryGs6iA5Km
class MetaGIN(nn.Module):
    def __init__(self, depth, num_head, conv_hop, conv_kernel, use_virt):
        super().__init__()
        self.depth = depth
        self.width = width = DIM_HEAD * num_head
        self.conv_hop = conv_hop
        self.conv_kernel = conv_kernel
        self.use_virt = use_virt
        print('#model:', depth, width, num_head, conv_hop, conv_kernel, use_virt)

        self.atom_encoder = nn.EmbeddingBag(EMBED_ATOM, width, scale_grad_by_freq=True, padding_idx=0)
        self.atom_conv = ConvKernel(width, num_head, conv_hop, 1, *WIDTH_EDGE)
        self.atom_main = MetaFormerBlock(width, num_head, *WIDTH_NODE, False, 'main')

        self.conv = nn.ModuleList()
        self.virt = nn.ModuleList()
        self.main = nn.ModuleList()
        for layer in range(depth):
            self.conv.append(ConvKernel(width, num_head, conv_hop, conv_kernel[layer], *WIDTH_EDGE) if conv_kernel[layer] > 0 else None)
            self.virt.append(VirtKernel(width, num_head) if use_virt else None)
            if layer < depth-1:
                self.main.append(MetaFormerBlock(width, num_head, *WIDTH_NODE, False, 'main'))
            else:
                self.main.append(MetaFormerBlock(width, num_head, *WIDTH_NODE, True, 'main'))

        self.head = HeadBlock(width, num_head)
        print('#params:', np.sum([np.prod(p.shape) for n, p in self.named_parameters()]))

    def forward(self, graph):
        x, batch, batch_size = graph['atom'].x, graph['atom'].batch, graph['atom'].ptr[1:]-graph['atom'].ptr[:-1]

        idx1, attr1 = graph['bond'].edge_index, graph['bond'].edge_attr
        deg1 = degree(idx1[1], graph.num_nodes).float(); deg1.clamp_(1, None)
        edge = [[idx1, attr1, deg1]]

        idx2, attr2 = graph['angle'].edge_index, graph['angle'].edge_attr
        attr2.clamp_(None, EMBED_BOND[1]-1)
        deg2 = degree(idx2[1], graph.num_nodes).float()
        deg2.clamp_(1, None)
        edge += [[idx2, attr2, deg2]]

        idx3, attr3 = graph['torsion'].edge_index, graph['torsion'].edge_attr
        attr3.clamp_(None, EMBED_BOND[2]-1)
        deg3 = degree(idx3[1], graph.num_nodes).float()
        deg3.clamp_(1, None)
        edge += [[idx3, attr3, deg3]]

        #idx0, attr0 = graph['pair'].edge_index, graph['pair'].edge_attr
        #attr0.clamp_(None, EMBED_DIST-2); attr0.masked_fill_(attr0<0, EMBED_DIST-1)
        #deg0 = batch_size * (batch_size - 1) / 2
        #deg0.clamp_(1, None)
        #edge += [[idx0, attr0, deg0]]

        h_in, h_virt, h_out = self.atom_encoder(x), 0, 0
        h_out = self.atom_conv(h_in, h_out, edge)
        h_in = self.atom_main(h_in, h_out)
        for layer in range(self.depth):
            h_out = 0
            if self.conv[layer] is not None:
                h_out = self.conv[layer](h_in, h_out, edge)
            if self.virt[layer] is not None:
                h_out, h_virt = self.virt[layer](h_in, h_out, h_virt, batch, batch_size)
            h_in = self.main[layer](h_in, h_out)
        h_out, h_embed = self.head(h_in, h_virt, batch, batch_size)

        return h_out

