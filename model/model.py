import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.parameter as nnp
import torch.nn.functional as nnf

from torch_scatter import scatter
from torch_geometric.utils import degree
from torch.distributions.log_normal import LogNormal


DROPOUT = 0.1
EMBED_POS  = 16          # random walk positional embedding
EMBED_ATOM = 138         # zero for masked
EMBED_BOND = [33, 3, 4]  # zero for masked
EMBED_DIST = 8           # zero for self-loop
RESCALE_GRAPH, RESCALE_NODE, RESCALE_EDGE = (2, 4), (1, 3), (1, 2)


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
    def __init__(self, width, num_head, resca_norm=1, resca_act=1, skip_pre=False, width_in=None, width_out=None):
        super().__init__()
        self.width = width
        assert width >= 256
        self.nhead = num_head * resca_norm
        self.dhead = width * resca_act // self.nhead
        if width_in is None: width_in = width
        width_norm = width * resca_norm
        width_act = width * resca_act
        if width_out is None: width_out = width

        if skip_pre:
            self.pre = nn.GroupNorm(self.nhead, width_norm, affine=False)
        else:
            self.pre = nn.Sequential(nn.Conv1d(width_in, width_norm, 1, bias=False),
                           nn.GroupNorm(self.nhead, width_norm, affine=False))
        self.gate  = nn.Sequential(nn.Conv1d(width_norm, width_act, 1, bias=False, groups=self.nhead),
                         nn.ReLU(), nn.Dropout(DROPOUT))
        self.value = nn.Conv1d(width_norm, width_act, 1, bias=False, groups=self.nhead)
        self.post  = nn.Conv1d(width_act, width_out, 1, bias=False)

    def forward(self, x, gate_bias=None, out_norm=False):
        xn = self.pre(x.unsqueeze(-1))
        if gate_bias is None:
            xx = self.gate(xn) * self.value(xn)
        else:
            xx = self.gate(xn + gate_bias.unsqueeze(-1)) * self.value(xn)
        xx = self.post(xx).squeeze(-1)
        if out_norm:
            return xx, xn.squeeze(-1)
        else:
            return xx

# MetaFormer: https://arxiv.org/abs/2210.13452v1
class MetaFormerBlock(nn.Module):
    def __init__(self, width, num_head, resca_norm=1, resca_act=1, use_residual=True, name=None):
        super().__init__()
        self.width = width
        self.nhead = num_head * resca_norm
        self.dhead = width * resca_act // self.nhead

        self.sca_pre  = ScaleLayer(width, 1)
        self.sca_post = ScaleLayer(width, 1) if use_residual else None
        self.ffn        = GatedLinearBlock(width, num_head, resca_norm, resca_act)
        if name is not None:
            print('##params[%s]:' % name, np.sum([np.prod(p.shape) for p in self.parameters()]), use_residual)

    def forward(self, x, res):
        xx = self.sca_pre(x) + res
        if self.sca_post is None:
            xx = self.ffn(xx)
        else:
            xx = self.sca_post(xx) + self.ffn(xx)
        return xx


# VoVNet: https://arxiv.org/abs/1904.09730v1
# GNN-AK: https://openreview.net/forum?id=Mspk_WYKoEH
class ConvBlock(nn.Module):
    def __init__(self, width, num_head, bond_size, degree_init=-0.01):  # sum_init
        super().__init__()
        self.width = width
        width_norm = width * RESCALE_EDGE[0]

        self.src = nn.Conv1d(width, width_norm, 1, bias=False)
        self.tgt = nn.Conv1d(width, width_norm, 1, bias=False)
        self.conv_encoder = nn.EmbeddingBag(bond_size, width_norm, scale_grad_by_freq=True, padding_idx=0)
        self.fft = GatedLinearBlock(width, num_head, *RESCALE_EDGE, skip_pre=True)
        self.deg = DegreeLayer(width, degree_init)

    def forward(self, x, deg, edge_idx, edge_attr):
        xx = self.src(x.unsqueeze(-1))[edge_idx[0]] + self.tgt(x.unsqueeze(-1))[edge_idx[1]]
        xx = self.fft(xx.squeeze(-1), self.conv_encoder(edge_attr))
        xx = scatter(xx, edge_idx[1], dim=0, dim_size=len(x), reduce='sum')
        xx = self.deg(xx, deg)
        return xx

class ConvKernel(nn.Module):
    def __init__(self, width, num_head, hop, kernel):
        super().__init__()
        self.width = width
        self.hop = hop
        self.kernel = kernel

        self.msg = nn.ModuleList()
        for k in range(kernel):
            for h in range(hop):
                self.msg.append(ConvBlock(width, num_head, EMBED_BOND[h]))
        print('##params[conv]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, x_res, edge):
        x_kernel = x_hop = x; x_out = 0
        for k in range(self.kernel):
            for h in range(self.hop):
                edge_index, edge_attr, node_deg = edge[h]
                x_hop = self.msg[k*self.hop+h](x_hop, node_deg, edge_index, edge_attr)
                x_out = x_out + x_hop
        return x_out + x_res


# GIN-virtual: https://arxiv.org/abs/2103.09430
class VirtKernel(nn.Module):
    def __init__(self, width, num_head):
        super().__init__()
        self.width = width

        self.virt  = GatedLinearBlock(width, num_head, *RESCALE_GRAPH)
        print('##params[virt]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, x_res, virt_res, batch, batch_size):
        xx = virt_res = scatter(x, batch, dim=0, dim_size=len(batch_size), reduce='sum') + virt_res
        xx = self.virt(xx)[batch]
        return xx + x_res, virt_res


class HeadBlock(nn.Module):
    def __init__(self, width, num_head, degree_init=-0.99):  # mean_init
        super().__init__()
        self.width = width

        self.virt = MetaFormerBlock(width, num_head, *RESCALE_GRAPH, False)
        self.node = GatedLinearBlock(width, num_head, *RESCALE_NODE)
        self.head = GatedLinearBlock(width, num_head, 2, 2, width_out=1)
        print('##params[head]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, virt_res, batch, batch_size):
        x0 = scatter(x, batch, dim=0, dim_size=len(batch_size), reduce='sum')
        x0 = self.virt(virt_res, x0)

        x1 = self.node(x)
        x1 = scatter(x1, batch, dim=0, dim_size=len(batch_size), reduce='mean')

        xx, xn = self.head(x0 + x1, out_norm=True)
        return (xx + 1.0) * 5.5, xn


# GIN: https://openreview.net/forum?id=ryGs6iA5Km
class MetaGIN(nn.Module):
    def __init__(self, depth, num_head, conv_hop, conv_kernel, use_virt, dim_head=None):
        super().__init__()
        self.depth = depth
        dim_head = num_head if dim_head is None else dim_head
        self.width = width = dim_head * num_head
        self.conv_hop = conv_hop
        self.conv_kernel = conv_kernel
        self.use_virt = use_virt
        print('#model:', depth, width, num_head, conv_hop, conv_kernel, use_virt)

        self.atom_encoder = nn.EmbeddingBag(EMBED_ATOM, width, scale_grad_by_freq=True, padding_idx=0)
        self.atom_pos  = GatedLinearBlock(width, num_head, *RESCALE_NODE, width_in=EMBED_POS)
        self.atom_conv = ConvKernel(width, num_head, conv_hop, 1)
        self.atom_main = MetaFormerBlock(width, num_head, *RESCALE_NODE, False, 'main')

        self.conv = nn.ModuleList()
        self.virt = nn.ModuleList()
        self.main = nn.ModuleList()
        for layer in range(depth):
            self.conv.append(ConvKernel(width, num_head, conv_hop, conv_kernel[layer]))
            self.virt.append(VirtKernel(width, num_head) if use_virt else None)
            self.main.append(MetaFormerBlock(width, num_head, *RESCALE_NODE, layer>=depth-1, 'main'))

        self.head = HeadBlock(width, num_head)
        print('#params:', np.sum([np.prod(p.shape) for n, p in self.named_parameters()]))

    def getEdge(self, graph, batch, batch_size):
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

        return edge

    def forward(self, graph):
        x, z = graph['atom'].x, graph['atom'].pos_rw
        batch, batch_size = graph['atom'].batch, graph['atom'].ptr[1:]-graph['atom'].ptr[:-1]
        edge = self.getEdge(graph, batch, batch_size)

        h_in, h_virt, h_out = self.atom_encoder(x), 0, self.atom_pos(z)
        h_out = self.atom_conv(h_in, h_out, edge)
        h_in, h_out = self.atom_main(h_in, h_out), 0
        for layer in range(self.depth):
            if self.conv[layer] is not None:
                h_out = self.conv[layer](h_in, h_out, edge)
            if self.virt[layer] is not None:
                h_out, h_virt = self.virt[layer](h_in, h_out, h_virt, batch, batch_size)
            h_in, h_out = self.main[layer](h_in, h_out), 0
        h_out, embed = self.head(h_in, h_virt, batch, batch_size)

        return h_out

