import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.parameter as nnp
import torch.nn.functional as nnf

from torch_scatter import scatter
from torch_geometric.utils import degree
from torch.distributions.log_normal import LogNormal

print('#faiss:', end=' ')
import faiss
import faiss.contrib.torch_utils
print(faiss.__version__)


DROPOUT    = 0.1
EMBED_KNN  = 10          # 50% 8; 75% 10; 90% 12; 95% 13; 99% 14; max 19
EMBED_POS  = 16          # random walk positional embedding
EMBED_ATOM = 138         # zero for masked
EMBED_BOND = [33, 3, 4]  # zero for masked
EMBED_DIST = 8           # zero for self-loop
RESCALE_GRAPH, RESCALE_NODE, RESCALE_EDGE = (2, 4), (1, 3), (1, 2)


# ReZero: https://arxiv.org/abs/2003.04887
# LayerScale: https://arxiv.org/abs/2103.17239
class ScaleLayer(nn.Module):
    def __init__(self, width, scale_init):
        super().__init__()
        self.scale = nnp.Parameter(pt.zeros(width) + np.log(scale_init))

    def clamp_(self):
        self.scale.clamp_(np.log(1e-4), 0)

    def forward(self, x):
        return pt.exp(self.scale) * x

# PNA: https://arxiv.org/abs/2004.05718
# Graphormer: https://arxiv.org/abs/2106.05234
class DegreeLayer(nn.Module):
    def __init__(self, width, degree_init=-1e-2):
        super().__init__()
        self.degree = nnp.Parameter(pt.zeros(width) + degree_init)

    def clamp_(self):
        self.degree.clamp_(-1, 0)

    def forward(self, x, deg):
        return pt.pow(deg.unsqueeze(-1), self.degree) * x


# GLU: https://arxiv.org/abs/1612.08083
# GLU-variants: https://arxiv.org/abs/2002.05202
class GatedLinearBlock(nn.Module):
    def __init__(self, width, num_head, resca_norm=1, resca_act=1, skip_pre=False, width_in=None, width_out=None):
        super().__init__()
        assert width >= 256
        self.width = width
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

# MetaFormer: https://arxiv.org/abs/2210.13452
class MetaFormerBlock(nn.Module):
    def __init__(self, width, num_head, resca_norm=1, resca_act=1, use_residual=True, name=None):
        super().__init__()
        self.width = width
        self.nhead = num_head * resca_norm
        self.dhead = width * resca_act // self.nhead

        self.sca_pre  = ScaleLayer(width, 1)
        self.sca_post = ScaleLayer(width, 1) if use_residual else None
        self.ffn      = GatedLinearBlock(width, num_head, resca_norm, resca_act)
        if name is not None:
            print('##params[%s]:' % name, np.sum([np.prod(p.shape) for p in self.parameters()]), use_residual)

    def clamp_(self):
        self.sca_pre.clamp_()
        if self.sca_post is not None: self.sca_post.clamp_()

    def forward(self, x, res):
        xx = self.sca_pre(x) + res
        if self.sca_post is None:
            xx = self.ffn(xx)
        else:
            xx = self.sca_post(xx) + self.ffn(xx)
        return xx


# VoVNet: https://arxiv.org/abs/1904.09730
# GNN-AK: https://openreview.net/forum?id=Mspk_WYKoEH
class ConvBlock(nn.Module):
    def __init__(self, width, num_head, bond_size):
        super().__init__()
        self.width = width
        width_norm = width * RESCALE_EDGE[0]

        self.src = nn.Conv1d(width, width_norm, 1, bias=False)
        self.tgt = nn.Conv1d(width, width_norm, 1, bias=False)
        self.conv_encoder = nn.EmbeddingBag(bond_size, width_norm, scale_grad_by_freq=True, padding_idx=0)
        self.ffn = GatedLinearBlock(width, num_head, *RESCALE_EDGE, skip_pre=True)
        self.deg = DegreeLayer(width)

    def clamp_(self):
        self.deg.clamp_()

    def forward(self, x, deg, edge_idx, edge_attr):
        xx = self.src(x.unsqueeze(-1))[edge_idx[0]] + self.tgt(x.unsqueeze(-1))[edge_idx[1]]
        xx = self.ffn(xx.squeeze(-1), self.conv_encoder(edge_attr))
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
        self.mix = nn.ModuleList()
        for k in range(kernel):
            for h in range(hop):
                self.msg.append(ConvBlock(width, num_head, EMBED_BOND[h]))
            if k < kernel-1:
                self.mix.append(MetaFormerBlock(width, num_head, *RESCALE_NODE))
        print('##params[conv]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def clamp_(self):
        for msg in self.msg: msg.clamp_()
        for mix in self.mix: mix.clamp_()

    def forward(self, x, x_res, edge):
        x_kernel = x_hop = x; x_out = 0
        for k in range(self.kernel):
            for h in range(self.hop):
                edge_index, edge_attr, node_deg = edge[h]
                x_hop = self.msg[k*self.hop+h](x_hop, node_deg, edge_index, edge_attr)
                x_out = x_out + x_hop
            if k < len(self.mix):
                x_kernel = x_hop = self.mix[k](x_kernel, x_out); x_out = 0
        return x_out + x_res


# Vision GNN: https://arxiv.org/abs/2206.00272
# Transformer-M: https://arxiv.org/abs/2210.01765
width_ext, infinity_ext = 10, 100
knn_ext = pt.arange(2**width_ext).reshape(-1, 1).expand(-1, width_ext)
knn_ext = knn_ext // (2**pt.arange(width_ext)) % 2 * infinity_ext
knn_ext = knn_ext.cuda()
class KnnKernel(nn.Module):
    def __init__(self, width, num_head, knn):
        super().__init__()
        self.width = width
        self.knn = knn

        # KNN graph construction
        dev = faiss.StandardGpuResources()
        self.index = faiss.GpuIndexFlatIP(dev, width+width_ext)
        self.pdist = nn.PairwiseDistance(p=2)

        resca_norm, resca_act = RESCALE_EDGE
        # random walk positional embeding
        self.perw_ffn = GatedLinearBlock(width, num_head, resca_norm, resca_act, width_in=EMBED_POS*2)
        # distance embeding for training only
        dist_nhead = width // 2
        dmin, dmax = np.log(0.5), np.log(5.0); drange = dmax - dmin
        self.dist_mean  = nnp.Parameter(pt.rand(dist_nhead) * drange + dmin)
        self.dist_std   = nnp.Parameter(pt.rand(dist_nhead) * drange/5 + 0.1/5)
        self.dist_distr = LogNormal(self.dist_mean, self.dist_std)
        self.dist_ffn   = GatedLinearBlock(width, num_head, resca_norm, resca_act, width_in=dist_nhead)
        # distance prediction for training only
        self.pred_ffn = GatedLinearBlock(width, num_head, resca_norm, resca_act, width_out=1)

        self.src = nn.Conv1d(width, width*resca_norm, 1, bias=False)
        self.tgt = nn.Conv1d(width, width*resca_norm, 1, bias=False)
        self.ffn = GatedLinearBlock(width, num_head, resca_norm, resca_act, skip_pre=True)
        self.deg = DegreeLayer(width)
        print('##params[knn]: ', np.sum([np.prod(p.shape) for p in self.parameters()]), knn)

    def clamp_(self):
        self.dist_mean.clamp_(np.log(0.5), None)
        self.dist_std.clamp_(0.1/5, None)
        self.deg.clamp_()

    def forward(self, x, x_res, z_rw, z_3d, batch, batch_size, eps=1e-4):
        assert len(batch_size) <= 2**width_ext
        x0 = self.src(x.unsqueeze(-1)).squeeze(-1)
        x1 = self.tgt(x.unsqueeze(-1)).squeeze(-1)

        with pt.no_grad():
            self.index.add(pt.cat([x0, knn_ext[batch]], -1))
            d, knn = self.index.search(pt.cat([x1, knn_ext[batch]], -1), self.knn)
            self.index.reset()
            knn = pt.cat([knn.reshape(1, -1), pt.arange(len(x)).reshape(-1, 1).expand(-1, self.knn).reshape(1, -1).cuda()], 0)
            knn = knn[:, batch[knn[0]] != batch[knn[1]]]; knn = knn[:, knn[0] != knn[1]]
            deg = degree(knn[1], len(x)).float(); deg.clamp_(1, None)

        bias, dist_true = self.perw_ffn(pt.cat([z_rw[knn[0]], z_rw[knn[1]]], -1)), None
        if self.training:
            dist_true = self.pdist(z_3d[knn[0]], z_3d[knn[1]]).unsqueeze(-1)
            dist_bias = dist_true * pt.exp(pt.randn_like(dist_true)/5 * 0.2); dist_bias.clamp_(eps, None)
            dist_bias = self.dist_distr.log_prob(dist_bias); dist_bias.clamp_(None, 0)
            dist_bias = self.dist_ffn(dist_bias.exp())
            mask = pt.rand(len(batch_size)).cuda()[batch[knn[0]]].unsqueeze(-1)
            bias, dist_bias, mask = dist_bias * (mask>0.2).float() + bias * (mask<0.4).float(), None, None

        msg, x0, x1 = x0[knn[0]] + x1[knn[1]], None, None
        msg, bias = self.ffn(msg, bias), None
        dist_pred = self.pred_ffn(msg) if self.training else None
        xx, msg = scatter(msg, knn[1], dim=0, dim_size=len(x), reduce='sum'), None
        xx = self.deg(xx, deg)
        return xx + x_res, dist_pred, dist_true


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
    def __init__(self, width, num_head):
        super().__init__()
        self.width = width

        self.virt = MetaFormerBlock(width, num_head, *RESCALE_GRAPH, False)
        self.node = GatedLinearBlock(width, num_head, *RESCALE_NODE)
        self.head = GatedLinearBlock(width, num_head, 2, 2, width_out=1)
        print('##params[head]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def clamp_(self):
        self.virt.clamp_()

    def forward(self, x, virt_res, batch, batch_size):
        x0 = scatter(x, batch, dim=0, dim_size=len(batch_size), reduce='sum')
        x0 = self.virt(virt_res, x0)
        x1 = self.node(x)
        x1 = scatter(x1, batch, dim=0, dim_size=len(batch_size), reduce='mean')
        xx, xn = self.head(x0 + x1, out_norm=True)
        return (xx + 1.0) * 5.5, xn


# GIN: https://openreview.net/forum?id=ryGs6iA5Km
# Graph PE: https://arxiv.org/abs/2110.07875
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
            if layer < depth-1:
                self.conv.append(ConvKernel(width, num_head, conv_hop, conv_kernel[layer]))
            else:
                self.conv.append(KnnKernel(width, num_head, EMBED_KNN))
            self.virt.append(VirtKernel(width, num_head) if use_virt else None)
            self.main.append(MetaFormerBlock(width, num_head, *RESCALE_NODE, layer>=depth-1, 'main'))

        self.head = HeadBlock(width, num_head)
        print('#params:', np.sum([np.prod(p.shape) for n, p in self.named_parameters()]))

    def clamp_(self):
        self.atom_conv.clamp_()
        self.atom_main.clamp_()
        for conv in self.conv: conv.clamp_()
        for main in self.main: main.clamp_()
        self.head.clamp_()

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
        x, z_3d, z_rw = graph['atom'].x, graph['atom'].pos_3d, graph['atom'].pos_rw
        batch, batch_size = graph['atom'].batch, graph['atom'].ptr[1:]-graph['atom'].ptr[:-1]
        edge = self.getEdge(graph, batch, batch_size)

        h_in, h_virt, h_out = self.atom_encoder(x), 0, self.atom_pos(z_rw)
        h_out = self.atom_conv(h_in, h_out, edge)
        h_in, h_out = self.atom_main(h_in, h_out), 0
        for layer in range(self.depth):
            if self.conv[layer] is not None:
                if isinstance(self.conv[layer], ConvKernel):
                    h_out = self.conv[layer](h_in, h_out, edge)
                else:
                    h_out, d_pred, d_true = self.conv[layer](h_in, h_out, z_rw, z_3d, batch, batch_size)
            if self.virt[layer] is not None:
                h_out, h_virt = self.virt[layer](h_in, h_out, h_virt, batch, batch_size)
            h_in, h_out = self.main[layer](h_in, h_out), 0
        h_out, embed = self.head(h_in, h_virt, batch, batch_size)

        if self.training:
            return h_out, d_pred, d_true
        else:
            return h_out

