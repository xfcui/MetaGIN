import os
import os.path as osp
import shutil
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip

import h5py
import torch as pt
import numpy as np
import pandas as pd

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
from rdkit.Chem.Lipinski import RotatableBondSmarts

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from ogb.lsc import PCQM4Mv2Evaluator

from torch_sparse import coalesce, spspmm
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.loader import DataLoader


with h5py.File('data/atom2feat.hdf', 'r') as f:
    ATOM2FEAT = pt.from_numpy(f['atom2feat'][()])
    ATOM2FEAT = ATOM2FEAT - pt.min(ATOM2FEAT, 0)[0]
ATOM_SIZE = [36, 4, 12, 12, 10, 6, 6, 3] + (pt.max(ATOM2FEAT, 0)[0] + 1).tolist()
ATOM_CUMSIZE = pt.from_numpy(np.array([1] + ATOM_SIZE).cumsum())
#0 atomic_num: class36 *
#1 chirality: class4 *
#2 degree: value12
#3 formal_charge: value12[-5]
#4 numH: value10
#5 number_radical_e: value6
#6 hybridization: class6 *
#7 aromatic+ring: class3 *

BOND_SIZE = [5, 6, 2, 2, 17]
BOND_CUMSIZE = pt.from_numpy(np.array([1] + BOND_SIZE).cumsum())
#0 bond_type: class5 *
#1 bond_stereo: class6 *
#2 is_conjugated: bool
#3 rotatable: bool
#4 brics: class17 *


def hetero_transform(graph):
    # input
    size = graph.num_nodes
    head = [graph.edge_index[0] == i for i in range(size)]
    head = [[graph.edge_index[1, i], graph.edge_attr[i]] for i in head]
    tail = [graph.edge_index[1] == i for i in range(size)]
    tail = [[graph.edge_index[0, i], graph.edge_attr[i]] for i in tail]

    # one-hop neighbors
    hop1_attr, hop1_idx = [], []
    for (i0, i1), a0 in zip(graph.edge_index.T.tolist(), graph.edge_attr.tolist()):
        hop1_idx += [[i0, i1]]
        hop1_attr += [a0]
    if len(hop1_attr) > 0:
        hop1_idx = pt.tensor(hop1_idx).T
        hop1_attr = pt.tensor(hop1_attr)
    else:
        hop1_idx = pt.zeros([0, 2]).T
        hop1_attr = pt.zeros([0, 5])

    # two-hop neighbors
    hop2_attr, hop2_idx = [], []
    for i1 in range(size):
        ei0, ea0 = tail[i1]
        ei1, ea1 = head[i1]
        for i0, a0 in zip(ei0.tolist(), ea0.tolist()):
            for i2, a1 in zip(ei1.tolist(), ea1.tolist()):
                if i0 == i2: continue  # loop
                hop2_idx += [[i0, i2]]
                hop2_attr += [[1]]
    if len(hop2_attr) > 0:
        hop2_idx = pt.tensor(hop2_idx).long().T
        hop2_attr = pt.tensor(hop2_attr).int()
        hop2_idx, hop2_attr = coalesce(hop2_idx, hop2_attr, size, size, op='sum')
    else:
        hop2_idx = pt.zeros([0, 2]).T
        hop2_attr = pt.zeros([0, 1])

    # three-hop neighbors
    hop3_attr, hop3_idx = [], []
    for hop1, (i1, i2), a1 in zip(range(len(graph.edge_attr)), graph.edge_index.T.tolist(), graph.edge_attr.tolist()):
        ei0, ea0 = tail[i1]
        ei2, ea2 = head[i2]
        for i0, a0 in zip(ei0.tolist(), ea0.tolist()):
            for i3, a2 in zip(ei2.tolist(), ea2.tolist()):
                if i0 == i2 or i0 == i3 or i1 == i3: continue  # loop
                hop3_idx += [[i0, i3]]
                hop3_attr += [[1]]
    if len(hop3_attr) > 0:
        hop3_idx = pt.tensor(hop3_idx).long().T
        hop3_attr = pt.tensor(hop3_attr).int()
        hop3_idx, hop3_attr = coalesce(hop3_idx, hop3_attr, size, size, op='sum')
    else:
        hop3_idx = pt.zeros([0, 2]).T
        hop3_attr = pt.zeros([0, 1])

    # output
    g = HeteroData()
    g['atom'].x = graph.x.short()
    g['atom'].pos_3d = graph.pos_3d.half()
    g['atom'].pos_rw = graph.pos_rw.half()
    g['atom', 'bond', 'atom'].edge_index = hop1_idx.short()
    g['atom', 'bond', 'atom'].edge_attr = hop1_attr.short()
    g['atom', 'angle', 'atom'].edge_index = hop2_idx.short()
    g['atom', 'angle', 'atom'].edge_attr = hop2_attr.short()
    g['atom', 'torsion', 'atom'].edge_index = hop3_idx.short()
    g['atom', 'torsion', 'atom'].edge_attr = hop3_attr.short()
    g.y = graph.y.half()

    return g


def cast_transform(graph, nohydro=True):
    g = HeteroData()
    g['atom'].x = graph['atom'].x.long()
    g['atom'].x = pt.cat([g['atom'].x, ATOM2FEAT[g['atom'].x[:, 0]]], 1).long() + ATOM_CUMSIZE[:-1]
    g['atom'].pos_3d = graph['atom'].pos_3d.float()
    g['atom'].pos_rw = graph['atom'].pos_rw.float()
    if nohydro:
        # remove hydrogen 1-hop edges
        mask = g['atom'].x[:, 0][graph['bond'].edge_index.long()] > 1
        mask = mask[0] & mask[1]
        g['atom', 'bond', 'atom'].edge_index = graph['bond'].edge_index[:, mask].long()
        g['atom', 'bond', 'atom'].edge_attr = graph['bond'].edge_attr[mask].long() + BOND_CUMSIZE[:-1]
    else:
        g['atom', 'bond', 'atom'].edge_index = graph['bond'].edge_index.long()
        g['atom', 'bond', 'atom'].edge_attr = graph['bond'].edge_attr.long() + BOND_CUMSIZE[:-1]
    if nohydro:
        # remove hydrogen 2-hop edges
        mask = g['atom'].x[:, 0][graph['angle'].edge_index.long()] > 1
        mask = mask[0] & mask[1]
        g['atom', 'angle', 'atom'].edge_index = graph['angle'].edge_index[:, mask].long()
        g['atom', 'angle', 'atom'].edge_attr = graph['angle'].edge_attr[mask].long()
    else:
        g['atom', 'angle', 'atom'].edge_index = graph['angle'].edge_index.long()
        g['atom', 'angle', 'atom'].edge_attr = graph['angle'].edge_attr.long()
    if nohydro:
        # remove hydrogen 3-hop edges
        mask = g['atom'].x[:, 0][graph['torsion'].edge_index.long()] > 1
        mask = mask[0] & mask[1]
        g['atom', 'torsion', 'atom'].edge_index = graph['torsion'].edge_index[:, mask].long()
        g['atom', 'torsion', 'atom'].edge_attr = graph['torsion'].edge_attr[mask].long()
    else:
        g['atom', 'torsion', 'atom'].edge_index = graph['torsion'].edge_index.long()
        g['atom', 'torsion', 'atom'].edge_attr = graph['torsion'].edge_attr.long()
    if nohydro:
        # remove hydrogen nodes
        mask = g['atom'].x[:, 0] > 1
        g['atom'].x = g['atom'].x[mask]
        if len(g['atom'].pos_3d) > 0:
            g['atom'].pos_3d = g['atom'].pos_3d[mask]
        g['atom'].pos_rw = g['atom'].pos_rw[mask]
        mask = pt.cumsum(mask.long(), 0) - 1
        g['bond'].edge_index = mask[g['bond'].edge_index]
        g['angle'].edge_index = mask[g['angle'].edge_index]
        g['torsion'].edge_index = mask[g['torsion'].edge_index]
    g.y = graph.y.float()

    return g


class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, root='data', transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
        '''

        self.original_root = root
        self.folder = osp.join(root, 'pcqm4m-metagin')
        self.version = 1
        
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = pt.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def molecule2graph(self, mol):
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            v = atom_to_feature_vector(atom)
            v = v[:7] + [v[7]+v[8]]
            atom_features_list.append(v)
        x = np.array(atom_features_list, dtype = np.int16)
        num_nodes = len(x)

        # bonds
        num_bond_features = 5  # bond type, bond stereo, is_conjugated, rotatable, BRICS
        rotate = np.zeros([num_nodes, num_nodes], dtype = np.int16)
        for i, j in mol.GetSubstructMatches(RotatableBondSmarts):
            rotate[i, j] = rotate[j, i] = 1
        brics = np.zeros([num_nodes, num_nodes], dtype = np.int16)
        for (i, j), (s, t) in BRICS.FindBRICSBonds(mol):
            try: s, t = int(s), int(t)
            except: s = t = 7  # (s, t) = (7a, 7b)
            brics[i, j] = s
            brics[j, i] = t
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature + [rotate[i, j], brics[i, j]])
                edges_list.append((j, i))
                edge_features_list.append(edge_feature + [rotate[j, i], brics[j, i]])

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int32).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int16)
        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int32)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int16)

        graph = dict()
        graph['num_nodes'] = num_nodes
        graph['node_feat'] = x
        try:
            graph['node_pos3d'] = mol.GetConformer().GetPositions().astype(np.float16)
        except:
            graph['node_pos3d'] = np.zeros([0, 3], dtype=np.float16)
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        return graph 

    def process(self):
        split_dict = self.get_idx_split()
        size_dict = sum([len(v) for v in split_dict.values()])
        print('#dict:', size_dict, [[min(v).item(), max(v).item()] for v in split_dict.values()])

        print('#loading CSV file ...')
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']
        print('#loaded:', len(smiles_list), len(homolumogap_list), size_dict)
        assert len(smiles_list) == size_dict
        assert len(homolumogap_list) == size_dict

        try:
            print('#loading SDF file ...')
            data_sdf = Chem.SDMolSupplier(osp.join(self.raw_dir, 'embed3d-train.sdf'), removeHs=False)  # hydrogen
            embed_list = []
            for d in tqdm(data_sdf, total=len(split_dict['train'])):
                embed_list.append(d)
            print('#loaded:', len(embed_list), len(split_dict['train']))
            assert len(embed_list) == len(split_dict['train'])
            embed_list += [None] * (size_dict - len(embed_list))
        except:
            embed_list = [None] * size_dict
        print('#padded:', len(embed_list), size_dict)
        assert len(embed_list) == size_dict

        print('#converting molecules into graphs...')
        data_list = []
        for smiles, embed, homolumogap in tqdm(zip(smiles_list, embed_list, homolumogap_list), total=size_dict):
            if embed is None:
                molecule = Chem.MolFromSmiles(smiles)
                molecule = Chem.AddHs(molecule)  # hydrogen
                graph = self.molecule2graph(molecule)
            else:
                graph = self.molecule2graph(embed)
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data = Data()
            data.edge_index = pt.from_numpy(graph['edge_index']).long()
            data.edge_attr = pt.from_numpy(graph['edge_feat']).short()
            data.x = pt.from_numpy(graph['node_feat']).short()
            data.pos_3d = pt.from_numpy(graph['node_pos3d']).half()
            addPosRW(data)
            data.y = pt.Tensor([homolumogap]).half()
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        print('#converted:', len(data_list), size_dict)
        assert len(data_list) == size_dict

        # double-check prediction target
        assert(all([not pt.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert(all([not pt.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert(all([pt.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        assert(all([pt.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        data, slices = self.collate(data_list)

        print('Saving...')
        pt.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(pt.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict

addPosRW = AddRandomWalkPE(16, 'pos_rw')
dataset  = PygPCQM4Mv2Dataset(root='data', pre_transform=hetero_transform, transform=cast_transform)
dataidx  = dataset.get_idx_split()
dataeval = PCQM4Mv2Evaluator()


if __name__=="__main__":
    print('#atom_cumsize:', *ATOM_CUMSIZE.tolist())
    print('#bond_cumsize:', *BOND_CUMSIZE.tolist())
    print()

    g0 = dataset[0]
    print('#data:', g0)
    print('#atom:', g0['atom'].x[:12])
    for k in ['atom']:
        print('#dtype:', k, g0[k].x.dtype, g0[k].pos_3d.dtype, g0[k].pos_rw.dtype)
    print('#bond:', g0['bond'].edge_attr[:12])
    for k in ['bond', 'angle', 'torsion']:
        print('#dtype:', k, g0[k].edge_index.dtype, g0[k].edge_attr.dtype)
    print('#dtype:', 'y', g0.y.dtype)
    print()

    loader = DataLoader(dataset, batch_size=12, shuffle=True, drop_last=True)
    for b in loader:
        print('#batch:')
        print(b)
        print()
        break

    print('#done!!!')

