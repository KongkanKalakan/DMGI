import time
import numpy as np
import torch
from utils import process
import torch.nn as nn
from layers import AvgReadout


class embedder:
    def __init__(self,
                 embedder_name: str,
                 dataset: str,
                 metapaths: str,
                 nb_epochs: int,
                 hid_units: int,
                 lr: float,
                 l2_coef: float,
                 drop_prob: float,
                 reg_coef: float,
                 sup_coef: float,
                 sc: float,
                 margin: float,
                 gpu_num: str,
                 patience: int,
                 nheads: int,
                 activation: str,
                 isSemi: bool,
                 isBias: bool,
                 isAttn: bool,
                 batch_size: int = 1,
                 sparse: bool = True
                 ):

        self.embedder_name = embedder_name
        self.dataset = dataset
        self.metapaths = metapaths
        self.nb_epochs = nb_epochs
        self.hid_units = hid_units
        self.lr = lr
        self.l2_coef = l2_coef
        self.drop_prob = drop_prob
        self.reg_coef = reg_coef
        self.sup_coef = sup_coef
        self.sc = sc
        self.margin = margin
        self.gpu_num = gpu_num
        self.patience = patience
        self.nheads = nheads
        self.activation = activation
        self.isSemi = isSemi
        self.isBias = isBias
        self.isAttn = isAttn
        self.batch_size = batch_size
        self.sparse = sparse


        self.metapaths_list = metapaths.split(",")
        self.gpu_num_ = gpu_num

        if self.gpu_num_ == "cpu":
            self.device = "cpu"
        else:
            self.device = torch.device("cuda:" + str(self.gpu_num_) if torch.cuda.is_available() else "cpu")

        adj, features, labels, idx_train, idx_val, idx_test = process.load_data_dblp(dataset = dataset,
                                                                                     metapaths = self.metapaths_list,
                                                                                     sc = sc)
        features = [process.preprocess_features(feature) for feature in features]

        self.nb_nodes = features[0].shape[0]
        self.ft_size = features[0].shape[1]
        self.nb_classes = labels.shape[1]
        self.nb_graphs = len(adj)

        adj = [process.normalize_adj(adj_) for adj_ in adj]
        self.adj = [process.sparse_mx_to_torch_sparse_tensor(adj_) for adj_ in adj]

        self.features = [torch.FloatTensor(feature[np.newaxis]) for feature in features]

        self.labels = torch.FloatTensor(labels[np.newaxis]).to(self.device)
        self.idx_train = torch.LongTensor(idx_train).to(self.device)
        self.idx_val = torch.LongTensor(idx_val).to(self.device)
        self.idx_test = torch.LongTensor(idx_test).to(self.device)

        self.train_lbls = torch.argmax(self.labels[0, self.idx_train], dim=1)
        self.val_lbls = torch.argmax(self.labels[0, self.idx_val], dim=1)
        self.test_lbls = torch.argmax(self.labels[0, self.idx_test], dim=1)

        # How to aggregate
        self.readout_func = AvgReadout()

        # Summary aggregation
        self.readout_act_func = nn.Sigmoid()

    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s
