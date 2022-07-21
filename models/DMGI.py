import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from embedder import embedder
from layers import GCN, Discriminator, Attention
import numpy as np
np.random.seed(0)
from evaluate import evaluate
from models import LogReg
import pickle as pkl

import pandas as pd

class DMGI(embedder):
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
        embedder.__init__(self,
                        embedder_name,
                        dataset,
                        metapaths,
                        nb_epochs,
                        hid_units,
                        lr,
                        l2_coef,
                        drop_prob,
                        reg_coef,
                        sup_coef,
                        sc,
                        margin,
                        gpu_num,
                        patience,
                        nheads,
                        activation,
                        isSemi,
                        isBias,
                        isAttn,
                        batch_size,
                        sparse)

    def training(self):
        features = [feature.to(self.device) for feature in self.features]
        adj = [adj_.to(self.device) for adj_ in self.adj]

        model = modeler(hid_units = self.hid_units,
                        nb_graphs = self.nb_graphs,
                        nb_nodes = self.nb_nodes,
                        ft_size = self.ft_size,
                        activation = self.activation,
                        drop_prob = self.drop_prob,
                        isBias = self.isBias,
                        readout_func = self.readout_func,
                        isAttn = self.isAttn,
                        isSemi = self.isSemi,
                        nheads = self.nheads,
                        nb_classes = self.nb_classes,
                        device = self.device,
                        readout_act_func = self.readout_act_func).to(self.device)

        optimiser = torch.optim.Adam(model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.l2_coef)
        cnt_wait = 0; best = 1e9
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()

        for epoch in range(self.nb_epochs):
            xent_loss = None
            model.train()
            optimiser.zero_grad()
            idx = np.random.permutation(self.nb_nodes)

            shuf = [feature[:, idx, :] for feature in features]
            shuf = [shuf_ft.to(self.device) for shuf_ft in shuf]

            lbl_1 = torch.ones(self.batch_size, self.nb_nodes)
            lbl_2 = torch.zeros(self.batch_size, self.nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)

            result = model(features, adj, shuf, self.sparse, None, None, None)
            logits = result['logits']

            for view_idx, logit in enumerate(logits):
                if xent_loss is None:
                    xent_loss = b_xent(logit, lbl)
                else:
                    xent_loss += b_xent(logit, lbl)

            loss = xent_loss

            reg_loss = result['reg_loss']
            loss += self.reg_coef * reg_loss

            if self.isSemi:
                sup = result['semi']
                semi_loss = xent(sup[self.idx_train], self.train_lbls)
                loss += self.sup_coef * semi_loss

            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), 'saved_model/best_{}_{}_{}.pkl'.format(self.dataset, self.embedder_name, self.metapaths))
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                break

            loss.backward()
            optimiser.step()


        model.load_state_dict(torch.load('saved_model/best_{}_{}_{}.pkl'.format(self.dataset, self.embedder_name, self.metapaths)))

        # Evaluation
        model.eval()
        evaluate(model.H.data.detach(), self.idx_train, self.idx_val, self.idx_test, self.labels, self.device)

        embeds = model.H.data.detach().cpu().numpy().squeeze()
        np.save('output/embeds_{}_{}_{}.npy'.format(self.dataset, self.embedder_name, self.metapaths), embeds)

        df_embeds = pd.DataFrame(embeds)
        df_embeds.to_csv('output/embeds_{}_{}_{}.csv'.format(self.dataset, self.embedder_name, self.metapaths), index=False)


class modeler(nn.Module):
    def __init__(self,
                 hid_units,
                 nb_graphs,
                 nb_nodes,
                 ft_size,
                 activation,
                 drop_prob,
                 isBias,
                 readout_func,
                 isAttn,
                 isSemi,
                 nheads,
                 nb_classes,
                 device,
                 readout_act_func
                 ):
        super(modeler, self).__init__()
        self.gcn = nn.ModuleList([GCN(ft_size, hid_units, activation, drop_prob, isBias) for _ in range(nb_graphs)])

        self.disc = Discriminator(hid_units)
        self.H = nn.Parameter(torch.FloatTensor(1, nb_nodes, hid_units))
        self.readout_func = readout_func
        if isAttn:
            self.attn = nn.ModuleList([Attention(hid_units, nb_graphs, nb_nodes) for _ in range(nheads)])

        if isSemi:
            self.logistic = LogReg(hid_units, nb_classes).to(device)

        self.init_weight()

        self.nb_graphs = nb_graphs
        self.readout_act_func = readout_act_func
        self.isAttn = isAttn
        self.isSemi = isSemi
        self.nheads = nheads

    def init_weight(self):
        nn.init.xavier_normal_(self.H)

    def forward(self, feature, adj, shuf, sparse, msk, samp_bias1, samp_bias2):
        h_1_all = []; h_2_all = []; c_all = []; logits = []
        result = {}

        for i in range(self.nb_graphs):
            h_1 = self.gcn[i](feature[i], adj[i], sparse)

            # how to readout positive summary vector
            c = self.readout_func(h_1)
            c = self.readout_act_func(c)  # equation 9
            h_2 = self.gcn[i](shuf[i], adj[i], sparse)
            logit = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

            h_1_all.append(h_1)
            h_2_all.append(h_2)
            c_all.append(c)
            logits.append(logit)

        result['logits'] = logits

        # Attention or not
        if self.isAttn:
            h_1_all_lst = []; h_2_all_lst = []; c_all_lst = []

            for h_idx in range(self.nheads):
                h_1_all_, h_2_all_, c_all_ = self.attn[h_idx](h_1_all, h_2_all, c_all)
                h_1_all_lst.append(h_1_all_); h_2_all_lst.append(h_2_all_); c_all_lst.append(c_all_)

            h_1_all = torch.mean(torch.cat(h_1_all_lst, 0), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all_lst, 0), 0).unsqueeze(0)

        else:
            h_1_all = torch.mean(torch.cat(h_1_all), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all), 0).unsqueeze(0)


        # consensus regularizer
        pos_reg_loss = ((self.H - h_1_all) ** 2).sum()
        neg_reg_loss = ((self.H - h_2_all) ** 2).sum()
        reg_loss = pos_reg_loss - neg_reg_loss
        result['reg_loss'] = reg_loss

        # semi-supervised module
        if self.isSemi:
            semi = self.logistic(self.H).squeeze(0)
            result['semi'] = semi

        return result