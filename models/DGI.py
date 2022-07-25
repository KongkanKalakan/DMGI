# Code based on https://github.com/PetarV-/DGI/blob/master/models/dgi.py
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from embedder import embedder
from layers import GCN, Discriminator
import numpy as np
np.random.seed(0)
from evaluate import evaluate

import pandas as pd
from utils.visualization import draw_loss

class DGI(embedder):
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
                 sparse: bool = True):
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
        features_lst = [feature.to(self.device) for feature in self.features]
        adj_lst = [adj_.to(self.device) for adj_ in self.adj]

        final_embeds = []
        for m_idx, (features, adj) in enumerate(zip(features_lst, adj_lst)):
            metapath = self.metapaths_list[m_idx]
            print("- Training on {}".format(metapath))

            model = modeler(hid_units = self.hid_units,
                            ft_size = self.ft_size,
                            activation = self.activation,
                            drop_prob = self.drop_prob,
                            isBias = self.isBias,
                            readout_func = self.readout_func,
                            readout_act_func = self.readout_act_func,
                            ).to(self.device)

            optimiser = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2_coef)
            cnt_wait = 0; best = 1e9
            b_xent = nn.BCEWithLogitsLoss()

            loss_values = []

            for epoch in range(self.nb_epochs):
                model.train()
                optimiser.zero_grad()

                idx = np.random.permutation(self.nb_nodes)
                shuf_fts = features[:, idx, :].to(self.device)

                lbl_1 = torch.ones(self.batch_size, self.nb_nodes)
                lbl_2 = torch.zeros(self.batch_size, self.nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1)

                lbl = lbl.to(self.device)

                logits = model(features, shuf_fts, adj, self.sparse, None, None, None)

                loss = b_xent(logits, lbl)


                if loss < best:
                    best = loss
                    cnt_wait = 0
                    torch.save(model.state_dict(), 'saved_model/best_{}_{}_{}.pkl'.format(self.dataset, self.embedder_name, metapath))
                else:
                    cnt_wait += 1

                if cnt_wait == self.patience:
                    break

                loss_values.append(loss.detach().cpu().numpy())

                loss.backward()
                optimiser.step()

            loss_values = np.array(loss_values).tolist()
            draw_loss(loss_values, save_filename='output/losses_{}_{}_{}.jpg'.format(self.dataset, self.embedder_name, metapath))

            model.load_state_dict(torch.load('saved_model/best_{}_{}_{}.pkl'.format(self.dataset, self.embedder_name, metapath)))

            # Evaluation
            embeds, _ = model.embed(features, adj, self.sparse)
            # print("embeds shape: ", embeds.shape)
            evaluate(embeds, self.idx_train, self.idx_val, self.idx_test, self.labels, self.device)
            final_embeds.append(embeds)

        embeds = torch.mean(torch.cat(final_embeds), 0).unsqueeze(0)
        print("- Integrated")
        evaluate(embeds, self.idx_train, self.idx_val, self.idx_test, self.labels, self.device)

        embeds = embeds.cpu().numpy().squeeze()
        np.save('output/embeds_{}_{}_{}.npy'.format(self.dataset, self.embedder_name, self.metapaths), embeds)

        df_embeds = pd.DataFrame(embeds)
        df_embeds.to_csv('output/embeds_{}_{}_{}.csv'.format(self.dataset, self.embedder_name, self.metapaths), index=False)

class modeler(nn.Module):
    def __init__(self,
                 hid_units,
                 ft_size,
                 activation,
                 drop_prob,
                 isBias,
                 readout_func,
                 readout_act_func
                 ):
        super(modeler, self).__init__()
        self.gcn = GCN(ft_size, hid_units, activation, drop_prob, isBias)

        # one discriminator
        self.disc = Discriminator(hid_units)
        self.readout_func = readout_func
        self.readout_act_func = readout_act_func

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        c = self.readout_func(h_1)  # equation 9
        c = self.readout_act_func(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse):
        h_1 = self.gcn(seq, adj, sparse)

        c = self.readout_func(h_1)  # positive summary vector
        c = self.readout_act_func(c)  # equation 9

        return h_1.detach(), c.detach()
