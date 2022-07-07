import numpy as np
np.random.seed(0)
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import argparse
import datetime

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='DMGI')

    parser.add_argument('--embedder', nargs='?', default='DMGI')
    parser.add_argument('--dataset', nargs='?', default='imdb')
    parser.add_argument('--metapaths', nargs='?', default='MAM,MDM')

    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--hid_units', type=int, default=64)
    parser.add_argument('--lr', type = float, default = 0.0005)
    parser.add_argument('--l2_coef', type=float, default=0.0001)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--reg_coef', type=float, default=0.001)
    parser.add_argument('--sup_coef', type=float, default=0.1)
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--nheads', type=int, default=1)
    parser.add_argument('--activation', nargs='?', default='relu')
    parser.add_argument('--isSemi', action='store_true', default=False)
    parser.add_argument('--isBias', action='store_true', default=False)
    parser.add_argument('--isAttn', action='store_true', default=False)

    return parser.parse_known_args()

def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    print(args_names)
    print(args_vals)

def main():
    # args, unknown = parse_args()

    embedder_name = 'DMGI'
    dataset = 'tp1'
    metapaths = 'a_m_c,a_m_d'
    nb_epochs = 1
    hid_units = 3
    lr = 0.0005
    l2_coef = 0.0001
    drop_prob = 0.5
    reg_coef = 0.001
    sup_coef = 0.1
    sc = 3.0
    margin = 0.1
    gpu_num = 0
    patience = 20
    nheads = 1
    activation = 'relu'
    isSemi = False
    isBias = False
    isAttn = True

    batch_size = 1
    sparse = True

    start = datetime.datetime.now()
    print("Start: ", start)

    if embedder_name == 'DMGI':
        from models import DMGI
        embedder = DMGI(embedder_name=embedder_name,
                        dataset=dataset,
                        metapaths=metapaths,
                        nb_epochs=nb_epochs,
                        hid_units=hid_units,
                        lr=lr,
                        l2_coef=l2_coef,
                        drop_prob=drop_prob,
                        reg_coef=reg_coef,
                        sup_coef=sup_coef,
                        sc=sc,
                        margin=margin,
                        gpu_num=gpu_num,
                        patience=patience,
                        nheads=nheads,
                        activation=activation,
                        isSemi=isSemi,
                        isBias=isBias,
                        isAttn=isAttn,
                        batch_size=batch_size,
                        sparse=sparse
                        )
    elif embedder_name == 'DGI':
        from models import DGI
        embedder = DGI(embedder_name=embedder_name,
                        dataset=dataset,
                        metapaths=metapaths,
                        nb_epochs=nb_epochs,
                        hid_units=hid_units,
                        lr=lr,
                        l2_coef=l2_coef,
                        drop_prob=drop_prob,
                        reg_coef=reg_coef,
                        sup_coef=sup_coef,
                        sc=sc,
                        margin=margin,
                        gpu_num=gpu_num,
                        patience=patience,
                        nheads=nheads,
                        activation=activation,
                        isSemi=isSemi,
                        isBias=isBias,
                        isAttn=isAttn,
                        batch_size=batch_size,
                        sparse=sparse
                        )

    embedder.training()

    end = datetime.datetime.now()
    print("Done: ", end)
    print(f"Total: {(end-start).total_seconds()} sec")

if __name__ == '__main__':
    main()
