import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from .fc_layer import FCLayer

import pdb

class LowLankBilinearPooling(nn.Module):
    def __init__(self,
            v_dim,
            q_dim,
            h_dim,
            h_out,
            activation='ReLU',
            dropout=[0.2, 0.5],
        ):
        super(LowLankBilinearPooling, self).__init__()
        # self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCLayer(v_dim, h_dim, activation=activation, dropout=dropout[0])
        self.q_net = FCLayer(q_dim, h_dim, activation=activation, dropout=dropout[0])

        if h_out is not None:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())

        self.dropout = nn.Dropout(dropout[1]) # attention

    def forward(self, v, q, w):
        if w is None:
            if v.device.type == 'cuda':
                device = v.device.type + ':' + str(v.device.index)
                self.h_mat  = self.h_mat.to(device)
                self.h_bias = self.h_bias.to(device)

            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bnk,bmk->bhnm', (self.h_mat, v_, q_)) + self.h_bias
            return logits # b x h_out x v x q
        else:
            return self.forward_with_weights(v, q, w)


    def forward_with_weights(self, v, q, w):
        # この関数は、BAN論文の(5)に対応
        v_ = self.v_net(v) # b x v x d
        q_ = self.q_net(q) # b x q x d
        logits = torch.einsum('bnk,bvm,bmk->bkm', (v_, w, q_))

        return logits


class BilinearAttentionMap(nn.Module):
    def __init__(
            self,
            v_dim,
            q_dim,
            h_dim,
            glimpse=2,
            soft_attention=True,
            dropout=[0.2, 0.5]
        ):
        super(BilinearAttentionMap, self).__init__()

        self.glimpse = glimpse

        if soft_attention:
            self.logits = weight_norm(LowLankBilinearPooling(v_dim, q_dim, h_dim, glimpse, dropout=dropout),
                                  name='h_mat', dim=None)
        else :
            self.logits = LowLankBilinearPooling(v_dim, q_dim, h_dim, glimpse, dropout=dropout)

        pass

    def forward(self, v, q):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q, None)

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits

if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    llbp = LowLankBilinearPooling(10, 20, 30, 4)

    llbp.to(device)

    # llbp = LowLankBilinearPooling(10, 20, 30, 4, device='cpu')

    v = torch.rand([5, 8, 10]).to(device)
    q = torch.rand([5, 7, 20]).to(device)

    logits = llbp(v, q, None)

    print(logits)
    print(logits.shape) # logits.shape: (5, 4, 8, 7)

    biAttnMap = BilinearAttentionMap(10, 20, 30, 4)

    biAttnMap.to(device)

    attn, logits = biAttnMap(v, q)

    print(attn)
    print(logits)
    print(attn.shape) # atten.shape

    logits = llbp(v, q, attn[:,0,:,:])
    print(logits)
    print(logits.shape)