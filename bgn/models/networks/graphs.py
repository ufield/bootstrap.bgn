import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from .attention import LowLankBilinearPooling, BilinearAttentionMap
from .fc_layer import FCLayer

import pdb

class ImageGraph(nn.Module):

    def __init__(
            self,
            v_dim,
            q_dim,
            k_dim,
            i_glimpse=2
        ):
        super(ImageGraph, self).__init__()
        self.v_att = BilinearAttentionMap(v_dim, q_dim, k_dim, glimpse=i_glimpse)
        self.i_glimpse = i_glimpse

        self.b_net = []
        self.q_prj = []
        for i in range(i_glimpse):
            self.b_net.append(LowLankBilinearPooling(v_dim, q_dim, k_dim, h_out=None))
            self.q_prj.append(FCLayer(k_dim, q_dim))
        self.b_net = nn.ModuleList(self.b_net)
        self.q_prj = nn.ModuleList(self.q_prj)


    def forward(self, v, q):
        attn, _ = self.v_att(v, q)
        h = q

        for g in range(self.i_glimpse):
            h_tmp = self.b_net[g].forward_with_weights(v, h, attn[:,g,:,:]) #
            h_tmp = h_tmp.transpose(1,2)

            h = self.q_prj[g](h_tmp) + h

        return h


class QuestionGraph(nn.Module):
    def __init__(
            self,
            q_dim,
            k_dim,
            q_glimpse=2
        ):
        super(QuestionGraph, self).__init__()
        self.att = BilinearAttentionMap(q_dim, q_dim, k_dim, glimpse=q_glimpse)
        self.q_glimpse = q_glimpse

        self.b_net = []
        self.q_prj = []
        for i in range(q_glimpse):
            self.b_net.append(LowLankBilinearPooling(q_dim, q_dim, k_dim, h_out=None))
            self.q_prj.append(FCLayer(k_dim, q_dim))
        self.b_net = nn.ModuleList(self.b_net)
        self.q_prj = nn.ModuleList(self.q_prj)

    def forward(self, h):
        attn, _ = self.att(h, h)
        o = h

        for g in range(self.q_glimpse):
            o_tmp = self.b_net[g].forward_with_weights(o, o, attn[:,g,:,:]) #
            o_tmp = o_tmp.transpose(1,2)

            o = self.q_prj[g](o_tmp) + o
        return o


if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ig = ImageGraph(10, 20, 30, 4)
    ig.to(device)

    v = torch.rand([5, 3, 10]).to(device)
    q = torch.rand([5, 4, 20]).to(device)

    h = ig(v, q)

    print(h.shape) # shape: (5, 4, 20)

    qg = QuestionGraph(20, 30, 4)
    qg.to(device)

    o = qg(h)

    # print(o)
    print(o.shape) # shape: (5, 4, 20)
