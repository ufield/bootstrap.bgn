from copy import deepcopy
import itertools
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
import block
from block.models.networks.vqa_net import factory_text_enc
from block.models.networks.vqa_net import mask_softmax
from block.models.networks.mlp import MLP
from .fc_layer import FCLayer
from .attention import LowLankBilinearPooling, BilinearAttentionMap
from .classifier import Classifier
from .counting import Counter

import pdb

class BgnNet(nn.Module):

    def __init__(self,
            txt_enc={},
            q_max_length=14,
            glimpse=2,
            objects=36,
            use_counter=True,
            feat_dims={},
            biattention={},
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={}):
        super(BgnNet, self).__init__()
        # self.self_q_att = self_q_att
        self.glimpse = glimpse
        self.q_max_length = q_max_length
        self.objects = objects
        self.use_counter = use_counter
        # self.classif = classif
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        # Modules

        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        self.v_att = BilinearAttentionMap(**feat_dims, glimpse=glimpse)
        self.counter = Counter(objects)

        self.b_net = []
        self.q_prj = []
        self.c_prj = []

        for i in range(glimpse):
            self.b_net.append(LowLankBilinearPooling(**feat_dims, h_out=None, k=1))
            self.q_prj.append(FCLayer(feat_dims['h_dim'], feat_dims['h_dim'], '', .2))
            self.c_prj.append(FCLayer(objects + 1, feat_dims['h_dim'], 'ReLU', .0))

        self.b_net = nn.ModuleList(self.b_net)
        self.q_prj = nn.ModuleList(self.q_prj)
        self.c_prj = nn.ModuleList(self.c_prj)

        self.classifier = Classifier(feat_dims['h_dim'], feat_dims['h_dim']*2, 3000, 0.5)

        Logger().log_value('nparams',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)


    def forward(self, batch):
        # batch のもとは、 https://github.com/Cadene/block.bootstrap.pytorch/blob/master/block/datasets/vqa2.py で作成
        v = batch['visual']            # v.shape:  torch.Size([12, 36, 2048])
        q = batch['question']          # q.shape:  torch.Size([12, question_length_batch_max])
        b = batch['norm_coord']        # c.shape:  torch.Size([12, 36, 4])

        q = self.pad_trim_question(q, self.q_max_length)
        q_emb = self.process_question(q)  # q_emb.shape: torch.Size([12, 14, 2400])


        boxes = b[:,:,:4].transpose(1,2)

        b_emb = [0] * self.glimpse
        attn, logits = self.v_att(v, q_emb)   # attn.shape: torch.Size([12, 2, 36, 14])

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g](v, q_emb, attn[:, g, :, :])  # batch x h_dim x h_dim, eq. (5) in paper

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

            if self.use_counter:
                atten, _ = logits[:, g, :, :].max(2)
                embed = self.counter(boxes, atten)
                q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1) # ← ？ (15)式

        # b_emb[0].shape: torch.Size([12, 512])
        # q_emb.shape: torch.Size([12, 14, 2400])
        # pdb.set_trace()

        logits = self.classifier(q_emb.sum(1))
        # q.shape: torch.Size([bsize, 14, 2400])

        bsize = q.shape[0]
        n_regions = v.shape[1] # v.shape = (batch, 36, 2048)

        # q_expand = q[:,None,:].expand(bsize, n_regions, q.shape[1]) # q.shape[1] = 4800?
        # q_expand = q_expand.contiguous().view(bsize*n_regions, -1)  # q_expand.shape:  torch.Size([432, 4800])

        # logits = self.slp(q) # 暫定的
        # logits = torch.zeros([bsize,3000]) # 暫定的
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # logits = logits.to(device)
        out = {'logits': logits}
        # out = {'logits': logits}   # logits.shape: torch.Size([12, 3000])

        # print('v.shape: ', v.shape)
        # print('q.shape: ', q.shape)
        # print('q_expand.shape: ', q_expand.shape)
        # print('l.shape: ', l.shape)
        # print('c.shape: ', c.shape)
        # pdb.set_trace()


        return out

    def pad_trim_question(self, q, max_length):
        '''
            max_length 以下の question は 0 padding
            max_length 以上の question は trim
        '''
        tmp = torch.zeros([q.shape[0], max_length], dtype=torch.int)
        if q.device.type == 'cuda':
            device = q.device.type + ':' + str(q.device.index)
            tmp = tmp.to(device)

        q = torch.cat((q, tmp), 1)
        q = q[:, :max_length]
        return q

    def process_question(self, q):
        q_emb = self.txt_enc.embedding(q)
        q, _ = self.txt_enc.rnn(q_emb)
        return q

    def process_answers(self, out):
        batch_size = out['logits'].shape[0]
        _, pred = out['logits'].data.max(1)
        pred.squeeze_()
        out['answers'] = [self.aid_to_ans[pred[i]] for i in range(batch_size)]
        out['answer_ids'] = [pred[i] for i in range(batch_size)]
        return out
