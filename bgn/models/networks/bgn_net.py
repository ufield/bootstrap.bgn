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
# from .attention import LowLankBilinearPooling, BilinearAttentionMap
from .graphs import ImageGraph, QuestionGraph
from .classifier import Classifier
from .counting import Counter

import pdb

class BgnNet(nn.Module):

    def __init__(self,
            txt_enc={},
            q_max_length=14,
            i_glimpse=2,
            q_glimpse=2,
            layers=2,
            objects=36,
            use_counter=True,
            v_dim=2048,
            q_dim=2400,
            k_dim=2400,
            # feat_dims={},
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={}):
        super(BgnNet, self).__init__()
        self.layers = layers
        self.q_max_length = q_max_length
        self.objects = objects
        self.use_counter = use_counter
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid

        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        # self.counter = Counter(objects)

        self.image_graphs = []
        self.question_graphs = []

        for i in range(layers):
            self.image_graphs.append(ImageGraph(v_dim, q_dim, k_dim, i_glimpse=i_glimpse))
            self.question_graphs.append(QuestionGraph(q_dim, k_dim, q_glimpse=q_glimpse))

        self.image_graphs = nn.ModuleList(self.image_graphs)
        self.question_graphs = nn.ModuleList(self.question_graphs)

        self.classifier = Classifier(q_dim, q_dim*2, 3000, 0.5)

        Logger().log_value('nparams',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)


    def forward(self, batch):
        # batch のもとは、 https://github.com/Cadene/block.bootstrap.pytorch/blob/master/block/datasets/vqa2.py で作成
        v = batch['visual']            # v.shape:  torch.Size([12, 36, 2048])
        q = batch['question']          # q.shape:  torch.Size([12, question_length_batch_max])

        q = self.pad_trim_question(q, self.q_max_length)
        q_emb = self.process_question(q)  # q_emb.shape: torch.Size([12, 14, 2400])

        o = q_emb
        for l in range(self.layers):
            h = self.image_graphs[l](v, o)
            o = self.question_graphs[l](h)

        logits = self.classifier(o.sum(1))

        out = {'logits': logits}
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
