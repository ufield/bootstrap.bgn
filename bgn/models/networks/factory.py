import sys
import copy
import torch
import torch.nn as nn
from bootstrap.lib.options import Options
from bootstrap.models.networks.data_parallel import DataParallel
from block.models.networks.vqa_net import VQANet as AttentionNet
from .bgn_net import BgnNet

def factory(engine):
    mode = list(engine.dataset.keys())[0]
    dataset = engine.dataset[mode]
    opt = Options()['model.network']

    if opt['name'] == 'bgn_net':
        net = BgnNet(
            txt_enc=opt['txt_enc'],
            glimpse=opt['glimpse'],
            objects=opt['objects'],
            use_counter=opt['use_counter'],
            feat_dims=opt['feat_dims'],
            q_max_length=opt['q_max_length'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid)

    else:
        raise ValueError(opt['name'])

    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    return net


if __name__ == '__main__':
    factory()
