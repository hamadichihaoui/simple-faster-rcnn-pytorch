# --------------------------------------------------------
# Written by Hamadi Chihaoui at 3:42 PM 2/25/2021 
# --------------------------------------------------------
from __future__ import absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

import cv2
import torch

device = torch.device('cuda')
from utils.config import opt
from data.dataset import NewDataset, TestNewDataset, inverse_normalize
from model import FasterRCNNVIT
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils.metrics_tool import calculate_map

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

CONFIDENCE_THRESHOLD = 0.8

import torch

device = torch.device('cuda')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    tk0 = tqdm(dataloader, total=len(dataloader))
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(tk0)):
        if ii == 10: break
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        if sizes[0] != 1024 or sizes[0] != 1024:
            continue
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        img1 = imgs.squeeze(0).permute(1, 2, 0).cpu().numpy().copy()

        for box, score in zip(pred_bboxes_[0], pred_scores_[0]):

            if score > CONFIDENCE_THRESHOLD:
                print(box, score)
                img1 = cv2.rectangle(img1, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (255, 0, 0), 2)
        cv2.imwrite(str(ii) + '.jpg', cv2.cvtColor(img1 * 255., cv2.COLOR_RGB2BGR))
        gt_bboxes += list(gt_bboxes_)
        gt_labels += list(gt_labels)
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break
    cumm_pr = 0.
    cumm_rec = 0.
    len_viewed = 0.
    for k in range(len(pred_bboxes)):
        dets = pred_bboxes[k]
        scores = pred_scores[k]
        targets = gt_bboxes[k]
        # print('dets', type(dets), 'scores', type(scores), 'targets', type(targets))
        sc_precision, sc_rec, sc_pr = calculate_map([torch.from_numpy(dets).cuda()], torch.from_numpy(scores).cuda(),
                                                    [targets.cuda()], 1, device)
        len_viewed = len_viewed + 1
        cumm_pr = cumm_pr + sc_pr
        cumm_rec = cumm_rec + sc_rec

    return cumm_pr / len_viewed, cumm_rec / len_viewed


def train(**kwargs):
    opt._parse(kwargs)

    testset = TestNewDataset('/home/hamadic/airplanes_dataset/valid/', 1024, 1024)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVIT()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load(
        '/home/hamadic/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_02230039_0.3391428614619873')

    eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)


if __name__ == '__main__':
    import fire

    fire.Fire()

