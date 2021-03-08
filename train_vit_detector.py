# --------------------------------------------------------
# Written by Hamadi Chihaoui at 3:08 PM 2/25/2021 
# --------------------------------------------------------
from __future__ import absolute_import
import os

import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import NewDataset, TestNewDataset, inverse_normalize
# from data.coco_dataset import CocoDataset
from model import FasterRCNNVIT
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.metrics_tool import calculate_map
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import torch

# import gc
# gc.collect()
# torch.cuda.empty_cache()
device = torch.device('cuda')
# torch.backends.cudnn.enabled = False

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')
import numpy as np
from mean_average_precision import MetricBuilder


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults, gt_bboxes_np, gt_labels_np = list(), list(), list(), list(), list()
    tk0 = tqdm(dataloader, total=len(dataloader))
    for ii, (imgs, size, gt_bboxes_, gt_labels_) in tqdm(enumerate(tk0)):
        sizes = [512, 512]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_)
        gt_bboxes_np += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_)
        gt_labels_np += list(gt_labels_.numpy())
        gt_difficults.append(np.array([0 for _ in range(len(gt_labels_[0]))], dtype=np.bool).astype(
            np.uint8))  # np.zeros((len(gt_bboxes_), ))
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        pred_bboxes += pred_bboxes_
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

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes_np, gt_labels_np, gt_difficults,
        use_07_metric=True)
    print('Validation MAP@0.5 = ', result['ap'])
    if len_viewed == 0.:
        print('no prediction was generated')
        return 0.0, 0.0, 0.0
    else:
        return result['ap'], cumm_pr / len_viewed, cumm_rec / len_viewed


def train(**kwargs):
    opt._parse(kwargs)

    dataset = NewDataset('train_repo_path', 512, 512)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestNewDataset('validation_repo_path', 512, 512)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       # pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVIT()  # FasterRCNNVGG16()#
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    best_map = 0.
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        trainer.faster_rcnn.train()
        tk0 = tqdm(dataloader, total=len(dataloader))
        # eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(tk0)):
            # if ii == 5000: break
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            if len(bbox) == 0:
                continue
            elif len(bbox[0]) == 0:
                continue
            else:
                trainer.train_step(img, bbox, label, scale)
                rpn_cls = round(trainer.get_meter_data()['rpn_cls_loss'], 3)
                rpn_loc = round(trainer.get_meter_data()['rpn_loc_loss'], 3)
                roi_cls = round(trainer.get_meter_data()['roi_cls_loss'], 3)
                roi_loc = round(trainer.get_meter_data()['roi_loc_loss'], 3)
                total_loss = rpn_cls + rpn_loc + roi_cls + roi_loc
                tk0.set_postfix(rpn_cls=rpn_cls,
                                rpn_loc=rpn_loc,
                                roi_cls=roi_cls,
                                roi_loc=roi_loc, total_loss=total_loss)

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']

        # Save the new model if it has a bigger MAP@0.5
        if eval_result[0] > best_map:  #
            best_map = eval_result[0]  # eval_result[0]
            best_path = trainer.save(best_map=best_map)
        if epoch == 15:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 50:
            break


if __name__ == '__main__':
    import fire

    fire.Fire()