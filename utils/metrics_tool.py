# --------------------------------------------------------
# Written by Hamadi Chihaoui at 3:31 PM 2/25/2021 
# --------------------------------------------------------
import torch

def iou(bboxes_preds, bboxes_targets):
    area1 = (bboxes_preds[:, 2] - bboxes_preds[:, 0]) * (bboxes_preds[:, 3] - bboxes_preds[:, 1])
    area2 = (bboxes_targets[:, 2] - bboxes_targets[:, 0]) * (bboxes_targets[:, 3] - bboxes_targets[:, 1])
    width = (torch.min(bboxes_preds[:, 2, None], bboxes_targets[:, 2]) -
             torch.max(bboxes_preds[:, 0, None], bboxes_targets[:, 0])).clamp(min=0)
    height = (torch.min(bboxes_preds[:, 3, None], bboxes_targets[:, 3]) -
              torch.max(bboxes_preds[:, 1, None], bboxes_targets[:, 1])).clamp(min=0)
    inter = width * height
    return inter / (area1[:, None] + area2 - inter)  # p * t


def average_precision(bbox_preds, conf_preds, bbox_targets):
    if len(bbox_targets) == 0 and len(bbox_preds) == 0:
        # print("no predictions and no ground truth")
        return 1.0, 1.0, 1.0
    elif len(bbox_targets) == 0:
        # print("no ground truth")
        return 0.0, 1.0, 0.0
    elif len(bbox_preds) == 0:
        # print("no predictions")
        return 0.0, 0.0, 1.0
    else:
        thresholds = [0.5]  # 0.55, 0.6, 0.65, 0.7, 0.75
        bbox_preds = bbox_preds[torch.argsort(conf_preds, descending=True)]
        scores = iou(bbox_preds, bbox_targets)
        precision = 0.0
        rec = 0.0
        pr = 0.0
        for threshold in thresholds:
            matched_targets = torch.zeros(bbox_targets.size(0))
            tp = 0
            fp = 0
            fn = 0
            for score in scores:
                score[matched_targets == 1] = 0.0
                s, i = score.max(dim=0)
                if s >= threshold:
                    tp += 1
                    matched_targets[i] = 1
                else:
                    fp += 1
            fn = torch.sum((matched_targets == 0)).item()

            precision += tp / (tp + fp + fn)
            rec += tp / (tp + fn)
            pr += tp / (tp + fp)
            # print('tp', tp, 'fp', fp, 'fn', fn)
        precision /= len(thresholds)
        rec /= len(thresholds)
        pr /= len(thresholds)
        return precision, rec, pr


def calculate_map(dets, scores, targets, batch_size, device):
    sc_precision = 0.0
    sc_rec = 0.0
    sc_pr = 0.0
    score_threshold = 0.5
    for i in range(batch_size):
        try:
            boxes = dets[i]
        except:
            print('dets[i]', dets[i])
        # scores = dets[i][:,4]
        indexes = torch.where(scores > score_threshold)[0]
        boxes = boxes[indexes]
        scores = scores[indexes]
        target_boxes = targets[i]

        av_pre, rec, pr = average_precision(boxes, scores, target_boxes.to(device))
        sc_precision = sc_precision + av_pre
        sc_rec = sc_rec + rec
        sc_pr = sc_pr + pr
    return sc_precision, sc_rec, sc_pr
