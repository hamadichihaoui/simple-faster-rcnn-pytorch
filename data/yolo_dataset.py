# --------------------------------------------------------
# Written by Hamadi Chihaoui at 3:34 PM 2/25/2021 
# --------------------------------------------------------
import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image


class YoloDataset:

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.img_files = os.listdir(self.data_dir + 'images/')

    def __len__(self):
        return len(self.img_files)

    def get_example(self, i):

        id_ = self.img_files[i]
        bbox = list()
        label = list()
        difficult = list()
        img_file = self.data_dir + 'images/' + id_
        img = read_image(img_file, color=True)
        _, img_h, img_w = img.shape
        # print(img_h, img_w)
        try:
            with open(self.data_dir + 'labels/' + id_[:-4] + '.txt', 'r') as f:
                data = f.read().split('\n')[:-1]
                for x in data:
                    split = x.split(' ')
                    xc = float(split[1])
                    yc = float(split[2])
                    w = float(split[3])
                    h = float(split[4])

                    # subtract 1 to make pixel indexes 0-based
                    bbox.append([int((yc - 0.5 * h) * img_h), int((xc - 0.5 * w) * img_w), int((yc + 0.5 * h) * img_h),
                                 int((xc + 0.5 * w) * img_w)])

                    label.append(int(split[0]))
        except:
            bbox = np.zeros((0, 4)).astype(np.float32)
            label = np.zeros(0).astype(np.int32)
        if len(bbox) > 0:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
        else:
            bbox = np.zeros((0, 4)).astype(np.float32)
            label = np.zeros(0).astype(np.int32)

        return img, bbox, label

    __getitem__ = get_example

#
# if __name__ == '__main__':
#     import cv2
#
#     dataset = YoloDataset('/home/fatimat/data/airplane_dataset/train/')
#     img, bbox, label = dataset[0]
#     img1 = img.transpose((1, 2, 0))
#     for box in bbox:
#         print(box)
#         img1 = cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
#         print('img1', img1.shape)
#     cv2.imwrite('img1.jpg', img1)



