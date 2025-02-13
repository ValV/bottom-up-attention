#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus,
   each produces a separate tsv file that can be merged later (e.g. by using
   merge_tsv function). Modify the load_image_ids script as necessary for your
   data location.
"""

# Example:
# python ./tools/generate_tsv_v2.py
# --gpu 0,1,2,3,4,5,6,7
# --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml
# --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt
# --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel
# --split conceptual_captions_train --data_root {Conceptual_Captions_Root}
# --out {Conceptual_Captions_Root}/train_frcnn/


import _init_paths  # FIXME: initialize local library paths in another way

import argparse
import base64
import csv
import json
import os
import random
import sys

from multiprocessing import Process
from pprint import pprint

import caffe

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test import im_detect, _get_blobs
from utils.timer import Timer
from zip_helper import ZipHelper

import cv2 as cv
import numpy as np

csv.field_size_limit(sys.maxsize)  # TODO: check unused

FIELDNAMES = [
    'image_id', 'image_w', 'image_h', 'num_boxes',
    'boxes', 'classes', 'attrs', 'features'
]

# Settings for the number of features per image. 
MIN_BOXES = 10
MAX_BOXES = 100


def load_image_ids(split_name, data_root):
    """
    Load a list of (path,image_id tuples). Modify this to suit your data locations
    """
    split = []
    if split_name == 'conceptual_captions_train':
        with open(os.path.join(data_root, 'utils/train.json')) as f:
            for cnt, line in enumerate(f):
                d = json.loads(line)
                cap = d['caption']
                filepath = d['image']
                image_id = int(filepath.split('/')[-1][:-4])
                split.append((filepath, image_id))
    elif split_name == 'conceptual_captions_val':
        with open(os.path.join(data_root, 'utils/val.json')) as f:
            for cnt, line in enumerate(f):
                d = json.loads(line)
                cap = d['caption']
                filepath = d['image']
                image_id = int(filepath.split('/')[-1][:-4])
                split.append((filepath, image_id))
    else:
        print('Unknown split')
    return split


def get_detections_from_im(net, im_file, image_id, ziphelper, data_root, conf_thresh=0.5):
    zip_image = ziphelper.imread(str(os.path.join(data_root, im_file)))
    im = cv.cvtColor(np.array(zip_image), cv.COLOR_RGB2BGR)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': base64.b64encode(cls_boxes[keep_boxes]).decode('ascii'),
        'classes': base64.b64encode(scores[keep_boxes]).decode('ascii'),
        'attrs': base64.b64encode(attr_scores[keep_boxes]).decode('ascii'),
        'features': base64.b64encode(pool5[keep_boxes]).decode('ascii')
    }


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfolder',
                        help='output folder',
                        default=None, type=str)
    parser.add_argument('--data_root', dest='data_root',
                        help='data root path',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def generate_tsv(gpu_id, prototxt, weights, image_ids, data_root, outfolder):
    # First check if file exists, and if it is complete
    wanted_ids = set([int(image_id[1]) for image_id in image_ids])
    found_ids = set()
    if os.path.exists(outfolder):
        for ids in wanted_ids:
            json_file = f"{ids:08d}.json"
            if os.path.exists(os.path.join(outfolder, json_file)):
                found_ids.add(ids)

    missing = wanted_ids - found_ids
    device = f'GPU {gpu_id:d}' if isinstance(gpu_id, int) and gpu_id >= 0 else 'CPU'
    if len(missing) == 0:
        print(f"{device}: already completed {len(image_ids):d}")
    else:
        print(f"{device}: missing {len(missing):d}/{len(image_ids):d}")

    if len(missing) > 0:
        if isinstance(gpu_id, int) and gpu_id >= 0:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
        else:
            caffe.set_mode_cpu()
        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        ziphelper = ZipHelper()
        _t = {'misc': Timer()}
        count = 0
        for im_file, image_id in image_ids:
            if int(image_id) in missing:
                _t['misc'].tic()
                json_file = f"{image_id:08d}.json"
                with open(os.path.join(outfolder, json_file), 'w') as f:
                    json.dump(get_detections_from_im(net, im_file, image_id,
                                                     ziphelper, data_root), f)
                _t['misc'].toc()
                if (count % 100) == 0:
                    time_avg = _t['misc'].average_time
                    print(
                        f"{device}: {count + 1:d}/{len(missing):d} "
                        f"{time_avg:.3f}s (projected finish: "
                        f"{time_avg * (len(missing) - count) / 3600:.2f} hours)"
                    )
                count += 1


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint(cfg)
    assert cfg.TEST.HAS_RPN

    image_ids = load_image_ids(args.data_split, args.data_root)
    random.seed(10)
    random.shuffle(image_ids)
    # Split image ids between gpus
    if not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder)

    image_ids = [image_ids[i::len(gpus)] for i in range(len(gpus))]

    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []

    for i, gpu_id in enumerate(gpus):
        p = Process(target=generate_tsv,
                    args=(gpu_id, args.prototxt, args.caffemodel, image_ids[i], args.data_root, args.outfolder))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
