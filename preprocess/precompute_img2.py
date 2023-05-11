#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys

sys.path.append('/n/fs/kl-project/Matterport3DSimulator/build')
import MatterSim

import argparse
import numpy as np
import math
import h5py
from PIL import Image
from progressbar import ProgressBar
import pdb

import torch
import torch.multiprocessing as mp

from utils import load_viewpoint_ids
from transformers import Blip2Processor, Blip2ForConditionalGeneration

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features', 'logits']
VIEWPOINT_SIZE = 36  # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

WIDTH = 640
HEIGHT = 480
VFOV = 60


def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim


def process_images(args):
    scanvp_list = load_viewpoint_ids(args.connectivity_dir)
    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    for scan_id, viewpoint_id in scanvp_list:
        # Loop all discretized views from this location
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            image = np.array(state.rgb, copy=True)  # in BGR channel
            image = Image.fromarray(image[:, :, ::-1])  # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            path = os.path.join(args.out_dir, scan_id, viewpoint_id)
            if not os.path.exists(path):
                # Create a new directory because it does not exist
                os.makedirs(path)
            new_path = os.path.join(path, str(ix) + ".jpg")
            img1 = image.save(new_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connectivity_dir', default="../datasets/R2R/connectivity")
    parser.add_argument('--scan_dir', default="/n/fs/kl-project/datasets/r2r/base_dir/v1/scans")
    parser.add_argument('--out_dir', default="../datasets/R2R/view_imgs")
    args = parser.parse_args()
    process_images(args)


