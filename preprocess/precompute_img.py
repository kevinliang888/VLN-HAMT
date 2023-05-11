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

BLIP2DICT = {
    'FlanT5 XXL': 'Salesforce/blip2-flan-t5-xxl',
    'FlanT5 XL COCO': 'Salesforce/blip2-flan-t5-xl-coco',
    'OPT6.7B COCO': 'Salesforce/blip2-opt-6.7b-coco',
    'OPT2.7B COCO': 'Salesforce/blip2-opt-2.7b-coco',
    'FlanT5 XL': 'Salesforce/blip2-flan-t5-xl',
    'OPT6.7B': 'Salesforce/blip2-opt-6.7b',
    'OPT2.7B': 'Salesforce/blip2-opt-2.7b',
}

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features', 'logits']
VIEWPOINT_SIZE = 36  # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

WIDTH = 640
HEIGHT = 480
VFOV = 60


def build_feature_extractor(model_name, bit8=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = {'load_in_8bit': True} if bit8 else {'torch_dtype': torch.float16}
    processor = Blip2Processor.from_pretrained(BLIP2DICT[model_name])
    # model = Blip2ForConditionalGeneration.from_pretrained(BLIP2DICT[model_name], **dtype).to(device)
    model = None

    return model, processor, device


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


def process_images(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, processor, device = build_feature_extractor(args.model_name)

    for scan_id, viewpoint_id in scanvp_list:
        # Loop all discretized views from this location
        images = []
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

            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
            image = inputs["pixel_values"]
            images.append(image)

        images = torch.vstack(images)
        images = images.cpu().numpy()
        out_queue.put((scan_id, viewpoint_id, images))

    out_queue.put(None)


def build_images_file(args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_images,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        )
        process.start()
        processes.append(process)

    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(scanvp_list))
    progress_bar.start()

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, imgs = res
                key = '%s_%s' % (scan_id, viewpoint_id)
                data = imgs
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV

                num_finished_vps += 1
                print(num_finished_vps)
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='FlanT5 XL')
    parser.add_argument('--connectivity_dir', default="../datasets/R2R/connectivity")
    parser.add_argument('--scan_dir', default="/n/fs/kl-project/datasets/r2r/base_dir/v1/scans")
    parser.add_argument('--out_image_logits', action='store_true', default=False)
    parser.add_argument('--output_file', default="../datasets/R2R/images/all_imgs.hdf5")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    build_images_file(args)


