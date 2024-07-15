"""
Written by Yian Zhao
"""

import numpy as np
import cv2
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import logging
from scipy.io import loadmat
from albumentations import *
import os
import os.path as osp
import pickle

from isegm.inference.clicker import Click, Clicker
from isegm.inference.predictors import BasePredictor
from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from isegm.utils.serialization import load_model
from isegm.utils.log import logger, TqdmToLogger
from isegm.data.transforms import *
from torchvision import transforms
from isegm.inference.transforms import ZoomIn

def parse_args():
    parser = argparse.ArgumentParser()

    group_checkpoints = parser.add_mutually_exclusive_group(required=True)
    group_checkpoints.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='The path to the checkpoint. '
        'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
        'or an absolute path. The file extension can be omitted.')

    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument('--gpus', type=str, default='0', help='ID of used GPU.')
    group_device.add_argument(
        '--cpu', action='store_true', default=False, help='Use only CPU for inference.')

    parser.add_argument('--save-path', type=str, default='part_output')
    parser.add_argument('--save-name', type=str, default='proposal.pkl')

    parser.add_argument('--dataset-path', type=str, default='/path/to/datasets/SBD/dataset')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")
    
    return args

class ObjDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_path, split='train', buggy_mask_thresh=0.08, min_area_res=1500):
        super(ObjDataset, self).__init__()
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'img'
        self._insts_path = self.dataset_path / 'inst'
        self._buggy_objects = dict()
        self._buggy_mask_thresh = buggy_mask_thresh
        self.min_area_res = min_area_res
        self.to_tensor = transforms.ToTensor()

        with open(self.dataset_path / f'{split}.txt', 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]

    def __getitem__(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        inst_info_path = str(self._insts_path / f'{image_name}.mat')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = loadmat(str(inst_info_path))['GTinst'][0][0][0].astype(np.int32)
        instances_mask = self.remove_buggy_masks(index, instances_mask)
        instances_ids, instances_areas = get_labels_with_sizes(instances_mask)
        if len(instances_ids) == 0:
            return image, instances_mask
        instances_ids = np.array(instances_ids)
        instances_areas = np.array(instances_areas)
        if len(instances_ids) > 0:
            instances_masks = []
            for instance_id in instances_ids:
                mask = instances_mask.copy()
                mask[mask != instance_id] = 0
                mask[mask > 0] = 1
                instances_masks.append(mask)
            instances_masks = np.transpose(np.array(instances_masks), (1,2,0))
            return self.to_tensor(image), self.to_tensor(instances_masks), index
        else:
            instance_id = np.random.choice(instances_ids)

        instances_mask[instances_mask != instance_id] = 0
        instances_mask[instances_mask > 0] = 1
        instances_mask = instances_mask[:, :, np.newaxis]
        return self.to_tensor(image), self.to_tensor(instances_mask), index

    def remove_buggy_masks(self, index, instances_mask):
        if self._buggy_mask_thresh > 0.0:
            buggy_image_objects = self._buggy_objects.get(index, None)
            if buggy_image_objects is None:
                buggy_image_objects = []
                instances_ids, _ = get_labels_with_sizes(instances_mask)
                for obj_id in instances_ids:
                    obj_mask = instances_mask == obj_id
                    mask_area = obj_mask.sum()
                    bbox = get_bbox_from_mask(obj_mask)
                    bbox_area = (bbox[1] - bbox[0] + 1) * (bbox[3] - bbox[2] + 1)
                    obj_area_ratio = mask_area / bbox_area
                    if obj_area_ratio < self._buggy_mask_thresh:
                        buggy_image_objects.append(obj_id)

                self._buggy_objects[index] = buggy_image_objects
            for obj_id in buggy_image_objects:
                instances_mask[instances_mask == obj_id] = 0

        return instances_mask
    
    def augment(self, image, instance_mask):
        aug_output = self.augmentator(image=image, mask=instance_mask)
        return aug_output['image'], aug_output['mask']

    def __len__(self):
        return len(self.dataset_samples)


class PartPredictor(object):
    def __init__(self, device, checkpoint=None, min_area_res=1500, min_area_del=400):
        super(PartPredictor, self).__init__()
        self._predictor = None
        self.device = device
        self.checkpoint = checkpoint
        self.min_area_res = min_area_res
        self.min_area_del = min_area_del
        self.gra_smt = Granularitysmt(checkpoint, device)
        self._load_model()

    def _load_model(self):
        state_dict = torch.load(self.checkpoint, map_location='cpu')
        model = load_single_is_model(state_dict, device=self.device, eval_ritm=False)
        zoom_in = ZoomIn(skip_clicks=-1, target_size=(448, 448))
        self._predictor = BasePredictor(model, device=self.device, zoom_in=zoom_in, with_flip=True)

    def _add_click(self, is_positive, coords=None, clicker=None):
        click = Click(is_positive=is_positive, coords=coords)
        clicker.add_click(click)

    def _add_pos_click(self, coords=None, clicker=None):
        return self._add_click(is_positive=True, coords=coords, clicker=clicker)

    def _add_neg_click(self, coords=None, clicker=None):
        return self._add_click(is_positive=False, coords=coords, clicker=clicker)

    @torch.no_grad()
    def solve(self, image_tensor, gt_mask):
        """
        image_tensor.shape [bs, 3, h, w]
        gt_mask.shape [bs, 1, h, w]
        only support bs == 1 now
        """
        gras = []
        masks = []
        new_gt_mask = gt_mask.clone()
        gt_mask = gt_mask.cpu().numpy()[:, 0, :, :]
        image = image_tensor.clone()
        self._predictor.set_input_image(image)
        clickers = []
        ori_areas = []
        for bindx in range(gt_mask.shape[0]):
            cur_clicker = Clicker()
            cur_clicker.reset_clicks()
            cur_gt_mask = gt_mask[bindx].astype(np.int32)
            mask_label_sizes = np.bincount(cur_gt_mask.flatten())
            if len(mask_label_sizes) == 1:
                clickers.append(cur_clicker)
                ori_areas.append(0)
                continue
            ori_areas.append(mask_label_sizes[1])

            indices = np.argwhere(cur_gt_mask)
            click = indices[np.random.randint(0, len(indices))]
            self._add_pos_click(click, cur_clicker)
            clickers.append(cur_clicker)

        # model forward (optional)
        pred_probs = self._predictor._batch_infer(image, clickers, prev_mask=new_gt_mask)

        # add negative clicks
        max_iter = np.random.randint(3, 6)  # select iters number for a batch
        num_click = 1
        while num_click < max_iter:
            for bindx in range(gt_mask.shape[0]):
                prev_mask = pred_probs[bindx] > 0.5
                cur_clicker = clickers[bindx]
                # check area
                mask_id_size = np.bincount(prev_mask.flatten())
                if len(mask_id_size) == 1 or mask_id_size[1] < self.min_area_del:
                    clickers[bindx] = cur_clicker
                    continue
                indices = np.argwhere(prev_mask)
                click = indices[np.random.randint(0, len(indices))]
                self._add_neg_click(click, cur_clicker)
                clickers[bindx] = cur_clicker
            # model forward
            pred_probs = self._predictor._batch_infer(image, clickers)                    
            num_click += 1

        for bindx in range(gt_mask.shape[0]):
            prev_mask = pred_probs[bindx] > 0.5
            cur_gt_mask = gt_mask[bindx].astype(np.uint8)
            prev_mask = prev_mask.astype(np.uint8)
            num_objects, labels = cv2.connectedComponents(prev_mask)
            if num_objects > 2:
                size_for_each_obj = np.bincount(labels.flatten())[1:]
                candidates_obj_id = np.argwhere(size_for_each_obj > self.min_area_res)[:, 0]
                if len(candidates_obj_id) > 0:
                    tgt_obj_id = np.random.choice(candidates_obj_id)
                else:
                    tgt_obj_id = np.argmax(size_for_each_obj)
                prev_mask[labels != tgt_obj_id + 1] = 0

            if len(np.bincount(prev_mask.flatten())) == 1:
                gra_scale = 0.0
                gra_semantic = 0.0
            else:
                gra_scale = max(0.1, round(np.bincount(prev_mask.flatten())[1] / ori_areas[bindx], 1))
                gra_semantic = self.gra_smt.getgra_semantic(image, prev_mask, new_gt_mask)
            
            if gra_scale > 1.0:
                prev_mask = cur_gt_mask
                gra_scale = 1.0
                gra_semantic = 1.0
            
            gra = [gra_semantic, gra_scale]
            
            res_mask = np.logical_and(np.logical_not(prev_mask), cur_gt_mask).astype(np.uint8)

            num_objects, labels = cv2.connectedComponents(res_mask)
            if num_objects > 2:
                size_for_each_obj = np.bincount(labels.flatten())[1:]
                candidates_obj_id = np.argwhere(size_for_each_obj > self.min_area_res)[:, 0]
                if len(candidates_obj_id) > 0:
                    tgt_obj_id = np.random.choice(candidates_obj_id)
                else:
                    tgt_obj_id = np.argmax(size_for_each_obj)
                res_mask[labels != tgt_obj_id + 1] = 0

            if len(np.bincount(res_mask.flatten())) == 1:
                gra_res = None
                res_mask = None
            else:
                gra_scale_res = round(np.bincount(res_mask.flatten())[1] / ori_areas[bindx], 1)
                if gra_scale_res > 0.1:
                    gra_semantic_res = self.gra_smt.getgra_semantic(image, res_mask, new_gt_mask)
                    gra_res = [gra_semantic_res, gra_scale_res]
                else:
                    gra_res = None
                    res_mask = None

            masks.append((prev_mask, res_mask))
            gras.append((gra, gra_res))
        
        return masks, gras
        
def load_single_is_model(state_dict, device, eval_ritm, **kwargs):
    model = load_model(state_dict['config'], eval_ritm, **kwargs)
    print("Load predictor weights...")
    msg = model.load_state_dict(state_dict['state_dict'], strict=False)
    print(msg)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    return model

def main():
    args = parse_args()
    dataset = ObjDataset(args.dataset_path, args.split)
    dataloader = DataLoader(dataset, args.batch_size, drop_last=False, pin_memory=True, num_workers=args.workers)
    part_predictor = PartPredictor(device=args.device, checkpoint=args.checkpoint)
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    tbar = tqdm(dataloader, file=tqdm_out, ncols=100)
    data = {}
    for i, batch_data in enumerate(tbar):
        image, gt_masks, index = batch_data
        image, gt_masks = image.to(args.device), gt_masks.to(args.device)
        valid_masks = []
        valid_gras = []
        for cnt in range(gt_masks.shape[1]):
            gt_mask = gt_masks[:, cnt:cnt+1, :, :]
            # batch forward
            masks, gras = part_predictor.solve(image, gt_mask)
            gt_mask, res_mask = masks[0]
            gra, gra_res = gras[0]
            valid_masks.append(gt_mask)
            valid_gras.append(gra)
            if res_mask is not None:
                valid_masks.append(res_mask)
                valid_gras.append(gra_res)

            # save mask and gra with pkl files
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            img_name = dataset.dataset_samples[index[0]]
        data.update({img_name: dict(mask=np.array(valid_masks), gra=np.array(valid_gras))})
    with open(osp.join(args.save_path, args.save_name), 'wb') as f:
        pickle.dump(data, f)


class Granularitysmt(object):
    def __init__(self, checkpoint, device):
        self.checkpoint = checkpoint
        self.device = device
        self._load_model()

    def _load_model(self):
        state_dict = torch.load(self.checkpoint, map_location='cpu')
        model = load_single_is_model(state_dict, device=self.device, eval_ritm=False)
        zoom_in = ZoomIn(skip_clicks=-1, target_size=(448, 448))
        self._predictor = BasePredictor(model, device=self.device, zoom_in=zoom_in, with_flip=True)

    def _add_click(self, is_positive, coords=None, clicker=None):
        click = Click(is_positive=is_positive, coords=coords)
        clicker.add_click(click)

    def _add_pos_click(self, coords=None, clicker=None):
        return self._add_click(is_positive=True, coords=coords, clicker=clicker)

    def getgra_semantic(self, image, prev_mask, gt_mask):
        gra_clicker = Clicker()
        erode_r = int(np.ceil(0.1 * np.sqrt(prev_mask.sum())))
        erode_mask = cv2.erode(prev_mask, None, iterations=erode_r)
        indices = np.argwhere(erode_mask)
        if len(indices) == 0:
            indices = np.argwhere(prev_mask)
        click = indices[np.random.randint(0, len(indices))]
        self._add_pos_click(click, gra_clicker)
        self._predictor.set_input_image(image)
        pred_probs = self._predictor.get_prediction(gra_clicker)
        gt_mask = gt_mask.cpu().numpy()[:, 0, :, :]
        cur_gt_mask = gt_mask[0].astype(np.uint8)
        gt_var = np.ptp(pred_probs[cur_gt_mask.astype(np.bool_)])
        mask_var = np.ptp(pred_probs[prev_mask.astype(np.bool_)])
        gra_semantic = min(round(float(mask_var / gt_var), 1), 1.0)

        return gra_semantic

if __name__ == '__main__':
    main()