from time import time

import numpy as np
import torch
import cv2
from isegm.inference import utils
from isegm.inference.clicker import Click, Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, sam_type=None, oracle=False, **kwargs):
    all_ious = []
    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        for object_id in sample.objects_ids:
            if sam_type is None:
                sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask(object_id), predictor,
                                                sample_id=index, **kwargs)
            else:
                _, sample_ious, _ = evaluate_sample_sam(sample.image, sample.gt_mask(object_id), predictor,
                                                sample_id=index, sam_type=sam_type, oracle=oracle, **kwargs)
            all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time
    return all_ious, elapsed_time


def evaluate_sample_sam(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, sam_type=False, oracle=False, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    with torch.no_grad():
        predictor.set_input_image(image)
        if sam_type == 'SAM':
            for click_indx in range(max_clicks):
                clicker.make_next_click(pred_mask)
                point_coords, point_labels = get_sam_input(clicker)
                if oracle:
                    ious = []
                    pred_masks = []
                    pred_probs, _, _ = predictor.predict(point_coords, point_labels, multimask_output=True, return_logits=True)
                    for idx in range(pred_probs.shape[0]):
                        pred_masks.append(pred_probs[idx] > predictor.model.mask_threshold)
                        ious.append(utils.get_iou(gt_mask, pred_masks[-1]))
                    tgt_idx = np.argmax(np.array(ious))
                    iou = ious[tgt_idx]
                    pred_mask = pred_masks[tgt_idx]
                else:
                    pred_probs, _, _ = predictor.predict(point_coords, point_labels, multimask_output=False, return_logits=True)
                    pred_probs = pred_probs[0]
                    pred_mask = pred_probs > predictor.model.mask_threshold
                    iou = utils.get_iou(gt_mask, pred_mask)
                
                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)
                
                ious_list.append(iou)
                if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                    break
            return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
        else:
            for click_indx in range(max_clicks):
                clicker.make_next_click(pred_mask)
                if oracle:
                    ious = []
                    pred_masks = []
                    for gra in range(1, 11):
                        cur_gra = round(gra * 0.1, 1)
                        pred_probs = predictor.get_prediction(clicker, gra=cur_gra)
                        pred_masks.append(pred_probs > pred_thr)
                        ious.append(utils.get_iou(gt_mask, pred_masks[-1]))
                    tgt_idx = np.argmax(np.array(ious))
                    iou = ious[tgt_idx]
                    pred_mask = pred_masks[tgt_idx]
                else:
                    pred_probs = predictor.get_prediction(clicker)
                    pred_mask = pred_probs > pred_thr
                    iou = utils.get_iou(gt_mask, pred_mask)

                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

                ious_list.append(iou)
                if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                    break
            return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    ious_lists = []
    click_indxs = []
    with torch.no_grad():
        predictor.set_input_image(image)
        min_num = 100
        for gra in range(1, 11):
            cur_gra = round(gra * 0.1, 1)
            ious_list = []
            clicker.reset_clicks()
            pred_mask = np.zeros_like(gt_mask)
            predictor.prev_prediction = torch.zeros_like(predictor.original_image[:, :1, :, :])
            for click_indx in range(max_clicks):
                clicker.make_next_click(pred_mask)
                pred_probs = predictor.get_prediction(clicker, gra=cur_gra)

                pred_mask = pred_probs > pred_thr
                iou = utils.get_iou(gt_mask, pred_mask)

                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

                ious_list.append(iou)
                if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                    min_num = min(min_num, click_indx + 1)
                    break
                if min_num <= max_clicks and click_indx + 1 > min_num:
                    break
            ious_lists.append(np.array(ious_list, dtype=np.float32))
            click_indxs.append(click_indx)
        click_indxs = np.array(click_indxs)
        tgt_idxs = np.squeeze(np.argwhere(click_indxs == np.min(click_indxs)), axis=1)
        selected_ious = [ious_lists[i] for i in tgt_idxs]
        max_index = np.argmax([ious[0] for ious in selected_ious])
        ious = selected_ious[max_index]
        tgt_idx = tgt_idxs[max_index]

    return ious, tgt_idx


def get_sam_input(clicker, reverse=True):
    clicks_list = clicker.get_clicks()
    points_nd = get_points_nd([clicks_list])
    point_length = len(points_nd[0]) // 2
    point_coords = []
    point_labels = []
    for i, point in enumerate(points_nd[0]):
        if point[0] == -1:
            continue
        if i < point_length:
            point_labels.append(1)
        else:
            point_labels.append(0)
        if reverse:
            point_coords.append([point[1], point[0]])  # for SAM
    return np.array(point_coords), np.array(point_labels)

def get_points_nd(clicks_lists):
    total_clicks = []
    num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
    num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
    num_max_points = max(num_pos_clicks + num_neg_clicks)
    num_max_points = max(1, num_max_points)

    for clicks_list in clicks_lists:
        pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

        neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)

    return total_clicks
