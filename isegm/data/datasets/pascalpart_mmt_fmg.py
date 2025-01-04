"""
Written by Yian Zhao
"""

from pathlib import Path
import pickle as pkl
import cv2
import numpy as np
from scipy.io import loadmat

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from isegm.model.modeling.clip import clip


class PascalPartMMTFMGDataset(ISDataset):
    '''
    MMT: Multi-grained Mask Trie (for extending the original GT)
    FMG: Fine-grained Mask Generator (original any granularity generator)
    '''
    def __init__(self, dataset_path, split='train', buggy_mask_thresh=0.08, partmask_path=None, select_proposal=False,
                 **kwargs):
        super(PascalPartMMTFMGDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'JPEGImages'
        self._anno_path = self.dataset_path / 'Annotations_Part'
        self._buggy_objects = dict()
        self._buggy_mask_thresh = buggy_mask_thresh
        self.select_proposal = select_proposal

        with open(self.dataset_path / f'ImageSets/Main/{self.dataset_split}.txt', 'r') as f:
            self.dataset_samples = [name.strip() for name in f.readlines()]

        self.abbr2word = {
            "lwing": "left wing", "rwing": "right wing", "fwheel": "front wheel", "bwheel": "back wheel",
            "haddlebar": "haddle bar", "chainwheel": "chain wheel",
            "leye": "left eye", "reye": "right eye", "lwing": "left wing", "rwing": "right wing", "lleg": "left leg",
            "rleg": "right leg", "lfoot": "left foot",
            "rfoot": "right foot", "fliplate": "front license plate", "bliplate": "back license plate",
            "lear": "left ear", "rear": "right ear", "lfleg": "left front leg",
            "lfpa": "left front paw", "rfleg": "right front leg", "rfpa": "right front paw", "lbleg": "left back leg",
            "lbpa": "left back paw", "rbleg": "right back leg",
            "rbpa": "right back paw", "lhorn": "left horn", "rhorn": "right horn", "lfuleg": "left front upper leg",
            "lflleg": "left front lower leg", "rfuleg": "right front upper leg",
            "rflleg": "right front lower leg", "lbuleg": "left back upper leg", "lblleg": "left back lower leg",
            "rbuleg": "right back upper leg", "rblleg": "right back lower leg",
            "lfho": "left front hoof", "rfho": "right front hoof", "lbho": "left back hoof", "rbho": "right back hoof",
            "lebrow": "left eyebrow", "rebrow": "right eyebrow",
            "llarm": "left lower arm", "luarm": "left upper arm", "rlarm": "right lower arm",
            "ruarm": "right upper arm", "lhand": "left hand", "rhand": "right hand", "llleg": "left lower leg",
            "luleg": "left upper leg", "rlleg": "right lower leg", "ruleg": "right upper leg", "lfoot": "left foot",
            "rfoot": "right foot", "hfrontside": "head front side",
            "hleftside": "head left side", "hrightside": "head right side", "hbackside": "head back side",
            "hroofside": "head roof side", "cfrontside": "coach front side",
            "cleftside": "coach left side", "crightside": "coach right side", "cbackside": "coach back side",
            "croofside": "coach roof side", "leftmirror": "left mirror",
            "rightmirror": "right mirror", "frontside": "front side", "backside": "back side", "leftside": "left side",
            "rightside": "right side", "roofside": "roof side",
        }
        self.probs = {
            0.2: 0.06960800373840834, 0.3: 0.09004898499961728, 1.0: 0.06696065989491415, 0.1: 0.012166262603621974,
            0.4: 0.10218302562007273,
            0.6: 0.12601478544277883, 0.5: 0.11521376884572793, 0.7: 0.1343502368084108, 0.9: 0.14037031069344932,
            0.8: 0.14308396135299856
        }  # precompute
        if self.probs is None:
            self._count_gra()
        self.partmask_path = partmask_path
        if partmask_path is not None:
            with open(partmask_path, 'rb') as f:
                self.partmask_samples = pkl.load(f)
            print(f"load proposals from {partmask_path}.")
            self.gra_probs = self._count_gra_probs(self.partmask_samples)

    def _count_gra(self):
        from tqdm import tqdm
        gra_cnt = {}
        for image_name in tqdm(self.dataset_samples):
            inst_info_path = str(self._anno_path / f'{image_name}.mat')
            ann = loadmat(inst_info_path)['anno']['objects'][0, 0]
            for instance_id in range(ann.shape[1]):
                parts = ann['parts'][0, instance_id]
                masks = [PartNode(ann[0, instance_id]['mask'], ann['class'][0, instance_id][0])]
                for part_id in range(parts.shape[1]):
                    masks.append(
                        PartNode(parts['mask'][0, part_id], self._prase_part_name(parts['part_name'][0, part_id][0])))
                masks = self._build_mask_tree(masks)
                # calculate gra for each mask
                gras = self._get_gra_list(masks, ann[0, instance_id]['mask'])
                for gra in gras:
                    gra_cnt[gra] = gra_cnt.get(gra, 0) + 1

        gra_probs = {}
        total_cnt = sum(gra_cnt.values())
        for gra, cnt in gra_cnt.items():
            gra_probs[gra] = np.log(total_cnt / cnt)

        self.probs = {gra: x / sum(gra_probs.values()) for gra, x in gra_probs.items()}
        print(self.probs)

    def _count_gra_probs(self, partmask_samples):
        gra_num = {}
        for img_name in partmask_samples.keys():
            gras = partmask_samples[img_name]["gra"]
            for idx in range(gras.shape[0]):
                gra = gras[idx][1]
                gra_num[gra] = gra_num.get(gra, 0) + 1

        sum_num = 0
        for k, v in gra_num.items():
            if k != 0.0:
                sum_num += v

        gra_probs = {}
        for k, v in gra_num.items():
            if k != 0.0:
                gra_probs[k] = sum_num / v
        gra_probs[0.0] = 0.

        return gra_probs

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        inst_info_path = str(self._anno_path / f'{image_name}.mat')

        # load img
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.select_proposal and self.partmask_path is not None:
            instances_mask, gra = self.get_proposal(image_name)
            prompt = clip.tokenize("").squeeze(0)
            if instances_mask is not None:
                instances_mask = self.remove_buggy_masks(index, instances_mask)
                instances_ids, _ = get_labels_with_sizes(instances_mask)
                return DSample(image, instances_mask, gra=gra, prompt=prompt, objects_ids=instances_ids,
                               sample_id=index)

        # load ann
        ann = loadmat(inst_info_path)['anno']['objects'][0, 0]

        # select instance
        instance_id = np.random.randint(0, ann.shape[1])
        class_name = ann['class'][0, instance_id][0]

        # select part
        parts = ann['parts'][0, instance_id]
        if parts.shape[1] > 0:
            # build part mask tree
            masks = [PartNode(ann[0, instance_id]['mask'], class_name)]
            for part_id in range(parts.shape[1]):
                masks.append(
                    PartNode(parts['mask'][0, part_id], self._prase_part_name(parts['part_name'][0, part_id][0])))
            masks = self._build_mask_tree(masks)
            # calculate gra for each mask
            gras = self._get_gra_list(masks, ann[0, instance_id]['mask'])
            # calculate probs for each mask
            probs = self._get_prob_list(gras)
            tgt_idx = np.random.choice(range(len(masks)), p=probs)
            instances_mask = masks[tgt_idx].mask
            part_name = masks[tgt_idx].name
            gra = gras[tgt_idx]
        else:
            # part == instance
            instances_mask = ann[0, instance_id]['mask']
            part_name = class_name
            gra = 1.0

        if part_name is not None:
            prompt = clip.tokenize(part_name).squeeze(0)

        return DSample(image, instances_mask, gra=gra, prompt=prompt, objects_ids=[1], sample_id=index)

    def _get_gra_list(self, masks, obj_mask):
        obj_area = np.bincount(obj_mask.flatten())[1]
        gras = []
        for node in masks:
            part_mask = node.mask
            part_area = np.bincount(part_mask.flatten())[1]
            gras.append(max(round(part_area / obj_area, 1), 0.1))

        return gras

    def _get_prob_list(self, gras):
        gra_num_each, probs_nouni = {}, {}
        for gra in gras:
            gra_num_each[gra] = gra_num_each.get(gra, 0) + 1

        for gra, cnt in gra_num_each.items():
            probs_nouni[gra] = self.probs[gra] / cnt
        p = []
        for gra in gras:
            p.append(probs_nouni[gra])
        sum_p = sum(p)
        p = [x / sum_p for x in p]
        return p

    def _build_mask_tree(self, masks):
        # Step 1: Save category dict
        category_dict = {}
        directional_words = ["left", "right", "back", "front"]
        for node in masks:
            mask = node.mask
            name = node.name
            if len(name.split()) > 1:
                name_list = name.split()
                obj = name_list[-1]  # Extract the last word from the name (add 's' in the end)
                if obj == "side":
                    if len(name.split()) == 3:
                        obj = name_list[0]

                # obj self
                if obj in category_dict:
                    category_dict[obj].append(mask)
                else:
                    category_dict[obj] = [mask]

                # directional + obj
                for directional in directional_words:
                    if set([directional, obj]).issubset(set(name_list)):
                        comb = " ".join([directional, obj])
                        if comb in category_dict:
                            category_dict[comb].append(mask)
                        else:
                            category_dict[comb] = [mask]

        # Step 2: Merge masks with the same category
        merged_masks = []
        for category, masks_list in category_dict.items():
            if len(masks_list) > 1:
                merged_mask = masks_list[0]  # Initialize the merged mask with the first mask
                for i in range(1, len(masks_list)):
                    merged_mask = merged_mask | masks_list[i]  # Perform bitwise or operation to merge masks
                if len(category.split()) == 1:
                    merged_name = category + 's' if category not in ["head",
                                                                     "coach"] else category  # Append 's' to the category name
                else:
                    merged_name = category
                merged_masks.append(PartNode(merged_mask, merged_name))

        return merged_masks + masks

    def _prase_part_name(self, part_name):
        part_name = part_name.split("_")[0]
        if part_name in self.abbr2word.keys():
            part_name = self.abbr2word[part_name]

        return part_name

    def get_proposal(self, image_name):
        if image_name in self.partmask_samples.keys():
            part_data = self.partmask_samples[str(image_name)]
            instances_mask, gra = part_data['mask'], part_data['gra']

            if isinstance(gra[0], np.ndarray):
                gra = gra[:, 1]
            gra_num_each, probs = {}, {}
            for g in gra:
                gra_num_each[g] = gra_num_each.get(g, 0) + 1
            for g, cnt in gra_num_each.items():
                probs[g] = self.gra_probs[g] / cnt
            p = []
            for g in gra:
                p.append(probs[g])
            sum_p = sum(p)
            p = [x / sum_p for x in p]
            tgt_idx = np.random.choice(range(gra.shape[0]), p=p)
            instances_mask, gra = instances_mask[tgt_idx], gra[tgt_idx]

            return instances_mask, gra
        else:
            return None, None

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

class PartNode:
    def __init__(self, mask, name):
        self.mask = mask
        self.name = name

    def __repr__(self):
        return f"{self.name}"
