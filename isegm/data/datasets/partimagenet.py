"""
Written by Yian Zhao
"""

from pathlib import Path
import pickle as pkl
import cv2
import numpy as np

from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class PartINDataset(ISDataset):
    def __init__(self, dataset_path, split='val', images_dir_name='images', masks_dir_name='annotations', **kwargs):
        super(PartINDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / images_dir_name / split
        self._insts_path = self.dataset_path / masks_dir_name / split

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.png')}
        self.dataset_samples = self.get_images_and_ids_list()

    def get_sample(self, index) -> DSample:
        image_name, instance_id = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path).astype(np.int32)[:, :, 0]  
        instances_mask[instances_mask != instance_id] = 0
        instances_mask[instances_mask > 0] = 1

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)

    def get_images_and_ids_list(self):
        pkl_path = self.dataset_path / f'{self.dataset_split}_images_and_ids_list.pkl'

        if pkl_path.exists():
            with open(str(pkl_path), 'rb') as fp:
                images_and_ids_list = pkl.load(fp)
        else:
            images_and_ids_list = []

            for sample in self.dataset_samples:
                mask_path = str(self._masks_paths[sample.split('.')[0]])
                instances_mask = cv2.imread(mask_path).astype(np.int32)
                instances_ids, _ = get_labels_with_sizes(instances_mask)
                instances_ids.remove(40)
                for instances_id in instances_ids:
                    images_and_ids_list.append((sample, instances_id))

            with open(str(pkl_path), 'wb') as fp:
                pkl.dump(images_and_ids_list, fp)

        return images_and_ids_list