"""
Written by Yian Zhao
"""

from pathlib import Path
import cv2
import numpy as np
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

class PascalVocDataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super().__init__(**kwargs)
        assert split in {'train', 'val', 'trainval', 'test'}
        self.name = 'Pascal'
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / "JPEGImages"
        self._insts_path = self.dataset_path / "SegmentationObject"
        self.dataset_split = split

        with open(self.dataset_path / f'ImageSets/Segmentation/{split}.txt', 'r') as f:
            dataset_samples = [name.strip() for name in f.readlines()]
        image_id_lst = self.get_images_and_ids_list(dataset_samples)
        self.dataset_samples = image_id_lst

    def get_sample(self, index) -> DSample:
        sample_id, obj_id = self.dataset_samples[index]

        image_path = str(self._images_path / f'{sample_id}.jpg')
        mask_path = str(self._insts_path / f'{sample_id}.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)

        instance_id = obj_id
        mask = np.zeros_like(instances_mask)
        mask[instances_mask == 220] = 220  # ignored area
        mask[instances_mask == instance_id] = 1
        objects_ids = [1]
        instances_mask = mask
        return DSample(image, instances_mask, objects_ids=objects_ids, ignore_ids=[220], sample_id=index)

    def get_images_and_ids_list(self,dataset_samples):
        images_and_ids_list = []
        for i in range(len(dataset_samples)):
            sample_id = dataset_samples[i]
            mask_path = str(self._insts_path / f'{sample_id}.png')
            instances_mask = cv2.imread(mask_path)
            instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)
            objects_ids = np.unique(instances_mask)
            objects_ids = [x for x in objects_ids if x != 0 and x != 220]
            for j in objects_ids:
                images_and_ids_list.append([sample_id,j])
        return images_and_ids_list