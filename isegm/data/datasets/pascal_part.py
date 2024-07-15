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

class PascalPartDataset(ISDataset):

    def __init__(self, dataset_path, split='train', enable_gra=False, **kwargs):
        super(PascalPartDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'JPEGImages'
        self._anns_path = self.dataset_path / 'Annotations_Part'
        self.enable_gra = enable_gra

        with open(self.dataset_path / f'ImageSets/Main/part_{self.dataset_split}.txt', 'r') as f:
            self.dataset_samples = [name.strip() for name in f.readlines()]

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        inst_info_path = str(self._anns_path / f'{image_name}.mat')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ann = loadmat(inst_info_path)['anno']['objects'][0, 0]

        instance_id = np.random.randint(0, ann.shape[1])
        parts = ann['parts'][0, instance_id]
        if parts.shape[1] > 0:
            part_id = np.random.randint(0, parts.shape[1])
            instances_mask = parts['mask'][0, part_id]
        else:
            instances_mask = ann[0, instance_id]['mask']
        
        if self.enable_gra:
            obj_mask = ann[0, instance_id]['mask']
            part_area = np.bincount(instances_mask.flatten())[1]
            obj_area = np.bincount(obj_mask.flatten())[1]
            gra = max(round(part_area / obj_area, 1), 0.1)
            return DSample(image, instances_mask, gra=gra, objects_ids=[1], sample_id=index)

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index) 


class PascalPartEvaluationDataset(ISDataset):

    def __init__(self, dataset_path, split='val', class_name=None, part_name=None, **kwargs):
        super(PascalPartEvaluationDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'JPEGImages'
        self._anns_path = self.dataset_path / 'Annotations_Part'

        with open(self.dataset_path / f'ImageSets/Main/part_{self.dataset_split}.txt', 'r') as f:
            self.dataset_samples = [name.strip() for name in f.readlines()]

        self.all_class = ['bottle', 'cow', 'motorbike', 'pottedplant', 'cat', 'car', 'aeroplane', 'tvmonitor', 'train', 
            'dog', 'bus', 'bird', 'horse', 'sheep', 'bicycle', 'person']
        if class_name is not None:
            assert class_name in self.all_class
        self.dataset_samples = self.get_sbd_images_and_ids_list(class_name, part_name)

    def get_sample(self, index) -> DSample:
        image_name, instance_id, part_id, class_name, part_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        inst_info_path = str(self._anns_path / f'{image_name}.mat')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = loadmat(str(inst_info_path))['anno']['objects'][0, 0]['parts'][0, instance_id]['mask'][0, part_id]

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index, class_name=class_name, part_name=part_name)

    def get_sbd_images_and_ids_list(self, class_name=None, part_name=None):
        
        if class_name is not None:
            if part_name is not None:
                pkl_path = self.dataset_path / f'{class_name}_{part_name}_{self.dataset_split}_images_and_ids_list.pkl'
            else:
                pkl_path = self.dataset_path / f'{class_name}_{self.dataset_split}_images_and_ids_list.pkl'
        else:
            pkl_path = self.dataset_path / f'{self.dataset_split}_images_and_ids_list.pkl'

        if pkl_path.exists():
            with open(str(pkl_path), 'rb') as fp:
                images_and_ids_list = pkl.load(fp)
        else:
            images_and_ids_list = []
            for sample in self.dataset_samples:
                inst_info_path = str(self._anns_path / f'{sample}.mat')
                ann = loadmat(inst_info_path)['anno']['objects'][0, 0]
                n_objects = ann.shape[1]
                for i in range(n_objects):
                    if class_name is not None and ann['class'][0, i][0] != class_name:
                        continue
                    parts = ann['parts'][0, i]
                    n_parts = parts.shape[1]
                    for j in range(n_parts):
                        if part_name is not None and parts['part_name'][0, j][0] != part_name:
                            continue
                        images_and_ids_list.append((sample, i, j, ann['class'][0, i][0], parts['part_name'][0, j][0]))
        
            with open(str(pkl_path), 'wb') as fp:
                pkl.dump(images_and_ids_list, fp)
        
        return images_and_ids_list