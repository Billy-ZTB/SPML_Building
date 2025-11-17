from ast import List
import cv2
import os
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from spml.data.datasets.base_dataset import ListDataset
import spml.data.transforms as transforms

def pil_to_tensor_preserve_channels(img):
    """Convert PIL image to torch.FloatTensor preserving channels:
    - if grayscale (H,W) -> returns (1,H,W)
    - if RGB (H,W,3) -> returns (3,H,W)
    Result dtype: float32 (no normalization).
    """
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[None, ...]            # (1, H, W)
    else:
        arr = arr.transpose(2, 0, 1)    # (C, H, W)
    return torch.from_numpy(arr).float().squeeze()

class ContrastiveScribbleBuildingDataset(ListDataset):
    def __init__(self,
               data_dir,
               data_list,
               img_mean=(0, 0, 0),
               img_std=(1, 1, 1),
               size=None,
               random_crop=False,
               random_scale=False,
               random_mirror=False,
               training=False):
        """Base class for Dataset.

        Args:
        data_dir: A string indicates root directory of images and labels.
        data_list: A list of strings which indicate path of paired images
            and labels. 'image_path semantic_label_path instance_label_path'.
        img_mean: A list of scalars indicate the mean image value per channel.
        img_std: A list of scalars indicate the std image value per channel.
        size: A tuple of scalars indicate size of output image and labels.
            The output resolution remain the same if `size` is None.
        random_crop: enable/disable random_crop for data augmentation.
            If True, adopt randomly cropping as augmentation.
        random_scale: enable/disable random_scale for data augmentation.
            If True, adopt adopt randomly scaling as augmentation.
        random_mirror: enable/disable random_mirror for data augmentation.
            If True, adopt adopt randomly mirroring as augmentation.
        training: enable/disable training to set dataset for training and
            testing. If True, set to training mode.
        """
        self.image_paths, self.semantic_label_paths, self.instance_label_paths, self.edge_label_paths, self.gray_image_paths = (
        self._read_image_and_label_paths(data_dir, data_list))

        self.training = training
        self.img_mean = img_mean
        self.img_std = img_std
        self.size = size
        self.random_crop = random_crop
        self.random_scale = random_scale
        self.random_mirror = random_mirror

    def _read_image_and_label_paths(self, data_dir, data_list):
        """Parse strings into lists of image, semantic label and
        instance label paths.

        Args:
        data_dir: A string indicates root directory of images and labels.
        data_list: A list of strings which indicate path of paired images
            and labels. 'image_path semantic_label_path instance_label_path'.

        Return:
        Threee lists of file paths.
        """
        images, semantic_labels, instance_labels, edges, grays = [], [], [], [], []
        with open(data_list, 'r') as list_file:
            for line in list_file:
                line = line.strip('\n')
                try:
                    img, semantic_lab, instance_lab, edge, gray = line.split(' ')
                except:
                    img, semantic_lab, instance_lab = line.split(' ')
                    edge = gray = None

                images.append(os.path.join(data_dir, img))

                if semantic_lab is not None:
                    semantic_labels.append(os.path.join(data_dir, semantic_lab))

                if instance_lab is not None:
                    instance_labels.append(os.path.join(data_dir, instance_lab))
                
                if edge is not None:
                    edges.append(os.path.join(data_dir, edge))
                if gray is not None:
                    grays.append(os.path.join(data_dir, gray))

        return images, semantic_labels, instance_labels, edges, grays

    def _get_datas_by_index(self, idx):
        image_path = self.image_paths[idx]
        image = self._read_image(image_path)

        if len(self.semantic_label_paths) > 0:
            semantic_label_path = self.semantic_label_paths[idx]
            semantic_label = self._read_label(semantic_label_path)
        else:
            semantic_label = None

        if len(self.instance_label_paths) > 0:
            instance_label_path = self.instance_label_paths[idx]
            instance_label = self._read_label(instance_label_path)
        else:
            instance_label = None

        if len(self.edge_label_paths) > 0:
            edge_label_path = self.edge_label_paths[idx]
            edge_label = self._read_label(edge_label_path) / 255
        else:
            edge_label = None
        
        if len(self.gray_image_paths) > 0:
            gray_image_path = self.gray_image_paths[idx]
            gray_image = self._read_label(gray_image_path) / 255
        else:
            gray_image = None
        
        if semantic_label is not None:
            cats = np.unique(semantic_label)
            semantic_tags = np.zeros((256, ), dtype=np.uint8)
            semantic_tags[cats] = 1
        else:
            semantic_tags = None
    
        return image, semantic_label, instance_label, edge_label, gray_image, semantic_tags

    def _training_preprocess(self, idx):
        """Data preprocessing for training.
        """
        assert(self.size is not None)
        image, semantic_label, instance_label, edge_label, gray_image, semantic_tags = self._get_datas_by_index(idx)
        mask = (semantic_label != 2).astype(np.uint8)
        edge_label = np.squeeze(edge_label)
        gray_image = np.squeeze(gray_image)
        mask = np.squeeze(mask)
        label = np.stack([semantic_label, instance_label, edge_label, gray_image, mask], axis=2)
        if self.random_mirror:
            is_flip = np.random.uniform(0, 1.0) >= 0.5
            if is_flip:
                image = image[:, ::-1, ...]
                label = label[:, ::-1, ...]

        if self.random_scale:
            image, label = transforms.random_resize(image, label, 0.5, 1.5)

        if self.random_crop:
            image, label = transforms.random_crop_with_pad(
                image, label, self.size, self.img_mean, 255)
        semantic_label, instance_label, edge_label, gray_image, mask =  \
                label[..., 0], label[..., 1], label[..., 2], label[..., 3], label[..., 4]

        return image, semantic_label.astype(np.int64), instance_label.astype(np.int64), edge_label, gray_image, semantic_tags

    def _eval_preprocess(self, idx):
        """Data preprocessing for evaluationg.
        """
        image, semantic_label, instance_label, edge_label, gray_image, semantic_tags = self._get_datas_by_index(idx)
        if self.size is not None:
            image = transforms.resize_with_pad(
                image, self.size, self.img_mean)

        #image = image[:self.size[0], :self.size[1], ...]

        return image, semantic_label, instance_label, edge_label, gray_image, semantic_tags

    def __getitem__(self, index):
        """Retrive image and label by index.
        """
        if self.training:
            image, semantic_label, instance_label, edge_label, gray_image, semantic_tags = self._training_preprocess(index)
        else:
            image, semantic_label, instance_label, edge_label, gray_image, semantic_tags = self._eval_preprocess(index)

        image = image - np.array(self.img_mean, dtype=image.dtype)
        image = image / np.array(self.img_std, dtype=image.dtype)

        inputs = {'image': image.transpose(2, 0, 1)}
        if self.training:
            labels = {'semantic_label': semantic_label,
                    'instance_label': instance_label,
                    'edge_label': edge_label,
                    'gray_image': gray_image,
                    'semantic_tag': semantic_tags}
        else:
            labels = {'semantic_label': semantic_label,
                    'instance_label': instance_label}
        return inputs, labels, index
    

    def __len__(self):
        """Total number of datas in the dataset.
        """
        return len(self.image_paths)