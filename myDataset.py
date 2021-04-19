import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, img_dir_view2,mask_dir_view2, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

        self.img_dir_view2 = img_dir_view2
        self.mask_dir_view2 = mask_dir_view2


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img1 = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        img1 = img1.astype('float32') / 255
        img1 = img1.transpose(2, 0, 1)

        img2 = cv2.imread(os.path.join(self.img_dir_view2, img_id + self.img_ext))
        img2 = img2.astype('float32') / 255
        img2 = img2.transpose(2, 0, 1)
        
        mask1 = []
        for i in range(self.num_classes):
            mask1.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask1 = np.dstack(mask1)

        mask2 = []
        for i in range(self.num_classes):
        	mask2.append(cv2.imread(os.path.join(self.mask_dir_view2, str(i),
        	            img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask2 = np.dstack(mask2)        

        #if self.transform is not None:
        #    augmented = self.transform(image=img, mask=mask)
        #    img = augmented['image']
        #    mask = augmented['mask']
                
        mask1 = mask1.astype('float32') / 255
        mask1 = mask1.transpose(2, 0, 1)

        mask2 = mask2.astype('float32') / 255
        mask2 = mask2.transpose(2, 0, 1)

        return img1, img2, mask1, mask2, {'img_id': img_id}
