import os.path as osp

import albumentations as A
from albumentations.pytorch import ToTensorV2
from change_detection_pytorch.datasets.custom import CustomDataset


class SVCD_Dataset(CustomDataset):
    """ season-varying change detection dataset"""

    def __init__(self, img_dir, sub_dir_1='A', ann_dir=None, img_suffix='.png', seg_map_suffix='.png',
                 transform=None, split=None, data_root=None, test_mode=False, size=256, debug=False):
        super().__init__(img_dir, sub_dir_1, ann_dir, img_suffix, seg_map_suffix, transform, split,
                         data_root, test_mode, size, debug)

    def get_default_transform(self):
        """Set the default transformation."""

        default_transform = A.Compose([
            A.Resize(self.size, self.size),
            # A.HorizontalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ], additional_targets={'image'})
        return default_transform

    def get_test_transform(self):
        """Set the test transformation."""

        test_transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ], additional_targets={'image'})
        return test_transform

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if not self.ann_dir:
            ann = None
            img1, filename = self.prepare_img(idx)
            transformed_data = self.transform(image=img1)
            img1 = transformed_data['image']
            return img1, filename
        else:
            img1, ann, filename = self.prepare_img_ann(idx)
            transformed_data = self.transform(image=img1, mask=ann)
            img1, ann = transformed_data['image'], transformed_data['mask']
            return img1, ann, filename
