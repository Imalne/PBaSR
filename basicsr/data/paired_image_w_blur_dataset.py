import os
import cv2
import random
import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from .data_util import make_dataset
import glob, os

def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

@DATASET_REGISTRY.register()
class PairedImageBFDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageBFDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder, self.blur_mask_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_blur_mask']
                
        self.lq_paths = sorted(glob.glob(os.path.join(self.lq_folder,"*")))
        self.gt_paths = sorted(glob.glob(os.path.join(self.gt_folder,"*")))
        self.blur_mask_paths = sorted(glob.glob(os.path.join(self.blur_mask_folder,"*")))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        #  scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.
        lq_path = self.lq_paths[index]
        img_lq = cv2.imread(lq_path).astype(np.float32) / 255.
        blur_mask_path = self.blur_mask_paths[index]
        img_blur_mask = cv2.imread(blur_mask_path).astype(np.float32) / 255.

        # augmentation for training
        if self.opt['phase'] == 'train':
            input_gt_size = img_gt.shape[0]
            input_lq_size = img_lq.shape[0]
            scale = input_gt_size // input_lq_size
            gt_size = self.opt['gt_size']

            if self.opt['use_resize_crop']:
                # random resize
                input_gt_random_size = random.randint(gt_size, input_gt_size)
                input_gt_random_size = input_gt_random_size - input_gt_random_size % scale # make sure divisible by scale 
                resize_factor = input_gt_random_size / input_gt_size
                img_gt = random_resize(img_gt, resize_factor)
                img_lq = random_resize(img_lq, resize_factor)
                img_blur_mask = random_resize(img_blur_mask, resize_factor)

                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, input_gt_size // input_lq_size,
                                               gt_path)
                RuntimeError("Random crop with mask not implemented yet")

            # flip, rotation
            img_gt, img_blur_mask, img_lq = augment([img_gt, img_blur_mask, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        if self.opt['phase'] != 'train':
            crop_eval_size = self.opt.get('crop_eval_size', None)
            if crop_eval_size:
                input_gt_size = img_gt.shape[0]
                input_lq_size = img_lq.shape[0]
                scale = input_gt_size // input_lq_size
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, crop_eval_size, input_gt_size // input_lq_size,
                                               gt_path)
                RuntimeError("Random crop with mask not implemented yet")


        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_blur_mask, img_lq = img2tensor([img_gt, img_blur_mask, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'blur_mask': img_blur_mask,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.gt_paths)
