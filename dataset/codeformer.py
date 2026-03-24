from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random
import os

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data

from .degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
    add_gaussian_noise,
    add_jpg_compression,
)
from .utils import load_file_list, center_crop_arr, random_crop_arr
from ..utils.common import instantiate_from_config


class VideoDeblurDataset(data.Dataset):
  
    def __init__(
        self,
        data_info: str,
        file_path: str,
        crop_size: int,
        num_frame: int,
    ) -> "CodeformerDataset":
        super(VideoDeblurDataset, self).__init__()
        self.file_paths = file_path
        self.num_frame = num_frame
        # random.shuffle(self.file_paths)
        self.crop_size = crop_size
        self.data_info = []
        with open(data_info, 'r') as fin:
                for line in fin:
                    folder, frame_num = line.replace('\n','').split('/')
                    self.data_info.append(
                        dict(
                            folder=folder,
                            sequence_length=int(frame_num)
                        )
                    )

    def __len__(self):
        return len(self.data_info)

    def random_crop(self, lq, gt, mv, resi, crop_size):

        crop_y = random.randrange(lq.shape[0] - crop_size + 1)
        crop_x = random.randrange(lq.shape[1] - crop_size + 1)

        crop_lq = lq[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size, :]
        crop_gt = gt[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size, :]
        crop_mv = mv[:,crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
        crop_resi = resi[:,crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

        return crop_lq, crop_gt, crop_mv, crop_resi

    def __getitem__(self, idx):
        
        clip_name = self.data_info[idx]["folder"]
        max_frame = self.data_info[idx]['sequence_length']

        start_frame_idx = np.random.randint(1, max_frame - self.num_frame)
        end_frame_idx = start_frame_idx + self.num_frame
        neighbor_list = list(range(start_frame_idx, end_frame_idx))

        #lq_params
        kernel = random_mixed_kernels(
            ['iso', 'aniso'],
            [0.5, 0.5],
            41,
            [0.1, 12],
            [0.1, 12],
            [-math.pi, math.pi],
            noise_range=None,
        ) #blur kernal
        sigma = np.random.uniform(0, 15)#noise range
        scale = np.random.uniform(1, 2)
        quality = np.random.uniform(40, 60)
        
        if np.random.uniform() < 0.5:
            prompt = "high-quality, clear, fine details, sharp edges, realistic textures, detailed enhancement, high resolution, high fidelity, fine-grained details."
        else:
            prompt = ''

        lqs = []
        gts = []
        mvs = []
        resis = []
        prompts = []
        
        #load
        for neighbor in neighbor_list:
        
            img_root = os.path.join(self.file_paths, clip_name, 'img_pairs', clip_name + '_' + f'{neighbor:05d}.npy')
            mv_root = os.path.join(self.file_paths, clip_name, 'mv', clip_name + '_' + f'{neighbor:05d}.npy')
            resi_root = os.path.join(self.file_paths, clip_name, 'resi', clip_name + '_' + f'{neighbor:05d}.npy')

            input = np.load(img_root)
            gt = input[:,:,:3]
            lq = input[:,:,3:]
            lq = lq.astype(np.float32)/255
            gt = gt.astype(np.float32)/255
            # check = np.concatenate((gt,lq),axis = 1)
            h,w,_ = input.shape
            
            mv = np.load(mv_root).astype('float32')   #MV的保存尺寸为图片尺寸1/4，1/4
            mv = np.repeat(mv, 4, axis = 1)
            mv = np.repeat(mv, 4, axis = 2)
            mv = mv/16

            resi = (np.load(resi_root).astype('float32'))/512. #(h,w) 残差范围-2^(10-1)到2^(10-1)-1 变换为-1/1
            resi = np.expand_dims(resi, axis=0)
            # ------------------------ generate lq image ------------------------ #
            
            # lq = cv2.filter2D(lq, -1, kernel)  # blur
            # lq = cv2.resize(
            #     lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR
            # )# downsample
            # lq = add_gaussian_noise(lq, sigma)# noise
            # lq = add_jpg_compression(lq, int(quality))# jpeg compression
            # lq = cv2.resize(lq, (w, h), interpolation=cv2.INTER_LINEAR)# resize to original size

            # check_2 = np.concatenate((check,lq),axis = 1)
            # if(neighbor == 9):
            #     cv2.imwrite('./generate_lq/' + clip_name + '_lq.png', check_2*255)
            
            lq_rs, gt_rs, mv_rs, resi_rs = self.random_crop(lq, gt, mv, resi, self.crop_size)
        
            lq_rs = (lq_rs * 2 - 1).astype(np.float32)
            gt_rs = (gt_rs * 2 - 1).astype(np.float32)

            lqs.append(lq_rs)
            gts.append(gt_rs)
            mvs.append(mv_rs)
            resis.append(resi_rs)
            # prompts.append(prompt)

        #check
        # cv2.imwrite(clip_name + '_lq.png', lq*255)

        lqs = np.stack(lqs, axis=0)
        gts = np.stack(gts, axis=0)
        mvs = np.stack(mvs, axis=0)
        resis = np.stack(resis, axis=0)
        
        return gts, lqs, mvs, resis, prompt

class CodeformerDataset(data.Dataset):

    def __init__(
        self,
        file_list: str,
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        crop_type: str,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        self.file_list = file_list
        self.image_files = load_file_list(file_list)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def load_gt_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        img_gt = None
        while img_gt is None:
            # load meta file
            image_file = self.image_files[index]
            gt_path = image_file["image_path"]
            prompt = image_file["prompt"]
            img_gt = self.load_gt_image(gt_path)
            if img_gt is None:
                print(f"filed to load {gt_path}, try another image")
                index = random.randint(0, len(self) - 1)

        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        h, w, _ = img_gt.shape
        if np.random.uniform() < 0.5:
            prompt = ""

        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            noise_range=None,
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(
            img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR
        )
        # noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, [-1, 1]
        gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        lq = img_lq[..., ::-1].astype(np.float32)

        return gt, lq, prompt

    def __len__(self) -> int:
        return len(self.image_files)
