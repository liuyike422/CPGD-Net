import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
import os

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.utils_tools import make_gif, flow_warp, mv2tensor

@DATASET_REGISTRY.register()
class DeblurRecurrentDataset(data.Dataset):
    """Debblur dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(DeblurRecurrentDataset, self).__init__()
        self.opt = opt
        self.type = opt['DataType'] 
        if self.type == 'BSD':
            self.data_root = opt['dataroot']
        else:
            self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
            self.MVs_root, self.resi_root = Path(opt['dataroot_MVs']), Path(opt['dataroot_resi'])
        
        self.num_frame = opt['num_frame']   
        self.file_end = opt["file_end"]
        self.cache_data = opt["cache_data"]
        self.keys = []
        self.max_frames = {}
        self.data_infos = []

        if self.type== 'BSD':
            for dirs in os.listdir(self.data_root):
                self.data_infos.append(
                            dict(
                                folder=dirs,
                                sequence_length=100
                            )
                        )
        else:
            with open(opt['meta_info_file'], 'r') as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(' ')
                    self.max_frames[folder] = int(frame_num)
                    self.keys.extend([f'{folder}/{i:05d}' for i in range(int(frame_num))])
                    self.data_infos.append(
                        dict(
                            folder=folder,
                            sequence_length=int(frame_num)
                        )
                    )


        # remove the video clips used in validation
        """ if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition] """

        # file client (io backend)
        self.file_client = None

        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']
        
        self.using_MVs = opt['load_MVs']
        self.using_Resi = opt['load_resi']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        index = index%len(self.data_infos)
        clip_name = self.data_infos[index]["folder"]
        max_frame = self.data_infos[index]['sequence_length']

        interval = random.choice(self.interval_list)

        start_frame_idx = np.random.randint(0, max_frame - self.num_frame * interval + 1)
        end_frame_idx = start_frame_idx + self.num_frame * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        if self.type == 'BSD':
            self.lq_root = os.path.join(self.data_root, clip_name, 'Blur/RGB')
            self.gt_root = os.path.join(self.data_root, clip_name, 'Sharp/RGB')
            self.MVs_root = os.path.join(self.data_root, clip_name, 'Blur/MVs')
            self.resi_root = os.path.join(self.data_root, clip_name, 'Blur/resi')

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        f_mvs = []
        b_mvs = []
        resis = []
        for neighbor in neighbor_list:
            if self.type == 'BSD':
                img_lq_path = os.path.join(self.lq_root, f'{neighbor:08d}.{self.file_end}')
                img_gt_path = os.path.join(self.gt_root, f'{neighbor:08d}.{self.file_end}')
            else:
                if self.is_lmdb:
                    img_lq_path = f'{clip_name}/{neighbor:05d}'
                    img_gt_path = f'{clip_name}/{neighbor:05d}'
                else:
                    img_lq_path = self.lq_root / clip_name / f'{neighbor:05d}.{self.file_end}'
                    img_gt_path = self.gt_root / clip_name / f'{neighbor:05d}.{self.file_end}'

            #get MVs and resi
            if self.using_MVs and neighbor != neighbor_list[-1]:
                tmp_str = neighbor
                tmp_end = neighbor + 1
                if self.type == 'BSD':
                    mv_name = clip_name + "_" + f'{tmp_str:08d}_{tmp_end:08d}.npy'
                    mv_path = os.path.join(self.MVs_root, mv_name)
                else:
                    mv_name = clip_name + "_" + f'{tmp_str:05d}_{tmp_end:05d}.npy'
                    mv_path = self.MVs_root / clip_name / mv_name
                mv_in = np.load(mv_path).astype('float32')   #MV的保存尺寸为图片尺寸1/4，1/4
                mv_full = np.repeat(mv_in, 4, axis = 1)
                mv_full = np.repeat(mv_full, 4, axis = 2)
                mv_full = mv_full/16
                # mv_f = torch.tensor(mv_full).unsqueeze(0)
                # mv_b = -flow_warp(mv_f, mv_f).squeeze(0).numpy()
                f_mvs.append(mv_full)
                # b_mvs.append(mv_b)
            if self.using_Resi and neighbor != neighbor_list[-1]:
                tmp_str = neighbor
                tmp_end = neighbor + 1
                if self.type == 'BSD':
                    resi_name = clip_name + "_" + f'{tmp_str:08d}_{tmp_end:08d}.npy'
                    resi_path = os.path.join(self.resi_root, resi_name)
                else:
                    resi_name = clip_name + "_" + f'{tmp_str:05d}_{tmp_end:05d}.npy'
                    resi_path = self.resi_root / clip_name / resi_name
                resi = (np.load(resi_path).astype('float32')+1)/512. #(h,w) 残差范围-2^(10-1)到2^(10-1)-1 变换为-1/1
                resis.append(resi)

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

            # make_gif(img_lq*255, img_gt*255, 'lq_hq', 1)

        #check make gifs
        # resi1 = torch.tensor(resi_1s[1]).unsqueeze(0).unsqueeze(0)
        # mv_b0 = torch.tensor(b_mvs[0]).unsqueeze(0)
        # resi0 = flow_warp(resi1, mv_b0).squeeze(0).squeeze(0).numpy()

        # resis = []
        # resis.append(resi0)
        # # resi_1s.insert(0, "resi0")
        # for i in range(len(resi_1s)):
        #     resis.append(resi_1s[i])
        # img1f = img1f_t.squeeze(0).permute(1,2,0).numpy()

        # randomly crop
        img_gts, img_lqs, f_mvs, resis = paired_random_crop(img_gts, img_lqs, gt_size, scale, f_mvs, resis, img_gt_path)
    
        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results, f_mvs, resis  = augment(img_lqs, f_mvs, resis, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        resis = mv2tensor(resis)
        resis = torch.stack(resis, dim=0)
        resis = resis.unsqueeze(1)

        f_mvs = mv2tensor(f_mvs)
        # b_mvs = mv2tensor(b_mvs)
        f_mvs = torch.stack(f_mvs, dim=0)
        # b_mvs = torch.stack(b_mvs, dim=0)   

        return {'lq': img_lqs, 'gt': img_gts, 'mv_f': f_mvs, 'mv_b': b_mvs, 'resi': resis, 'folder': [f"{clip_name}.{neighbor_list[0]}",f"{clip_name}.{neighbor_list[1]}"]}

    def __len__(self):
        return len(self.data_infos)*10000


@DATASET_REGISTRY.register()
class DeblurRecurrentDatasetloadmemory(data.Dataset):
    """Debblur dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super( DeblurRecurrentDatasetloadmemory, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']
        self.file_end = opt["file_end"]
        
        self.keys = []
        self.max_frames = {}
        self.data_infos = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.max_frames[folder] = int(frame_num)
                self.keys.extend([f'{folder}/{i:05d}' for i in range(int(frame_num))])
                self.data_infos.append(
                    dict(
                        folder=folder,
                        sequence_length=int(frame_num)
                    )
                )
        
        
        
            


        # remove the video clips used in validation
        """ if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition] """

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        self.gt_cache_images = {}
        self.lq_cache_images = {}
        for info_data in self.data_infos:
            folder = info_data['folder']
            sequence_length = info_data['sequence_length']
            self.lq_cache_images[folder] = []
            self.gt_cache_images[folder] = []
            for neighbor in range(sequence_length):
                img_lq_path = self.lq_root / folder / f'{neighbor:05d}.{self.file_end}'
                img_gt_path = self.gt_root / folder / f'{neighbor:05d}.{self.file_end}'

                img_bytes = self.file_client.get(img_lq_path, 'lq')
                img_lq = imfrombytes(img_bytes, float32=True)

                img_bytes = self.file_client.get(img_gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)

                self.lq_cache_images[folder] += [img_lq]
                self.gt_cache_images[folder] += [img_gt]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        # key = self.keys[index]
        clip_name = self.data_infos[index]["folder"]
        max_frame = self.data_infos[index]['sequence_length']

        
        # clip_name, frame_name = key.split('/')  # key example: 000/00000000
        # max_frame = self.max_frames[clip_name]
        # print(max_frame)
        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        # start_frame_idx = int(frame_name)
        # if start_frame_idx > max_frame - self.num_frame * interval:
            # start_frame_idx = random.randint(0, max_frame - self.num_frame * interval)
        
        start_frame_idx = np.random.randint(0, max_frame - self.num_frame * interval + 1)
        end_frame_idx = start_frame_idx + self.num_frame * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            """ if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:05d}'
                img_gt_path = f'{clip_name}/{neighbor:05d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:05d}.{self.file_end}'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:05d}.{self.file_end}' """
            img_gt_path = self.gt_root / clip_name / f'{neighbor:05d}.{self.file_end}'

            # get LQ
            img_lqs = self.lq_cache_images[clip_name][neighbor_list]
            

            # get GT
            img_gts = self.gt_cache_images[clip_name][neighbor_list]


        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        # return {'lq': img_lqs, 'gt': img_gts, 'key': key}
        return {'lq': img_lqs, 'gt': img_gts, 'folder': [f"{clip_name}.{neighbor_list[0]}",f"{clip_name}.{neighbor_list[1]}"]}

    def __len__(self):
        return len(self.data_infos)