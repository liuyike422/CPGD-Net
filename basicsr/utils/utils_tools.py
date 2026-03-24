import numpy as np
import imageio
import cv2
import torch
import inspect

import torch.nn as nn
from os import path as osp
import torch.nn.functional as F
from basicsr.data.transforms import mod_crop
from basicsr.utils import img2tensor, scandir


def getBackFromMV(mv_fs):

    def _getback(mv_f):

        mv_f = torch.tensor(mv_f).unsqueeze(0)
        mv_b = -flow_warp(mv_f, mv_f).squeeze(0).numpy()

        return mv_b

    if isinstance(mv_fs, list):
        return [_getback(mv_f) for mv_f in mv_fs]
    else:
        return _getback(mv_fs)


def recoverMV(mvs):
    
    def _recover(mv):
        #266
        # mv = np.repeat(mv, 4, axis = 1)
        # mv = np.repeat(mv, 4, axis = 2)
        # mv = mv.astype('float32') /16
        #265/264
        mv = np.repeat(mv, 8, axis = 1)
        mv = np.repeat(mv, 8, axis = 2)
        mv = mv.astype('float32') /4
        return mv

    if isinstance(mvs, list):
        return [_recover(mv) for mv in mvs]
    else:
        return _recover(mvs)

def mv2tensor(mvs, float32=True):

    def _totensor(mv, float32):

        mv = torch.from_numpy(mv)
        if float32:
            mv = mv.float()
        return mv

    if isinstance(mvs, list):
        return [_totensor(mv, float32) for mv in mvs]
    else:
        return _totensor(mvs, float32)

def make_gif(img1, img2, name, duration):   

    with imageio.get_writer(name+'.gif', mode='I', duration = duration,loop = 0) as writer:
        writer.append_data(img1.astype(np.uint8))
        writer.append_data(img2.astype(np.uint8))

def flow_warp(x, flow12, pad="border", mode="bilinear"):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if "align_corners" in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()
    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2

def load_mv_seq(path):

    if isinstance(path, list):
        mv_paths = path
    else:
        mv_paths = sorted(list(scandir(path, full_path=True)))
    mv_fs = [np.load(v) for v in mv_paths]
    mv_fs = recoverMV(mv_fs)

    mv_fs = mv2tensor(mv_fs)
    mv_fs = torch.stack(mv_fs, dim=0)

    return mv_fs

def load_resi_seq(path):

    if isinstance(path, list):
        resi_paths = path
    else:
        resi_paths = sorted(list(scandir(path, full_path=True)))
    #265/264
    resis = [(np.load(v).astype('float32')+1)/255. for v in resi_paths]
    #266
    # resis = [(np.load(v).astype('float32')+1)/512. for v in resi_paths]
    
    resis = mv2tensor(resis)
    resis = torch.stack(resis, dim=0)

    return resis