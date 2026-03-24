# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch import distributed as dist
from collections import OrderedDict
from tqdm import tqdm

from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn import functional as F
from basicsr.models.base_model import BaseModel
from basicsr.archs import build_network
from basicsr.utils import get_root_logger, tensor2img
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.logger import AverageMeter
from basicsr.archs.RAFT.raft import RAFT
from basicsr.utils.utils_tools import make_gif
from basicsr.archs.BSST_arch import flow_warp
import numpy as np
import cv2

#sdxl tile
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from CPC.cpc.utils.common import wavelet_reconstruction

def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    ph, pw = map(lambda x: (x + multiple - 1) // multiple * multiple - x, (h, w))
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)

def visualize_optical_flow(flow):

    flow = flow.cpu().numpy()

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='mean')

def get_motion_map(flow_forwards,flow_backwards):
    b,t,c,h,w = flow_forwards.shape
    motion_map = torch.cat([flow_backwards[:,0,...].reshape(b,1,2,h,w),flow_forwards.reshape(b,t,2,h,w)],dim=1)
    motion_max,_ = torch.max(motion_map.reshape(b,-1),dim=-1)
    motion_map_nl = (motion_map)/(motion_max).reshape(b,1,1,1,1)*2 - 1
    return motion_map_nl

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

@MODEL_REGISTRY.register()
class ModelBSST(BaseModel):
    """Base Deblur model for single image deblur."""
    def __init__(self, opt):
        super(ModelBSST, self).__init__(opt)

        self.fix_raft = RAFT().to(self.device)
        load_path_f = self.opt['path'].get('pretrain_network_f', None)
        if load_path_f is not None:
            self.load_network(self.fix_raft, load_path_f,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
        self.fix_raft.requires_grad_(False)

        self.net_g = build_network(opt["network_g"])
        load_path_g = self.opt['path'].get('pretrain_network_g', None)
        if load_path_g is not None:
            self.load_network(self.net_g, load_path_g,
                              self.opt['path'].get('strict_load_g', False), param_key=self.opt['path'].get('param_key', 'params'))
        self.net_g = self.model10_to_device(self.net_g)

        if self.is_train:
            self.init_training_settings()
        self.scaler = torch.cuda.amp.GradScaler()

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length = 610000
                # length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def _replace_unet_conv_in(self):
        # replace the first layer to accept 8 in_channels
        _weight = self.LDM.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.LDM.unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, 3, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight *= 0.33
        # new conv_in channel
        _n_convin_out_channel = self.LDM.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            12, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.LDM.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        #????
        self.LDM.unet.config["in_channels"] = 12
        logging.info("Unet config is updated")
        return

    def get_bi_flows(self,lq, mv_f=None, mv_b=None):
        """
        Args:
            lqs (tensor): Input low quality sequence with
                shape (n, t, c, h, w). 
                h and w need to be divisible by 8. 

        Returns:
            Tensor: Output bidirectional flows with shape (n, t-1, 2, h//4, w//4).
        """
        b,t,c,h,w = lq.shape
        iter = 20
        with torch.no_grad():
            lq1 = lq[:,:-1,...].reshape(b*(t-1),c,h,w)
            lq2 = lq[:,1:,...].reshape(b*(t-1),c,h,w)
            lq1 = F.interpolate(lq1, scale_factor=0.5, mode='bilinear')
            lq2 = F.interpolate(lq2, scale_factor=0.5, mode='bilinear')
            if mv_b is not None:
                mv_b = mv_b.reshape(b*(t-1),2,h//4,w//4)
                mv_b = F.interpolate(mv_b, scale_factor=0.25, mode='bilinear')*0.25
                iter = 4

            flows_backwards_pred = self.fix_raft(lq1,lq2, iters = iter, flow_init = mv_b).detach()

            flows_backwards_pred = F.interpolate(flows_backwards_pred, scale_factor=0.5, mode='bilinear').view(b,t-1,2,h//4,w//4) / 2
        return mv_f, flows_backwards_pred
    
    def get_bi_flow_loss(self, gt, flow_f, flow_b):
        b,t,c,h,w = gt.shape
        gt_0 = gt[:,:-1,...].reshape(b*(t-1),c,h,w)
        gt_1 = gt[:,1:,...].reshape(b*(t-1),c,h,w)

        #forword:
        flow_f = F.interpolate(flow_f.reshape(b*(t-1),2,h//4,w//4), scale_factor=4, mode='bilinear') * 4
        f_1 = flow_warp(gt_0, flow_f.permute(0,2,3,1))

        #backword:
        flow_b = F.interpolate(flow_b.reshape(b*(t-1),2,h//4,w//4), scale_factor=4, mode='bilinear') * 4
        f_0 = flow_warp(gt_1, flow_b.permute(0,2,3,1))

        bi_loss = mse_loss(gt_0, f_0) + mse_loss(f_1, gt_1) 
        return bi_loss

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.log_dict = OrderedDict()
        # define losses
        # self.loss = get_loss(loss_name=self.opt['loss'].get('name'), **self.opt['loss'].get('kwargs'))
        
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.log_dict['l_pix'] = AverageMeter()
            self.log_dict['l_flows'] = AverageMeter()
            # self.loss_flow = 
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            # to do
            pass
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')
        
        self.setup_optimizers()
        self.setup_schedulers()

    def model10_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """

        net = net.to(self.device)
        if self.opt['dist']:
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=False
                )
            net._set_static_graph()
            
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_flowrefine = []
        optim_params_softconv = []
        optim_params_convoffset = []
        logger = get_root_logger()
        # conv_offset
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if "blur_motion_refine" in k:
                    optim_params_flowrefine.append(v)
                    logger.warning(f"flowrefine lr {k}")
                elif 'conv_offset' in k:
                    optim_params_convoffset.append(v)
                    logger.warning(f"convoffset lr {k}")
                elif  'softsplit' in k or 'softcomp' in k:
                    optim_params_softconv.append(v)
                    logger.warning(f"softconv lr {k}")
                else:
                    optim_params.append(v)
            else:
                
                logger.warning(f'Params {k} will not be optimized.')

        flowrefine_ratio = self.opt['flowrefine_ratio']
        offset_ratio = self.opt['offset_ratio']
        softconv_ratio = self.opt['softconv_ratio']
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}, {'params': optim_params_flowrefine, 'lr': train_opt['optim_g']['lr'] * flowrefine_ratio},{'params': optim_params_convoffset, 'lr': train_opt['optim_g']['lr'] * offset_ratio},{'params': optim_params_softconv, 'lr': train_opt['optim_g']['lr'] * softconv_ratio}],
                                                **train_opt['optim_g'])

        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        

    def feed_data(self, data):
        lq, gt, mv_f, resi = data['lq'],data['gt'],data['mv_f'], data['resi']
        self.lq = lq.to(self.device)
        self.gt = gt.to(self.device)
        self.mv_f = mv_f.to(self.device)
        self.resi = resi.to(self.device)
    
    def feed_data_test(self,data):
        lq, gt, mv_f, resi = data['lq'],data['gt'],data['mv_f'], data['resi']

        self.lq = lq.to(self.device).unsqueeze(0)
        self.gt = gt.to(self.device).unsqueeze(0)
        self.mv_f = mv_f.to(self.device).unsqueeze(0)
        self.resi = resi.to(self.device).unsqueeze(0).unsqueeze(2)
    
    def forward(self, lq, mv_b, mv_f, resi):

        b,t,c,h,w = lq.shape
        mv_b_rs = F.interpolate(mv_b.reshape(b*(t-1),2,h,w), scale_factor=0.25, mode='bilinear').view(b,t-1,2,h//4,w//4) / 4
        mv_f_rs = F.interpolate(mv_f.reshape(b*(t-1),2,h,w), scale_factor=0.25, mode='bilinear').view(b,t-1,2,h//4,w//4) / 4
        flows_forwards_raft, flows_backwards_raft = self.get_bi_flows(lq, mv_f_rs, mv_b_rs)

        lq = lq.half()
        with torch.cuda.amp.autocast():
            output = self.net_g(lq, flows_forwards_raft, flows_backwards_raft, resi)

        return output

    def optimize_parameters(self, current_iter):

        b,t,c,h,w = self.lq.shape
        mv_f_rs = F.interpolate(self.mv_f.reshape(b*(t-1),2,h,w), scale_factor=0.25, mode='bilinear').view(b,t-1,2,h//4,w//4) / 4
        mv_b_rs = -flow_warp(mv_f_rs.view(b*(t-1),2,h//4,w//4), mv_f_rs.view(b*(t-1),2,h//4,w//4).permute(0,2,3,1)).view(b,t-1,2,h//4,w//4)
        flows_forwards_raft, flows_backwards_raft = self.get_bi_flows(self.lq, mv_f_rs, mv_b_rs)

        f_b10 = F.interpolate(flows_backwards_raft[:,0,:,:,:], scale_factor=4, mode='bilinear').view(b,2,h,w) * 4
        resi0 = flow_warp(self.resi[:,0,:,:,:], f_b10.permute(0,2,3,1)).reshape(b,1,1,h,w)
        resi_patch = torch.cat((resi0, self.resi), dim = 1)

        self.lq = self.lq.half()
        with torch.cuda.amp.autocast():
            output = self.net_g(self.lq, flows_forwards_raft, flows_backwards_raft, resi_patch)

            self.output = output
            loss_dict = OrderedDict()
            l_pix = self.cri_pix(output, self.gt)
            loss_dict['l_pix'] = l_pix
            l_total = l_pix

        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer_g)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.scaler.step(self.optimizer_g)
        self.scaler.update()
        
        for k,v in self.reduce_loss_dict(loss_dict).items():
            self.log_dict[k].update(v)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    def generation_tile(self, current_iter, seq_num, folder):
        self.cldm.eval()

        diff_bs = 8
        b,t,c,h,w = self.lq.shape
        t_list = list(range(0, t*b, diff_bs))
        prompts = []
        prompt = "high-quality, clear, fine details, sharp edges, realistic textures, detailed enhancement, high resolution, high fidelity, fine-grained details."
        for i in range(diff_bs):
            prompts.append(prompt) 
        
        with torch.no_grad():
            output = torch.zeros_like(self.lq)
            mv_in = torch.cat((self.mv_f[:,0,:,:,:].unsqueeze(1), self.mv_f), dim = 1)
            resi_in = torch.cat((self.resi[:,0,:,:,:].unsqueeze(1), self.resi), dim = 1)
            mv_in = mv_in.reshape(b*t,2,h,w).contiguous().float()
            mv_in = mv_in/mv_in.max()
            resi_in = resi_in.reshape(b*t,1,h,w).contiguous().float().abs()
            cp = torch.cat((mv_in, resi_in), dim = 1)
        
            for i in t_list:

                cp_i = cp[i:i+diff_bs,:,:,:]
                cp_rs = F.interpolate(cp_i, scale_factor=2, mode='bilinear', align_corners=True).reshape(b,diff_bs//b,3,h*2,w*2)
                img = self.output.reshape(b*t,3,h,w).float()[i:i+diff_bs,:,:,:]
                img = torch.clip(img, min=0, max=1)
                cond_img = (img * 2 - 1).to(self.device)
                cond_img = F.interpolate(cond_img, scale_factor=2, mode='bilinear', align_corners=True)
                cond_img = pad_to_multiples_of(cond_img, multiple=8)
                cond = self.cldm.prepare_condition(
                    cond_img,
                    prompts,
                    True,
                    512,
                )
                h1, w1 = cond["c_img"].shape[2:]

                z = self.sampler.sample(
                    model=self.cldm,
                    device=self.device,
                    steps=50,
                    x_size=(diff_bs,4,h1,w1),
                    cond=cond,
                    cp=cp_rs,
                    uncond=None,
                    cfg_scale=1.0,
                    tiled=True,
                    tile_size= 512//8,
                    tile_stride= 256//8,
                    # progress=accelerator.is_main_process,
                )

                z = z[..., :h1, :w1]
                x = self.cldm.vae_decode(
                        z,
                        True,
                        512 // 8,
                    )
                x = F.interpolate(
                    wavelet_reconstruction((x + 1) / 2, (cond_img + 1) / 2),
                    size=(h,w),
                    mode="bicubic",
                    antialias=True,
                )
                x = x.reshape(b,diff_bs,3,h,w)
                output[:,i:i+diff_bs,:,:,:] = x[:,:,:,:,:]
                
                #save img
                for n in range(diff_bs):

                    out_tensor = x[0, n, :, :, :]
                    out_img = tensor2img([out_tensor])  # uint8, bgr

                    seq = folder.rsplit('_', 1)[0]
                    num = seq_num + n + i + 52
                    visual_dir_a = os.path.join('gopro','BSST', seq)
                    os.makedirs(visual_dir_a, exist_ok=True)
                    output_path_a = os.path.join(visual_dir_a, f"{num:05d}" + '.png')
                    cv2.imwrite(output_path_a, out_img)

        self.output_a = output[:, :, :, :, :]

    def generation_tile_wo_rs(self, current_iter, seq_num, folder):
        self.cldm.eval()

        diff_bs = 6
        b,t,c,h,w = self.lq.shape
        t_list = list(range(0, t*b, diff_bs))
        prompts = []
        prompt = "high-quality, clear, fine details, sharp edges, realistic textures, detailed enhancement, high resolution, high fidelity, fine-grained details."
        for i in range(diff_bs):
            prompts.append(prompt) 
        
        with torch.no_grad():
            output = torch.zeros_like(self.lq)
            mv_in = torch.cat((self.mv_f[:,0,:,:,:].unsqueeze(1), self.mv_f), dim = 1)
            resi_in = torch.cat((self.resi[:,0,:,:,:].unsqueeze(1), self.resi), dim = 1)
            mv_in = mv_in.reshape(b*t,2,h,w).contiguous().float()
            mv_in = mv_in/mv_in.abs().max()
            resi_in = resi_in.reshape(b*t,1,h,w).contiguous().float().abs()
            cp = torch.cat((mv_in, resi_in), dim = 1)
        
            for i in t_list:

                cp_i = cp[i:i+diff_bs,:,:,:]
                cp_rs = F.interpolate(cp_i, scale_factor=1, mode='bilinear', align_corners=True).reshape(b,diff_bs//b,3,h,w)
                img = self.output.reshape(b*t,3,h,w).float()[i:i+diff_bs,:,:,:]
                img = torch.clip(img, min=0, max=1)
                cond_img = (img * 2 - 1).to(self.device)
                # cond_img = F.interpolate(cond_img, scale_factor=2, mode='bilinear', align_corners=True)
                cond_img = pad_to_multiples_of(cond_img, multiple=8)
                cond = self.cldm.prepare_condition(cond_img,prompts,)
                h1, w1 = cond["c_img"].shape[2:]

                z = self.sampler.sample(
                    model=self.cldm,
                    device=self.device,
                    steps=50,#
                    x_size=(diff_bs,4,h1,w1),
                    cond=cond,
                    cp=cp_rs,
                    uncond=None,
                    cfg_scale=1.0,
                    tiled=True,
                    tile_size= 512//8,
                    tile_stride= 256//8,
                    # progress=accelerator.is_main_process,
                )

                z = z[..., :h1, :w1]
                x = self.cldm.vae_decode(z)
                # x = F.interpolate(
                #     wavelet_reconstruction((x + 1) / 2, (cond_img + 1) / 2),
                #     size=(h,w),
                #     mode="bicubic",
                #     antialias=True,
                # )
                x = wavelet_reconstruction((x + 1) / 2, (cond_img + 1) / 2)
                x = x.reshape(b,diff_bs,3,h,w)
                output[:,i:i+diff_bs,:,:,:] = x[:,:,:,:,:]
                
                #save img
                for n in range(diff_bs):

                    out_tensor = x[0, n, :, :, :]
                    out_img = tensor2img([out_tensor])  # uint8, bgr

                    seq = folder.rsplit('_', 1)[0]
                    num = seq_num + n + i
                    visual_dir_a = os.path.join('dvd_rebuttal','265_32', seq)
                    os.makedirs(visual_dir_a, exist_ok=True)
                    output_path_a = os.path.join(visual_dir_a, f"{num:05d}" + '.png')
                    cv2.imwrite(output_path_a, out_img)

        self.output_a = output[:, :, :, :, :]


    def generation(self, current_iter, seq_num, folder):
        self.cldm.eval()

        diff_bs = 8
        b,t,c,h,w = self.lq.shape
        t_list = list(range(0, t*b, diff_bs))
        prompts = []
        prompt = "high-quality, clear, fine details, sharp edges, realistic textures, detailed enhancement, high resolution, high fidelity, fine-grained details."
        for i in range(diff_bs):
            prompts.append(prompt) 
        
        with torch.no_grad():
            output = torch.zeros_like(self.lq)

            mv_in = torch.cat((self.mv_f[:,0,:,:,:].unsqueeze(1), self.mv_f), dim = 1)
            resi_in = torch.cat((self.resi[:,0,:,:,:].unsqueeze(1), self.resi), dim = 1)
            mv_in = mv_in.reshape(b*t,2,h,w).contiguous().float()
            mv_in = mv_in/mv_in.max()
            resi_in = resi_in.reshape(b*t,1,h,w).contiguous().float().abs()
            cp = torch.cat((mv_in, resi_in), dim = 1)


            seq = folder.rsplit('_', 1)[0]

        
            for i in t_list:

                cp_i = cp[i:i+diff_bs,:,:,:]
                cp_rs = F.interpolate(cp_i, size=(512,512), mode='bilinear', align_corners=True).reshape(b,diff_bs//b,3,512,512)
                img = self.output.reshape(b*t,3,h,w).float()[i:i+diff_bs,:,:,:]
                img = torch.clip(img, min=0, max=1)
                img = F.interpolate(img, size=(512,512), mode='bilinear', align_corners=True)
                cond_img = (img * 2 - 1).to(self.device)
                cond_img = pad_to_multiples_of(cond_img, multiple=8)
                
                dict_input = {
                    "img": img,
                    "bsz": img.shape[0],
                    "folder": seq,
                    "name": f'{i:02d}',
                }

                cond = self.cldm.prepare_condition(cond_img, prompts)
                h1, w1 = cond["c_img"].shape[2:]


                z = self.sampler.sample(
                    model=self.cldm,
                    device=self.device,
                    steps=50,
                    x_size=(diff_bs,4,h1,w1),
                    cond=cond,
                    cp=cp_rs,
                    uncond=None,
                    cfg_scale=1.0,
                    tiled=False,
                    dict_input=dict_input,
                )

                z = z[..., :h1, :w1]
                x = self.cldm.vae_decode(z)
                x = F.interpolate(
                    wavelet_reconstruction((x + 1) / 2, (cond_img + 1) / 2),
                    size=(h,w),
                    mode="bicubic",
                    antialias=True,
                )
                x = x.reshape(b,diff_bs,3,h,w)
                output[:,i:i+diff_bs,:,:,:] = x[:,:,:,:,:]
                
                #save img
                # for n in range(diff_bs):

                #     out_tensor = x[0, n, :, :, :]
                #     out_img = tensor2img([out_tensor])  # uint8, bgr

                #     seq = folder.rsplit('_', 1)[0]
                #     num = seq_num + n + i
                #     visual_dir_a = os.path.join('dvd','stage2', seq)
                #     os.makedirs(visual_dir_a, exist_ok=True)
                #     output_path_a = os.path.join(visual_dir_a, f"{num:05d}" + '.png')
                #     cv2.imwrite(output_path_a, out_img)

        self.output_a = output[:, :, :, :, :]

    def test_by_patch(self, current_iter):
        self.net_g.eval()
        self.fix_raft.eval()
        lq = self.lq
        check = 0

        b,t,c,h,w = self.lq.shape
        flows_forwards_all = F.interpolate(self.mv_f.reshape(b*(t-1),2,h,w), scale_factor=0.25, mode='bilinear').view(b,t-1,2,h//4,w//4) / 4
        flows_backwards_all = -flow_warp(flows_forwards_all.view(b*(t-1),2,h//4,w//4), flows_forwards_all.view(b*(t-1),2,h//4,w//4).permute(0,2,3,1)).view(b,t-1,2,h//4,w//4)

        with torch.no_grad():
            size_patch_testing = 256
            overlap_size = 64
            b,t,c,h,w = lq.shape
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
            w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
            E = torch.zeros(b, t, c, h, w)
            W = torch.zeros_like(E)
            E_a = torch.zeros(b, t, c, h, w)
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    with torch.cuda.amp.autocast():
                        in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                        
                        flows_forwards_patch = flows_forwards_all[..., (h_idx)//4:h_idx//4+size_patch_testing//4, w_idx//4:w_idx//4+size_patch_testing//4]
                        flows_backwards_patch = flows_backwards_all[..., (h_idx)//4:h_idx//4+size_patch_testing//4, w_idx//4:w_idx//4+size_patch_testing//4]
                        
                        # flows_forwards_refine, flows_backwards_refine = flows_forwards_patch, flows_backwards_patch
                        flows_forwards_refine, flows_backwards_refine = self.get_bi_flows(in_patch, flows_forwards_patch, flows_backwards_patch)
                        
                        resi_patch = self.resi[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                        f_b10 = F.interpolate(flows_backwards_refine[:,0,:,:,:], scale_factor=4, mode='bilinear').view(1,2,size_patch_testing,size_patch_testing) * 4
                        resi0 = flow_warp(resi_patch[:,0,:,:,:], f_b10.permute(0,2,3,1)).reshape(b,1,1,size_patch_testing,size_patch_testing)
                        resi_patch = torch.cat((resi0, resi_patch), dim = 1)

                        out_patch = self.net_g(in_patch, flows_forwards_refine, flows_backwards_refine, resi_patch)

                    out_patch = out_patch.detach().cpu().reshape(b,t,c,size_patch_testing,size_patch_testing)

                    out_patch_mask = torch.ones_like(out_patch)

                    if True:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -overlap_size//2:, :] *= 0
                            out_patch_mask[..., -overlap_size//2:, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -overlap_size//2:] *= 0
                            out_patch_mask[..., :, -overlap_size//2:] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., :overlap_size//2, :] *= 0
                            out_patch_mask[..., :overlap_size//2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, :overlap_size//2] *= 0
                            out_patch_mask[..., :, :overlap_size//2] *= 0

                    E[..., h_idx:(h_idx+size_patch_testing), w_idx:(w_idx+size_patch_testing)].add_(out_patch)
                    W[..., h_idx:(h_idx+size_patch_testing), w_idx:(w_idx+size_patch_testing)].add_(out_patch_mask)
            output = E.div_(W)
        self.output = output[:, :, :, :, :]
        self.net_g.train()
        self.fix_raft.train()

    def validation(self, dataloader, current_iter, tb_logger,wandb_logger=None,save_img=False):
        """Validation function.
        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            wandb_loggger (wandb logger): wandb runer logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, tb_logger, wandb_logger, save_img,rgb2bgr=True, use_image=True)
        else:
            self.dist_validation(dataloader, current_iter, tb_logger, wandb_logger, save_img,rgb2bgr=True, use_image=True)

    def dist_validation(self, dataloader, current_iter, tb_logger,wandb_logger, save_img, rgb2bgr=True, use_image=True):
        dataset = dataloader.dataset
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {}
            for folder,seq_index in dataset.splite_seqs_index.items():
                if seq_index["seq_index"][0] == 52:
                    self.metric_results[folder] = torch.zeros(
                        8, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
                elif seq_index["seq_index"][0] == 102:
                    self.metric_results[folder] = torch.zeros(
                        14, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
                elif seq_index["seq_index"][0] == 86:
                    self.metric_results[folder] = torch.zeros(
                        len(seq_index["seq_index"]) - 6, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
                elif seq_index["seq_index"][0] == 29:
                    self.metric_results[folder] = torch.zeros(
                        len(seq_index["seq_index"]) - 19, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
                else:
                    self.metric_results[folder] = torch.zeros(
                            len(seq_index["seq_index"]) - 4, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
        self._initialize_best_metric_results(dataset_name)
            
        rank, world_size = get_dist_info()
        num_seq = len(dataset)
        num_pad = (world_size - (num_seq % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='image')
        metric_data_05 = dict()
        metric_data_a = dict()
        metric_data = dict()
        for i in range(rank, num_seq + num_pad, world_size):
            idx_data = min(i,num_seq - 1)
            # print(idx_data)
            val_data = dataset[idx_data]
            folder = val_data["seq_name"]
            seq_index = val_data["seq"]
            self.feed_data_test(val_data)
            #一阶段
            self.test_by_patch(current_iter)
            # self.output = self.lq
            self.output_a = torch.zeros_like(self.output)
            #二阶段
            self.generation_tile_wo_rs(current_iter, seq_index[0], folder)
            # self.generation_tile(current_iter, seq_index[0], folder)

            visuals = self.get_current_visuals()
            del self.lq
            del self.output
            del self.gt
            del self.output_a

            if seq_index[0] == 52:
                seq_num = seq_index[0] + 48 - 10
                visuals['lq'] = visuals['lq'][:,-10:-2,...]
                visuals['result'] = visuals['result'][:,-10:-2,...]
                visuals['result_a'] = visuals['result_a'][:,-10:-2,...]
                visuals['gt'] = visuals['gt'][:,-10:-2,...]
            elif seq_index[0] == 86:
                seq_num = seq_index[0] + 4
                visuals['lq'] = visuals['lq'][:,4:-2,...]
                visuals['result'] = visuals['result'][:,4:-2,...]
                visuals['result_a'] = visuals['result_a'][:,4:-2,...]
                visuals['gt'] = visuals['gt'][:,4:-2,...]
            elif seq_index[0] == 29:
                seq_num = seq_index[0] + 17
                visuals['lq'] = visuals['lq'][:,17:-2,...]
                visuals['result'] = visuals['result'][:,17:-2,...]
                visuals['result_a'] = visuals['result_a'][:,17:-2,...]
                visuals['gt'] = visuals['gt'][:,17:-2,...]
            elif seq_index[0] == 102:
                seq_num = seq_index[0] + len(seq_index) - 16
                visuals['lq'] = visuals['lq'][:,-16:-2,...]
                visuals['result'] = visuals['result'][:,-16:-2,...]
                visuals['result_a'] = visuals['result_a'][:,-16:-2,...]
                visuals['gt'] = visuals['gt'][:,-16:-2,...]
            else:
                seq_num = seq_index[0] + 2
                visuals['lq'] = visuals['lq'][:,2:-2,...]
                visuals['result'] = visuals['result'][:,2:-2,...]
                visuals['result_a'] = visuals['result_a'][:,2:-2,...]
                visuals['gt'] = visuals['gt'][:,2:-2,...]
            torch.cuda.empty_cache()
            if i < num_seq:
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_a = visuals['result_a'][0, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    result_img_a = tensor2img([result_a]) 
                    metric_data['img'] = result_img
                    metric_data_a['img'] = result_img_a
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        metric_data['img2'] = gt_img
                        metric_data_a['img2'] = gt_img
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            # if metric_idx == 0:
                            #     result_a = calculate_metric(metric_data_a, opt_)
                            result = calculate_metric(metric_data_a, opt_)
                            self.metric_results[folder][idx, metric_idx] += result
                    
                    #save img
                    # seq = folder.rsplit('_', 1)[0]
                    # num = seq_num + idx
                    # visual_dir = os.path.join('dvd_rebuttal','266_42_stage1', seq)
                    # os.makedirs(visual_dir, exist_ok=True)
                    # output_path_a = os.path.join(visual_dir, f"{num:05d}"+ '.png')
                    # cv2.imwrite(output_path_a, result_img)
                    
                    # visual_dir = os.path.join('gopro','gt','test')
                    # os.makedirs(visual_dir, exist_ok=True)
                    # output_path = os.path.join(visual_dir, seq + '_' + str(num)+ '.png')
                    # cv2.imwrite(output_path, gt_img)

                if rank == 0:
                    for _ in range(world_size):
                        
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()
        if with_metrics:
            if self.opt['dist']:
                
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                
                dist.barrier()
                
            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger,wandb_logger)
                
        out_metric = 0.
        for name in self.metric_results.keys():
            out_metric = self.metric_results[name]
        
        return out_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger,wandb_logger):
        metric_results_avg = {
            folder: torch.mean(tensor, dim=0).cpu()
            for (folder, tensor) in self.metric_results.items()
        }
        total_avg_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        for folder, value in self.metric_results.items():
            
            for idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += metric_results_avg[folder][idx].item()
           
        for metric in total_avg_results.keys():
            total_avg_results[metric] /= len(metric_results_avg)
            # update the best metric result
            self._update_best_metric_result(dataset_name, metric, total_avg_results[metric], current_iter)
        log_str = f'Validation {dataset_name},\t'
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f'\t # {metric}: {value:.4f}'
            for folder, tensor in metric_results_avg.items():
                log_str += f'\t # {folder}: {tensor[metric_idx].item():.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in total_avg_results.items():
                tb_logger.add_scalar(f'{dataset_name}/metrics/{metric}', value, current_iter)
                if wandb_logger is not None:
                    wandb_logger.log({f'{dataset_name}/metrics/{metric}':value},current_iter)
                

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['result_a'] = self.output_a.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
    
    def save_checkpoint(self, current_iter, ckpt_name, save_train_state):
        ckpt_dir = os.path.join('z_ckpt_512_lq', str(current_iter))
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        self.LDM.unet.save_pretrained(unet_path, safe_serialization=False)
        logging.info(f"UNet is saved to: {unet_path}")


    def save(self, epoch, current_iter):
        # self.save_checkpoint(current_iter, ckpt_name="latest", save_train_state=True)
        self.save_network(self.net_g, 'net_g', current_iter)
        # self.save_network(self.f_mask, 'net_m', current_iter)
        # self.save_network(self.fix_raft,'net_flow', current_iter)
        # self.save_training_state(epoch, current_iter)
