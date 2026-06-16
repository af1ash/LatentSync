import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.optim import AdamW
import torch.optim as optim
import itertools
from .warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from .IFNet_HDv3 import *
import torch.nn.functional as F
from .loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        self.version = 4.25
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))), False)
            else:
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location ='cpu')), False)
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [16, 8, 4, 2, 1]
        flow, mask, merged = self.flownet(torch.cat((imgs, gt), 1), scale=scale, training=training)
        loss_l1 = (merged[-1] - gt).abs().mean()
        loss_smooth = self.sobel(flow[-1], flow[-1]*0).mean()
        # loss_vgg = self.vgg(merged[-1], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_cons + loss_smooth * 0.1
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[-1], {
            'mask': mask,
            'flow': flow[-1][:, :2],
            'loss_l1': loss_l1,
            'loss_cons': loss_cons,
            'loss_smooth': loss_smooth,
            }


from .pytorch_msssim import ssim_matlab
class RIFE:

    def __init__(self, 
        model_dir, multi=2, exp=1, ext='mp4', 
        scale=1.0, 
        usefp16=False
    ):
        """
        scale in [0.25. 0.5, 1.0, 2.0, 4.0]
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if(usefp16):
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.fp16 = usefp16
        self.model = Model()
        if not hasattr(self.model, 'version'):
            self.model.version = 0
        self.model.load_model(model_dir, -1)
        # print("Loaded 3.x/4.x HD model.")
        self.model.eval()
        self.model.device()
        self.scale = scale
        self.multi = multi
        if exp != 1:
            self.multi = (2 ** exp)
    

    def make_inference(self, I0, I1, n):
        if self.model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(self.model.inference(I0, I1, (i+1) * 1. / (n+1), self.scale))
            return res
        else:
            middle = self.model.inference(I0, I1, self.scale)
            if n == 1:
                return [middle]
            first_half = self.make_inference(I0, middle, n=n//2)
            second_half = self.make_inference(middle, I1, n=n//2)
            if n%2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]

    def pad_image(self, img, padding):

        # img = torch.from_numpy(np.transpose(img, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        if(self.fp16):
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)

    def normalize_input_img(self, frames):
        for lastframe in frames:
            if list(lastframe.shape)[-1] == 4:
                lastframe = lastframe[:, :, :3]
            lastframe = cv2.cvtColor(lastframe, cv2.COLOR_RGB2BGR)
            yield lastframe


    def inference(self, frames):
        nor_frames = self.normalize_input_img(frames)
        lastframe = next(nor_frames)
        h, w, _ = lastframe.shape

        tmp = max(128, int(128 / self.scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = self.pad_image(I1, padding)
        temp = None # save lastframe when processing static frame

        break_flag = False
        while True:
            if temp is not None:
                frame = temp
                temp = None
            else:
                try:
                    frame = next(nor_frames)
                except StopIteration:
                    break_flag = True
                    break
            # if frame is None:
            #     break
            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = self.pad_image(I1, padding)
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            if ssim > 0.996:
                try:
                    frame = next(nor_frames)# read a new frame
                except StopIteration:
                    break
                if frame is None:
                    break_flag = True
                    frame = lastframe
                else:
                    temp = frame
                I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
                I1 = self.pad_image(I1, padding)
                I1 = self.model.inference(I0, I1, scale=self.scale)
                I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
                ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        
            if ssim < 0.2:
                output = []
                for i in range(self.multi - 1):
                    output.append(I0)
                '''
                output = []
                step = 1 / args.multi
                alpha = 0
                for i in range(args.multi - 1):
                    alpha += step
                    beta = 1-alpha
                    output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
                '''
            else:
                output = self.make_inference(I0, I1, self.multi - 1)

            if False and self.montage:
                yield np.concatenate((lastframe, lastframe), 1)
                for mid in output:
                    mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                    yield np.concatenate((lastframe, mid[:h, :w]), 1)
            else:
                yield cv2.cvtColor(lastframe, cv2.COLOR_BGR2RGB) 
                for mid in output:
                    mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                    yield cv2.cvtColor(mid[:h, :w], cv2.COLOR_BGR2RGB)

            lastframe = frame
            if break_flag:
                break
