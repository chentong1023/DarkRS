import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .forward_warp_gaussian import ForwardWarp as ForwardWarp
from .Unet import UNet2 as UNet
from .raft import RAFT
from utils.build import BACKBONE_REGISTRY


def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

    gridX = torch.tensor(gridX, requires_grad=False,).cuda()
    gridY = torch.tensor(gridY, requires_grad=False,).cuda()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2*(x/W - 0.5)
    y = 2*(y/H - 0.5)
    # stacking X and Y
    grid = torch.stack((x,y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid)

    return imgOut

class SmallMaskNet(nn.Module):
    """A three-layer network for predicting mask"""
    def __init__(self, input, output):
        super(SmallMaskNet, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, output, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.conv3(x)
        return x

@BACKBONE_REGISTRY.register()
class QVInet(nn.Module):
    """The quadratic model"""
    def __init__(self, args, ro_rate, cam_pos=0.5):
        super().__init__()
        self.flownet = torch.nn.DataParallel(RAFT(args))
        self.fwarp = ForwardWarp()
        self.fwarp2 = ForwardWarp()
        self.reflownet = UNet(9, 4)
        self.refinenet = UNet(8, 3)
        # self.masknet = SmallMaskNet(38, 1)
        self.ro_rate = ro_rate
        self.pos = self.ro_rate * cam_pos

        self.flownet.load_state_dict(torch.load(args.model))
        self.flownet = self.flownet.module

    def forward(self, x):
        """
        args:
            x: shape(B, N, C, H, W)
        return:
            corrected image: (B, 3, H, W),
            correction flow: (B, 2, H, W)
        """
        B, N, C, H, W = x.shape

        flows = []
        t = []
        for i in range(N):
            _, flow = self.flownet(x[:,N//2,...], x[:,i,...], iters=2, test_mode = True)
            flows.append(flow)
            t_i = i - N //2 + self.ro_rate / H * flow[:,1,...]
            t.append(t_i)
        t_int = self.pos - self.ro_rate / H * torch.arange(H).view(-1, 1).expand(B, 1, H, W).cuda()
        flow_0t = torch.zeros_like(flows[0])
        for i in range(N):
            prod = torch.ones((B, 1, H, W)).cuda()
            for j in range(N):
                if i == j:
                    continue
                prod *= (t_int - t[j]) / (t[i] - t[j])
            flow_0t += flows[i] * prod.expand_as(flows[i])

        # Flow Reversal
        flow_t0, norm1 = self.fwarp(flow_0t, flow_0t)
        flow_t0 = -flow_t0

        flow_t0[norm1 > 0] = flow_t0[norm1 > 0]/norm1[norm1>0].clone()

        It_warp = backwarp(x[:,N//2,...], flow_t0)

        time_map = torch.arange(H).view(-1, 1).expand(B, 1, H, W).cuda() / H * 2 - 1

        output, _ = self.reflownet(torch.cat([x[:,N//2,...], It_warp, flow_t0, time_map], dim=1))

	# Adaptive filtering
        Ftr = backwarp(flow_t0, 10*torch.tanh(output[:, 2:4])) + output[:, :2]

        It_f = backwarp(x[:,N//2,...], Ftr)

        # res = self.refinenet(torch.cat([x[:,N//2,...], It_f, Ftr], dim=1))

        # It_warp_f = torch.clamp(It_f + res, 0, 1)

    #     M = torch.sigmoid(self.masknet(torch.cat([I1tf, I2tf, feature], dim=1))).repeat(1, 3, 1, 1)

    #     It_warp = ((1-t) * M * I1tf + t * (1 - M) * I2tf) / ((1-t) * M + t * (1-M)).clone()

        return [It_f], [Ftr]

@BACKBONE_REGISTRY.register()
class QVInet2(nn.Module):
    """The quadratic model"""
    def __init__(self, args, ro_rate, cam_pos=0.5):
        super().__init__()
        self.flownet = torch.nn.DataParallel(RAFT(args))
        self.fwarp = ForwardWarp()
        self.fwarp2 = ForwardWarp()
        self.reflownet = UNet(9, 4)
        self.refinenet = UNet(8, 3)
        # self.masknet = SmallMaskNet(38, 1)
        self.ro_rate = ro_rate
        self.pos = self.ro_rate * cam_pos

        self.flownet.load_state_dict(torch.load(args.model))
        self.flownet = self.flownet.module

    def forward(self, x):
        """
        args:
            x: shape(B, N, C, H, W)
        return:
            corrected image: (B, 3, H, W),
            correction flow: (B, 2, H, W)
        """
        B, N, C, H, W = x.shape

        # Flow Reversal
        flows = []
        t = []
        for i in range(N):
            _, flow = self.flownet(x[:,N//2,...], x[:,i,...], iters=2, test_mode = True)
            flows.append(flow)
            t_i = i - N //2 + self.ro_rate / H * flow[:,1,...]
            t.append(t_i)
        t_int = self.pos - self.ro_rate / H * torch.arange(H).view(-1, 1).expand(B, 1, H, W).cuda()
        flow_0t = torch.zeros_like(flows[0])
        for i in range(N):
            prod = torch.ones((B, 1, H, W)).cuda()
            for j in range(N):
                if i == j:
                    continue
                prod *= (t_int - t[j]) / (t[i] - t[j])
            flow_0t += flows[i] * prod.expand_as(flows[i])

        flow_t0, norm1 = self.fwarp(flow_0t, flow_0t)
        flow_t0 = -flow_t0

        flow_t0[norm1 > 0] = flow_t0[norm1 > 0]/norm1[norm1>0].clone()

        It_warp = backwarp(x[:,N//2,...], flow_t0)

        time_map = torch.arange(H).view(-1, 1).expand(B, 1, H, W).cuda() / H * 2 - 1

        output, _ = self.reflownet(torch.cat([x[:,N//2,...], It_warp, flow_t0, time_map], dim=1))

	# Adaptive filtering
        Ftr = backwarp(flow_t0, 10*torch.tanh(output[:, 2:4])) + output[:, :2]

        It_f = backwarp(x[:,N//2,...], Ftr)

        res, _ = self.refinenet(torch.cat([x[:,N//2,...], It_f, Ftr], dim=1))

        It_warp_f = torch.clamp(It_f + res, 0, 1)

        return [It_warp_f], [Ftr]

@BACKBONE_REGISTRY.register()
class QVInet3(nn.Module):
    """The quadratic model"""
    def __init__(self, args, ro_rate, cam_pos=0.5):
        super().__init__()
        self.flownet = torch.nn.DataParallel(RAFT(args))
        self.fwarp = ForwardWarp()
        self.fwarp2 = ForwardWarp()
        self.reflownet = UNet(9, 4)
        self.refinenet = UNet(8, 3)
        # self.masknet = SmallMaskNet(38, 1)
        self.ro_rate = ro_rate
        self.pos = self.ro_rate * cam_pos

        self.flownet.load_state_dict(torch.load(args.model))
        self.flownet = self.flownet.module

    def forward(self, x):
        """
        args:
            x: shape(B, N, C, H, W)
        return:
            corrected image: (B, 3, H, W),
            correction flow: (B, 2, H, W)
        """
        B, N, C, H, W = x.shape

        # Flow Reversal
        flows = []
        t = []
        for i in range(N):
            _, flow = self.flownet(x[:,N//2,...], x[:,i,...], iters=2, test_mode = True)
            flows.append(flow)
            t_i = i - N //2 + self.ro_rate / H * flow[:,1,...]
            t.append(t_i)
        t_int = self.pos - self.ro_rate / H * torch.arange(H).view(-1, 1).expand(B, 1, H, W).cuda()
        flow_0t = torch.zeros_like(flows[0])
        for i in range(N):
            prod = torch.ones((B, 1, H, W)).cuda()
            for j in range(N):
                if i == j:
                    continue
                prod *= (t_int - t[j]) / (t[i] - t[j])
            flow_0t += flows[i] * prod.expand_as(flows[i])

        flow_t0, norm1 = self.fwarp(flow_0t, flow_0t)
        flow_t0 = -flow_t0
        flow_t0[norm1 > 0] = flow_t0[norm1 > 0]/norm1[norm1>0].clone()
        It_warp = backwarp(x[:,N//2,...], flow_t0)

        time_map = torch.arange(H).view(-1, 1).expand(B, 1, H, W).cuda() / H * 2 - 1
        output, _ = self.reflownet(torch.cat([x[:,N//2,...], It_warp, flow_t0, time_map], dim=1))
	# Adaptive filtering
        Ftr = backwarp(flow_t0, 10*torch.tanh(output[:, 2:4])) + output[:, :2]
        It_f = backwarp(x[:,N//2,...], Ftr)
        res, _ = self.refinenet(torch.cat([x[:,N//2,...], It_f, Ftr], dim=1))
        It_warp_f = torch.clamp(It_f + res, 0, 1)

        return [It_warp_f], [Ftr]
