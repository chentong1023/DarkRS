"""
The losses colleted from Deep Shutter Unrolling Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, vgg16
import numpy as np

from torchvision import models

from utils.build import LOSS_REGISTRY

from model.raft import RAFT

@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight=1.0, eps=1e-6):
        """
        the original eps is 1e-12
        """
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return torch.mean(torch.sqrt((pred - target)**2 + self.eps)) / target.shape[0]


@LOSS_REGISTRY.register()
class PerceptualLossVGG19(nn.Module):
    def __init__(self, layer_idx=[2, 7, 14], layer_weights=[1, 0.2, 0.04], reduction="mean"):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_weights = layer_weights
        self.vggnet_feats_layers = vgg19(pretrained=True).features

        self.reduction = reduction

    def vgg_forward(self, img):
        selected_feats = []
        out = img
        self.vggnet_feats_layers = self.vggnet_feats_layers.to(img)
        for i, layer in enumerate(self.vggnet_feats_layers):
            out = layer(out)
            if i in self.layer_idx:
                selected_feats.append(out)
            if i == self.layer_idx[-1]:
                break
        assert len(selected_feats) == len(self.layer_idx)
        return selected_feats

    def forward(self, img1, img2):
        selected_feats1 = self.vgg_forward(img1)
        selected_feats2 = self.vgg_forward(img2)

        loss = 0
        for i, (feat1, feat2) in enumerate(zip(selected_feats1, selected_feats2)):
            assert feat1.shape == feat2.shape, "The input tensor should be in same shape!"
            loss += F.mse_loss(feat1, feat2, reduction=self.reduction) * self.layer_weights[i]

        return loss
    
class Grid_gradient_central_diff():
    def __init__(self, nc, padding=True, diagonal=False):
        self.conv_x = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_y = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_xy = None
        if diagonal:
            self.conv_xy = nn.Conv2d(
                nc, nc, kernel_size=2, stride=1, bias=False)

        self.padding = None
        if padding:
            self.padding = nn.ReplicationPad2d([0, 1, 0, 1])

        fx = torch.zeros(nc, nc, 2, 2).float().cuda()
        fy = torch.zeros(nc, nc, 2, 2).float().cuda()
        if diagonal:
            fxy = torch.zeros(nc, nc, 2, 2).float().cuda()

        fx_ = torch.tensor([[1, -1], [0, 0]]).cuda()
        fy_ = torch.tensor([[1, 0], [-1, 0]]).cuda()
        if diagonal:
            fxy_ = torch.tensor([[1, 0], [0, -1]]).cuda()

        for i in range(nc):
            fx[i, i, :, :] = fx_
            fy[i, i, :, :] = fy_
            if diagonal:
                fxy[i, i, :, :] = fxy_

        self.conv_x.weight = nn.Parameter(fx)
        self.conv_y.weight = nn.Parameter(fy)
        if diagonal:
            self.conv_xy.weight = nn.Parameter(fxy)

    def __call__(self, grid_2d):
        _image = grid_2d
        if self.padding is not None:
            _image = self.padding(_image)
        dx = self.conv_x(_image)
        dy = self.conv_y(_image)

        if self.conv_xy is not None:
            dxy = self.conv_xy(_image)
            return dx, dy, dxy
        return dx, dy


@LOSS_REGISTRY.register()
class DsunPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
        self.model = self.contentFunc()

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def forward(self, fakeIm, realIm):
        f_fake = self.model(fakeIm)
        f_real = self.model(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


@LOSS_REGISTRY.register()
class DSUNL1Loss(nn.Module):
    def __init__(self):
        super(DSUNL1Loss, self).__init__()

    def forward(self, output, target, weight=None, mean=False):
        error = torch.abs(output - target)
        if weight is not None:
            error = error * weight.float()
            if mean is not False:
                return error.sum() / weight.float().sum()
        if mean is not False:
            return error.mean()
        return error.sum()


@LOSS_REGISTRY.register()
class VariationLoss(nn.Module):
    def __init__(self, nc, grad_fn=Grid_gradient_central_diff, mean=True):
        super(VariationLoss, self).__init__()
        self.grad_fn = grad_fn(nc)
        self.mean = mean

    def forward(self, image, weight=None):
        dx, dy = self.grad_fn(image)
        variation = dx**2 + dy**2

        if weight is not None:
            variation = variation * weight.float()
            if self.mean is not False:
                return variation.sum() / weight.sum()
        if self.mean is not False:
            return variation.mean()
        return variation.sum()

def get_robust_weight(flow_pred, flow_gt, beta):
    epe = ((flow_pred.detach() - flow_gt) ** 2).sum(dim=1, keepdim=True) ** 0.5
    robust_weight = torch.exp(-beta * epe)
    return robust_weight

@LOSS_REGISTRY.register()
class FlowDistillationLoss(nn.Module):
    def __init__(self, args):
        super(FlowDistillationLoss, self).__init__()
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model))
        self.eps = 1e-6
        self.loss = torch.nn.SmoothL1Loss()

        self.model = self.model.module
        self.model.to("cuda")
        self.model.eval()

    def forward(self, gt_img, image, flow):
        B, N, C, H, W = image.shape
        _, flow_gt = self.model(gt_img, image[:,N//2,...], iters=2, test_mode = True)
        weight = get_robust_weight(flow_gt, flow, beta=0.3)
        diff = flow - flow_gt
        alpha = weight / 2
        epsilon = 10 ** (-(10 * weight - 1) / 3)
        loss = ((diff ** 2 + epsilon ** 2) ** alpha).mean()
        return loss


class Ternary(nn.Module):
    def __init__(self, patch_size=7):
        super(Ternary, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().cuda()

    def transform(self, tensor):
        tensor_ = tensor.mean(dim=1, keepdim=True)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size//2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_norm = loc_diff / torch.sqrt(0.81 + loc_diff ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size//2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask
        
    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y.detach()
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()
        return loss


class Geometry(nn.Module):
    def __init__(self, patch_size=3):
        super(Geometry, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().cuda()

    def transform(self, tensor):
        b, c, h, w = tensor.size()
        tensor_ = tensor.reshape(b*c, 1, h, w)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size//2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_ = loc_diff.reshape(b, c*(self.patch_size**2), h, w)
        loc_diff_norm = loc_diff_ / torch.sqrt(0.81 + loc_diff_ ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size//2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()
        return loss


class Charbonnier_L1(nn.Module):
    def __init__(self):
        super(Charbonnier_L1, self).__init__()

    def forward(self, diff, mask=None):
        if mask is None:
            loss = ((diff ** 2 + 1e-6) ** 0.5).mean()
        else:
            loss = (((diff ** 2 + 1e-6) ** 0.5) * mask).mean() / (mask.mean() + 1e-9)
        return loss

class Charbonnier_Ada(nn.Module):
    def __init__(self):
        super(Charbonnier_Ada, self).__init__()

    def forward(self, diff, weight):
        alpha = weight / 2
        epsilon = 10 ** (-(10 * weight - 1) / 3)
        loss = ((diff ** 2 + epsilon ** 2) ** alpha).mean()
        return loss

@LOSS_REGISTRY.register()
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
    
    def forward(self, imgt_pred, imgt):
        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)
        return loss_rec

@LOSS_REGISTRY.register()
class GeometryLoss(nn.Module):
    def __init__(self):
        super(GeometryLoss, self).__init__()
        self.gc_loss = Geometry(3)
    
    def forward(self, feat_pred, feat_gt):
        return self.gc_loss(feat_pred, feat_gt)

def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)
@LOSS_REGISTRY.register()
class VIFlowDistillationLoss(nn.Module):
    def __init__(self, args):
        super(VIFlowDistillationLoss, self).__init__()
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model))
        self.eps = 1e-6
        self.loss = torch.nn.SmoothL1Loss()

        self.model = self.model.module
        self.model.to("cuda")
        self.model.eval()
        self.rb_loss = Charbonnier_Ada()

    def forward(self, imgt, image, flow_0, flow_1):
        B, N, C, H, W = image.shape
        img0 = image[:,0,...]
        img1 = image[:,1,...]
        _, flow0_1 = self.model(imgt, img0, iters=2, test_mode=True)
        _, flow1_1 = self.model(imgt, img1, iters=2, test_mode=True)
        robust_weight0 = get_robust_weight(flow_0[0], flow0_1, beta=0.3)
        robust_weight1 = get_robust_weight(flow_1[0], flow1_1, beta=0.3)
        loss_dis = 0.01 * (self.rb_loss(2.0 * resize(flow_0[1], 2.0) - flow0_1, weight=robust_weight0) + self.rb_loss(2.0 * resize(flow_1[1], 2.0) - flow1_1, weight=robust_weight1))
        loss_dis += 0.01 * (self.rb_loss(4.0 * resize(flow_0[2], 4.0) - flow0_1, weight=robust_weight0) + self.rb_loss(4.0 * resize(flow_1[2], 4.0) - flow1_1, weight=robust_weight1))
        loss_dis += 0.01 * (self.rb_loss(8.0 * resize(flow_0[3], 8.0) - flow0_1, weight=robust_weight0) + self.rb_loss(8.0 * resize(flow_1[3], 8.0) - flow1_1, weight=robust_weight1))
        return loss_dis

@LOSS_REGISTRY.register()
class ZeroShotDenoiseLoss(nn.Module):
    def __init__(self):
        super(ZeroShotDenoiseLoss, self).__init__()

    def forward(self, x):
        pass