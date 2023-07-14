import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.build import META_ARCH_REGISTRY
from collections import OrderedDict
from utils import dist_utils
from utils.build import build_backbone, build_loss
import logging


logger = logging.getLogger("simdeblur")

class BaseArch():
    """
    The base architecture of different model architectures.
    The classes inherit this base class are used to adapt different inputs, losses, etc.
    """
    def __init__(self) -> None:
        self.model = None
        self.criterion = None

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError

    def update_params(self):
        """
        This is a the key training loop of SimDeblur, indicating the training strategy of each model.
        """
        raise NotImplementedError

    def __call__(self, *args):
        return self.model(*args)

    def load_ckpt(self, ckpt, **kwargs):
        """
        Args:
            ckpt: a parameter dict
        """
        model_ckpt = ckpt["model"]
        new_model_ckpt = OrderedDict()
        # strip `module.` prefix
        for k, v in model_ckpt.items():
            name = k[7:] if k.startswith("module.") else k
            new_model_ckpt[name] = v
        model_ckpt = new_model_ckpt

        if dist_utils.get_world_size() > 1:
            self.model.module.load_state_dict(model_ckpt, **kwargs)
        else:
            self.model.load_state_dict(model_ckpt, **kwargs)
        logger.info("Checkponit loaded to model successfully!")

    def generate_ckpt(self):
        """
        generate a dict containing model's parameters to be saved
        """
        if dist_utils.get_world_size() > 1:
            model_ckpt = self.model.module.state_dict()
        else:
            model_ckpt = self.model.state_dict()
        return {"model": model_ckpt}

    def build_model(self, cfg):
        """
        build a backbone model
        TODO: re-write these by dist_utils.py
        """
        model = build_backbone(cfg.model)
        if not cfg.args.get("gpus"):
            return model
        if cfg.args.gpus > 1:
            rank = cfg.args.local_rank
            model = nn.parallel.DistributedDataParallel(
                model.cuda(), device_ids=[rank], output_device=rank)
        if cfg.args.local_rank == 0:
            logger.info("Model:\n{}".format(model))
            print_model_params(model)

        return model

    @classmethod
    def build_losses(cls, loss_cfg):
        """
        build all losses and reture a loss dict
        """
        criterion_cfg = loss_cfg.get("criterion")
        weights_cfg = loss_cfg.get("weights")

        criterions = OrderedDict()
        weights = OrderedDict()
        if isinstance(criterion_cfg, list):
            assert len(criterion_cfg) == len(
                weights_cfg), "The length of criterions and weights in config file should be same!"
            for loss_item, loss_weight in zip(criterion_cfg, weights_cfg):
                criterions[loss_item.name] = build_loss(loss_item)
                weights[loss_item.name] = loss_weight
        else:
            criterions[criterion_cfg.name] = build_loss(criterion_cfg)
            weights[criterion_cfg.name] = 1.0
        if dist_utils.get_local_rank() == 0:
            logger.info("Loss items: ")
            for k in criterions.keys():
                logger.info(f"    {k}, weight: {weights[k]}")

        return criterions, weights

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()
    
    # def inference(self, x):
    #     if dist_utils.get_world_size() > 1:
    #         return self.model.module.inference(x)
    #     else:
    #         return self.model.inference(x)

@META_ARCH_REGISTRY.register()
class SingleScalePlainCNN(BaseArch):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.build_model(cfg).to(self.device)
        self.criterion, self.criterion_weights = self.build_losses(cfg.loss)
        self.criterion = {k: v.to(self.device) for k, v in self.criterion.items()}

    def preprocess(self, batch_data):
        """
        prepare for input, different model archs needs different inputs.
        """
        return batch_data["input_frames"].to(self.device)

    def postprocess(self, outputs):
        """
        transfer the outputs with 5 dims into 4 dims by flatten the batch and number frames.
        """
        if outputs.dim() == 5:
            return outputs.flatten(0, 1)
        return outputs

    def update_params(self, batch_data, optimizer):
        # forward to generate model results
        model_outputs = self.model(self.preprocess(batch_data))
        # 1 calculate losses
        gt_frames = batch_data["gt_frames"].to(self.device).flatten(0, 1)
        if model_outputs.dim() == 5:
            model_outputs = model_outputs.flatten(0, 1)  # (b*n, c, h, w)
        loss = self.calculate_loss(
            self.criterion, self.criterion_weights, gt_frames, model_outputs)

        # 2 optimize model parameters: a) zero_grad, b) backward, c) update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "results": model_outputs,
            "loss": {"loss": loss}
        }

    @classmethod
    def calculate_loss(cls, criterion, weights, gt_data, output_data):
        loss = 0.
        for key, cri in criterion.items():
            loss += cri(gt_data, output_data) * weights[key]
        return loss


def print_model_params(model: nn.Module):
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for param in model.parameters():
        mul_value = np.prod(param.size())
        total_params += mul_value
        if param.requires_grad:
            trainable_params += mul_value
        else:
            non_trainable_params += mul_value

    total_params /= 1e6
    trainable_params /= 1e6
    non_trainable_params /= 1e6

    loger = logging.getLogger(name="SimDeblur")
    loger.info(f'Total params: {total_params} M.')
    loger.info(f'Trainable params: {trainable_params} M.')
    loger.info(f'Non-trainable params: {non_trainable_params} M.')