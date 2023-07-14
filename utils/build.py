import copy
import torch
from .registry import Registry

BACKBONE_REGISTRY = Registry("backbone")
LOSS_REGISTRY = Registry("loss")
META_ARCH_REGISTRY = Registry("meta_arch")


def build(cfg, registry, args=None):
    """
    Build the module with cfg.
    Args:
        cfg (dict): the config of the modules
        registry(Registry): A registry the module belongs to.

    Returns:
        The built module.
    """
    args = copy.deepcopy(cfg)
    name = args.pop("name")
    ret = registry.get(name)(**args)
    return ret


def build_backbone(cfg):
    return build(cfg, BACKBONE_REGISTRY)


def build_loss(cfg):
    """
    build a loss from the loss config
    """
    return build(cfg, LOSS_REGISTRY)


def build_meta_arch(cfg):
    name = cfg.meta_arch
    ret = META_ARCH_REGISTRY.get(name)(cfg)
    return ret


def list_backbones(name=None):
    """
    List all available backbones.
    Args:
        name: (TODO) list specific models corresponds to a given name.
    """
    return list(BACKBONE_REGISTRY._obj_map.keys())


def list_meta_archs(name=None):
    """
    List all available meta model architectures
    Args:
        name: (TODO) list specific archs corresponds to a given name.
    """
    return list(META_ARCH_REGISTRY._obj_map.keys())


def list_losses(name=None):
    """
    List all available losses
    Args:
        name: (TODO) list specific losses corresponds to a given name.
    """
    return list(LOSS_REGISTRY._obj_map.keys())

from torch.optim import Adam, SGD, AdamW
OPTIMIZER_REGISTRY = Registry("optimizer")
OPTIMIZER_REGISTRY.register(Adam)
OPTIMIZER_REGISTRY.register(SGD)
OPTIMIZER_REGISTRY.register(AdamW)

from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR, ExponentialLR
LR_SCHEDULER_REGISTRY = Registry("lr_scheduler")
LR_SCHEDULER_REGISTRY.register(MultiStepLR)
LR_SCHEDULER_REGISTRY.register(CosineAnnealingLR)
LR_SCHEDULER_REGISTRY.register(LambdaLR)
LR_SCHEDULER_REGISTRY.register(ExponentialLR)


def build_optimizer(cfg, model) -> torch.optim.Optimizer:
    optimizer = OPTIMIZER_REGISTRY.get(cfg.pop("name"))(model.parameters(), **cfg)
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    lr_scheduler = LR_SCHEDULER_REGISTRY.get(cfg.pop("name"))(optimizer, **cfg)
    return lr_scheduler

DATASET_REGISTRY = Registry("dataset")

def build_dataset(cfg):
    """
    Build the module with cfg.
    Args:
        cfg (dict): the config of the modules

    Returns:
        The built module.
    """
    args = cfg
    name = args.get("name")
    dataset = DATASET_REGISTRY.get(name)(args)
    return dataset


def list_datasets(name=None):
    """
    List all available datasets
    Args:
        name: (TODO) list specific losses corresponds to a given name.
    """
    return list(DATASET_REGISTRY._obj_map.keys())