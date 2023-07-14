import os
import sys
import copy
import logging
import time

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
from datetime import datetime
from utils.metrics import *
from utils import dist_utils
from utils import hooks
from utils.logger import *
from utils.build import *

class Trainer:
    def __init__(self, cfg):
        """
        Args
            cfg(edict): the whole config file
            cfg.model: the model configurations
            cfg.dataset: the dataset configurations
            cfg.loss: the loss configuration
            cfg.schedule: the training configuration
            cfg.args: the arguments from the command line
        """
        self.cfg = copy.deepcopy(cfg)
        # initialize the distributed training
        if cfg.args.gpus > 1:
            dist_utils.init_distributed(cfg)
        # create the working dirs
        self.proj_dir = os.path.join(cfg.work_dir, cfg.name)
        self.experiment_name = f"{len(os.listdir(self.proj_dir)) + 1:03d}"
        self.experiment_time = cfg.experiment_time
        self.current_work_dir = os.path.join(self.proj_dir, self.experiment_time)
        if not os.path.exists(self.current_work_dir):
            os.makedirs(self.current_work_dir, exist_ok=True)
        init_logger(log_file_path=self.current_work_dir)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        # construct the training architecture
        self.arch = build_meta_arch(self.cfg)

        # construct training datasets
        self.train_dataloader, self.train_sampler = self.build_dataloder(
            cfg, mode="train")
        self.val_datalocaer, _ = self.build_dataloder(cfg, mode="val")

        # build the optimizer and lr_scheduler
        if hasattr(self.arch, "build_scheduler"):
            # arch-specific optimizer and lr_scheduler building
            self.optimizer, self.lr_scheduler = self.arch.build_scheduler()
        else:
            # default general optimizer and lr_scheduler building
            self.optimizer = self.build_optimizer(cfg, self.arch.model)
            self.lr_scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        # trainer hooks
        self._hooks = self.build_hooks()

        # some induces when training
        self.epochs = 0
        self.iters = 0
        self.batch_idx = 0

        self.start_epoch = 0
        self.start_iter = 0
        self.total_train_epochs = self.cfg.schedule.epochs
        self.total_train_iters = self.total_train_epochs * len(self.train_dataloader)

        # resume or load the ckpt as init-weights
        if self.cfg.resume_from != "None":
            self.resume_or_load_ckpt(ckpt_path=self.cfg.resume_from)

        # log bufffer(dict to save)
        self.log_buffer = LogBuffer()

    def preprocess(self, batch_data):
        """
        prepare for model input
        """
        return self.arch.preprocess(batch_data)

    def postprocess(self):
        """
        post process for model outputs
        """
        self.outputs["results"] = self.arch.postprocess(self.outputs["results"])

        # When the outputs is a video tensor
        if isinstance(self.outputs, torch.Tensor) and self.outputs.dim() == 5:
            self.outputs = self.outputs.flatten(0, 1)

    def update_params(self, batch_data):
        """
        update params
        pipline: zero_grad, backward and update grad
        """
        # arch-specific parameters updation
        self.outputs = self.arch.update_params(batch_data, self.optimizer)

    def train(self):
        self.arch.model.train()
        self.before_train()
        logger = logging.getLogger("simdeblur")
        logger.info("Starting training...")
        for self.epochs in range(self.start_epoch, self.cfg.schedule.epochs):
            # shuffle the dataloader when dist training: dist_data_loader.set_epoch(epoch)
            self.before_epoch()
            for self.batch_idx, self.batch_data in enumerate(self.train_dataloader):
                self.before_iter()

                self.outputs = self.arch.update_params(self.batch_data, self.optimizer)

                self.postprocess()

                self.iters += 1
                self.after_iter()

            if (self.epochs + 1) % self.cfg.schedule.val_epochs == 0 or self.epochs == 0 \
                    or (self.epochs + 1) == self.cfg.schedule.epochs:
                self.val()

            self.after_epoch()

    def before_train(self):
        for h in self._hooks:
            h.before_train(self)

    def after_train(self):
        for h in self._hooks:
            h.after_train(self)

    def before_epoch(self):
        for h in self._hooks:
            h.before_epoch(self)
        # shuffle the data when dist training ...
        if self.train_sampler:
            self.train_sampler.set_epoch(self.epochs)

    def after_epoch(self):
        for h in self._hooks:
            h.after_epoch(self)

        self.arch.model.train()

    def before_iter(self):
        for h in self._hooks:
            h.before_iter(self)

    def after_iter(self):
        for h in self._hooks:
            h.after_iter(self)

    def run_iter(self, batch_data):
        pass

    @torch.no_grad()
    def val(self):
        self.arch.model.eval()
        for self.batch_data in tqdm(self.val_datalocaer,
                                    ncols=80,
                                    desc=f"validation on gpu{self.cfg.args.local_rank}:"):
            self.before_iter()

            if hasattr(self.arch, "inference"):
                self.outputs = {"results": self.arch.inference(self.preprocess(self.batch_data))}
            else:
                self.outputs = {"results": self.arch.model(self.preprocess(self.batch_data))}

            self.postprocess()

            self.after_iter()

    def build_hooks(self):
        ret = [
            hooks.LRScheduler(self.lr_scheduler),
            hooks.CKPTSaver(**self.cfg.ckpt),
            # logging on the main process
            hooks.PeriodicWriter([
                SimpleMetricPrinter(self.current_work_dir),
                TensorboardWriter(self.current_work_dir),
            ],
            **self.cfg.logging),
        ]

        return ret

    def resume_or_load_ckpt(self, ckpt=None, ckpt_path=None):
        logger = logging.getLogger("simdeblur")
        try:
            kwargs = {'map_location': lambda storage,
                      loc: storage.cuda(self.cfg.args.local_rank)}
            ckpt = torch.load(ckpt_path, **kwargs)

            # initial mode: load the ckpt as the initialized weights
            logger.info("Inittial mode: %s, checkpoint loaded from %s." % (
                self.cfg.get("init_mode"), self.cfg.resume_from))
            if not self.cfg.get("init_mode"):
                # load the ckpt into arch.model
                self.arch.load_ckpt(ckpt, strict=True)

                # load optimizer and lr_scheduler
                if isinstance(self.optimizer, dict):
                    for name in self.optimizer.keys():
                        self.optimizer[name].load_state_dict(ckpt["optimizer"][name])
                else:
                    self.optimizer.load_state_dict(ckpt["optimizer"])

                if isinstance(self.optimizer, dict):
                    for name in self.optimizer.keys():
                        self.lr_scheduler[name].load_state_dict(ckpt["lr_scheduler"][name])
                else:
                    self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
                # generate the idx
                meta_info = ckpt["mata"]
                self.start_epoch = self.epochs = meta_info["epochs"]
                self.start_iter = self.iters = self.start_epoch * \
                    len(self.train_dataloader)
            else:
                # load the ckpt into arch.model with false strict
                self.arch.load_ckpt(ckpt, strict=True)

        except Exception as e:
            logger.warning(e)
            logger.warning("Checkpoint loaded failed, cannot find ckpt file from %s." % (
                self.cfg.resume_from))

    def save_ckpt(self, out_dir=None, ckpt_name="epoch_{}.pth", dence_saving=False):
        meta_info = {
            "epochs": self.epochs + 1,
            "iters": self.iters + 1
        }

        ckpt_name = ckpt_name.format(self.epochs + 1) if dence_saving else "latest.pth"
        if out_dir is None:
            out_dir = self.current_work_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(out_dir, ckpt_name)

        # construct checkpoint
        ckpt = {
            "meta": meta_info,
            "optimizer":
                {k: v.state_dict() for k, v, in self.optimizer.items()} if isinstance(
                    self.optimizer, dict) else self.optimizer.state_dict(),
            "lr_scheduler":
                {k: v.state_dict() for k, v in self.lr_scheduler.items()} if isinstance(
                    self.lr_scheduler, dict) else self.lr_scheduler.state_dict()
        }
        ckpt.update(self.arch.generate_ckpt())

        with open(ckpt_path, "wb") as f:
            torch.save(ckpt, ckpt_path)
            f.flush()

    def get_current_lr(self):
        if isinstance(self.optimizer, dict):
            return {k: optim.param_groups[0]["lr"] for k, optim in self.optimizer.items()}
        else:
            assert self.lr_scheduler.get_last_lr(
            )[0] == self.optimizer.param_groups[0]["lr"]
            return self.optimizer.param_groups[0]["lr"]

    @classmethod
    def build_model(cls, cfg):
        """
        build a model
        """
        # TODO change the build backbone to build model
        model = build_backbone(cfg.model)
        if cfg.args.gpus > 1:
            rank = cfg.args.local_rank
            model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[rank], output_device=rank)
        if cfg.args.local_rank == 0:
            logger = logging.getLogger("simdeblur")
            logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_losses(cls, cfg):
        """
        build all losses
        """
        criterions = {}
        if isinstance(cfg, list):
            for loss_cfg in cfg:
                criterions[loss_cfg.name] = build_loss(loss_cfg)
        else:
            criterions[cfg.name] = build_loss(cfg)

        return criterions

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        """
        if not cfg.schedule.get("optimizer"):
            return None
        return build_optimizer(cfg.schedule.optimizer, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        """
        if optimizer is None or (not cfg.schedule.get("lr_scheduler")):
            return None
        return build_lr_scheduler(cfg.schedule.lr_scheduler, optimizer)

    @classmethod
    def build_dataloder(cls, cfg, mode="train"):
        if mode == "train":
            dataset_cfg = cfg.dataset.train
        elif mode == "val":
            dataset_cfg = cfg.dataset.val
        elif mode == "test":
            dataset_cfg = cfg.dataset.test
        else:
            raise NotImplementedError
        dataset = build_dataset(dataset_cfg)
        if cfg.args.gpus > 1:
            # TODO reimplement the dist dataloader partition without distsampler,
            # that is because we must shuffe the dataloader by ourself before each epoch
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=cfg.args.gpus, rank=cfg.args.local_rank, shuffle=True)
            dataloder = torch.utils.data.DataLoader(
                dataset, **dataset_cfg.loader, sampler=sampler)
            return dataloder, sampler

        else:
            dataloder = torch.utils.data.DataLoader(
                dataset, **dataset_cfg.loader)
            return dataloder, None

    @classmethod
    def test(cls, cfg):
        """
        Only single GPU testing is surppored at now.
        TODO: Separate the testing process.
        Args:
            cfg(edict): the config file for testing, which contains "model" and "test dataloader" configs etc.
        """
        experiment_time = time.strftime("%Y%m%d_%H%M%S")
        current_work_dir = os.path.join(cfg.work_dir, cfg.name, "tested", experiment_time)
        if not os.path.exists(current_work_dir):
            os.makedirs(current_work_dir, exist_ok=True)
        init_logger(log_file_path=current_work_dir)
        logger = logging.getLogger("simdeblur")

        if cfg.args.gpus > 1:
            dist_utils.init_distributed(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        arch = build_meta_arch(cfg)
        test_dataloader, _ = Trainer.build_dataloder(cfg, "test")

        # load the trained checkpoint
        try:
            kwargs = {'map_location': lambda storage,
                      loc: storage.cuda(cfg.args.local_rank)}
            ckpt = torch.load(os.path.abspath(cfg.args.ckpt_file), **kwargs)

            arch.load_ckpt(ckpt, strict=True)

            logger.info("Using checkpoint loaded from %s for testing." %
                        (cfg.args.ckpt_file))
        except Exception as e:
            logger.warning(e)
            logger.warning("Checkpoint loaded failed, cannot find ckpt file from %s." % (
                cfg.args.ckpt_file))

        arch.model.eval()
        psnr_dict = {}
        ssim_dict = {}
        total_time = 0.
        with torch.no_grad():
            for batch_data in tqdm(test_dataloader,
                                   ncols=80,
                                   desc=f"validation on gpu{cfg.args.local_rank}:"):
                input_frames = arch.preprocess(batch_data)
                gt_frames = batch_data["gt_frames"].to(device)

                # record the testing time.
                torch.cuda.synchronize()
                time_start = time.time()
                if hasattr(arch, "inference"):
                    outputs = arch.postprocess(arch.inference(input_frames))
                else:
                    outputs = arch.postprocess(arch.model(input_frames))
                torch.cuda.synchronize()
                total_time += time.time() - time_start

                # print("video name: ", batch_data["video_name"])
                # print("frame name: ", batch_data["gt_names"])
                # calculate metrics
                b, n, c, h, w = gt_frames.shape
                outputs =  outputs.view(b, n, c, h, w)
                # single image output
                if outputs.dim() == 4:
                    outputs = outputs.detach().unsqueeze(1)  # (b, 1, c, h, w)
                for b_idx in range(b):
                    for n_idx in range(n):
                        frame_name = "{}_{}".format(
                            batch_data["video_name"][b_idx], batch_data["gt_names"][n_idx][b_idx])
                        psnr_dict[frame_name] = calculate_psnr(
                            gt_frames[b_idx, n_idx:n_idx+1], outputs[b_idx, n_idx:n_idx+1]).item()
                        ssim_dict[frame_name] = calculate_ssim(
                            gt_frames[b_idx, n_idx:n_idx+1], outputs[b_idx, n_idx:n_idx+1]).item()

                        # save the output images
                        save_path_base = os.path.abspath(
                            os.path.join(current_work_dir, batch_data["video_name"][b_idx]))
                        if not os.path.exists(save_path_base):
                            os.makedirs(save_path_base, exist_ok=True)
                        save_path = os.path.join(
                            save_path_base, batch_data["gt_names"][n_idx][b_idx])
                        save_image(outputs[b_idx, n_idx:n_idx+1], save_path)
                        # save testing logs
                        with open(os.path.abspath(os.path.join(current_work_dir, "test_log.txt")), "a") as f:
                            f.write("{}, {}, {}, {} \n".format(
                                batch_data["video_name"][b_idx],
                                batch_data["gt_names"][n_idx][b_idx],
                                psnr_dict[frame_name],
                                ssim_dict[frame_name]))
        mean_psnr = sum(psnr_dict.values()) / len(psnr_dict)
        mean_ssim = sum(ssim_dict.values()) / len(ssim_dict)
        with open(os.path.abspath(os.path.join(current_work_dir, "test_log.txt")), "a") as f:
            f.write("mean_psnr: {}  mean_ssim: {}".format(mean_psnr, mean_ssim))

        print("Memory: ", torch.cuda.memory_allocated())
        print("mean PSNR: {:.2f}  mean SSIM: {:.4f}  total time: {:.2f}s  average time: {:.4f}s FPS: {:.2f}".format(
            mean_psnr,
            mean_ssim,
            total_time,
            total_time / len(test_dataloader),
            len(test_dataloader.dataset) / total_time))
