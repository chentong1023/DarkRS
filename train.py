import os
import time
from easydict import EasyDict as edict
from trainer import Trainer
from utils.config import *
# register all modules
import model_init

def main():
    args = parse_arguments()

    cfg = build_config(args.config_file)
    cfg = merge_args(cfg, args)
    cfg.args = edict(vars(args))
    cfg.experiment_time = time.strftime("%Y%m%d_%H%M%S")
    if args.local_rank == 0:
        save_path = os.path.join(cfg.work_dir, cfg.name, cfg.experiment_time)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_configs_to_yaml(cfg, os.path.join(save_path, cfg.name+".yaml"))

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()