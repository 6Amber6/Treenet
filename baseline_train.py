"""
Baseline Training Script (fixed & runnable).
Runs training for three configurations:
1. 10-class (full CIFAR-10)
2. animal subset
3. vehicle subset
"""

import json, time, argparse, shutil, os
import numpy as np, pandas as pd
import torch

from core.data import get_data_info, load_data
from core.utils import format_time, Logger, parser_train, Trainer, seed
from core import animal_classes, vehicle_classes

import wandb

# ---------------- utils ----------------
class DotDict(dict):
    """dict <-> attribute 互通：既支持 info['key'] 也支持 info.key"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def _wandb_safe_init(name, cfg):
    try:
        return wandb.init(
            project=os.getenv("WANDB_PROJECT", "treenet-baseline"),
            entity=os.getenv("WANDB_ENTITY", None),
            name=name,
            mode=os.getenv("WANDB_MODE", "disabled"),  # 默认禁用，手动 export 改为 online 才启用
            config=cfg,
        )
    except Exception:
        os.environ["WANDB_MODE"] = "disabled"
        return None

def _wandb_log(data: dict):
    if getattr(wandb, "run", None) is not None:
        wandb.log(data)

def _wandb_summary(k, v):
    if getattr(wandb, "run", None) is not None:
        wandb.summary[k] = v

def _wandb_finish():
    if getattr(wandb, "run", None) is not None:
        wandb.finish()

# ---------------- training ----------------
def run_training(desc, num_classes=None, filter_classes=None, eval_metrics=None):
    parse = parser_train()
    args = parse.parse_args()
    args.desc = desc

    # 设置类别数
    if filter_classes is not None:
        args.num_classes = len(filter_classes)
    else:
        args.num_classes = int(num_classes) if num_classes is not None else 10

    # wandb 初始化（安全）
    _wandb_safe_init(
        name=args.desc,
        cfg={
            "data": args.data,
            "batch_size": args.batch_size,
            "num_adv_epochs": args.num_adv_epochs,
            "beta": args.beta,
            "model": args.model,
            "subset": "animal" if filter_classes is animal_classes else ("vehicle" if filter_classes is vehicle_classes else "all"),
            "num_classes": args.num_classes,
        },
    )

    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR = os.path.join(args.log_dir, args.desc)
    WEIGHTS_BEST = os.path.join(LOG_DIR, 'weights-best.pt')
    WEIGHTS_LAST = os.path.join(LOG_DIR, 'weights-last.pt')

    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))
    with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # ---------- dataset info ----------
    info_raw = get_data_info(DATA_DIR)
    if isinstance(info_raw, dict):
        info = DotDict(info_raw)
    else:
        info = DotDict(getattr(info_raw, "__dict__", {}))

    info['num_classes'] = args.num_classes
    if 'data' not in info or info['data'] is None:
        info['data'] = args.data

    BATCH_SIZE = args.batch_size
    BATCH_SIZE_VALIDATION = args.batch_size_validation
    NUM_ADV_EPOCHS = args.num_adv_epochs
    if args.debug:
        NUM_ADV_EPOCHS = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f'Using device: {device}')
    torch.backends.cudnn.benchmark = True

    # ---------- dataloaders ----------
    seed(args.seed)
    train_dataset, test_dataset, train_loader, test_loader = load_data(
        DATA_DIR,
        BATCH_SIZE,
        BATCH_SIZE_VALIDATION,
        use_augmentation=args.augment,
        shuffle_train=True,
        aux_data_filename=args.aux_data_filename,
        unsup_fraction=args.unsup_fraction,
        filter_classes=filter_classes
    )
    del train_dataset, test_dataset

    # ---------- trainer ----------
    seed(args.seed)
    trainer = Trainer(info, args)
    logger.log("Model Summary:")
    try:
        from torchsummary import summary
        summary(trainer.model, input_size=(3, 32, 32), device=str(device))
    except Exception:
        logger.log("torchsummary unavailable, skipping summary.")

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    logger.log(f"Total Trainable Parameters: {count_parameters(trainer.model)}")

    last_lr = args.lr
    metrics_df = pd.DataFrame()
    best_clean, best_adv = 0.0, 0.0

    # ---------- training loop ----------
    for epoch in range(1, NUM_ADV_EPOCHS + 1):
        t0 = time.time()
        logger.log(f'======= Epoch {epoch} =======')

        if args.scheduler:
            last_lr = trainer.scheduler.get_last_lr()[0]

        train_res = trainer.train(train_loader, epoch=epoch, adversarial=True)
        test_res = trainer.eval(test_loader)

        row = {'epoch': epoch, 'lr': last_lr}
        row.update({f"train_{k}": v for k, v in train_res.items()})

        if eval_metrics is None:
            eval_metrics = []
        for metric in eval_metrics:
            row[f'test_clean_{metric}'] = test_res.get(f'acc_{metric}', None)
            row[f'test_adversarial_{metric}'] = None

        test_adv_res = {}
        if (epoch % args.adv_eval_freq == 0) or (epoch > NUM_ADV_EPOCHS - 5):
            test_adv_res = trainer.eval(test_loader, adversarial=True)
            for metric in eval_metrics:
                row[f'test_adversarial_{metric}'] = test_adv_res.get(f'acc_{metric}', None)

        current_clean = float(test_res.get('acc', 0.0))
        current_adv   = float(test_adv_res.get('acc', best_adv)) if test_adv_res else best_adv
        if current_clean >= best_clean:
            best_clean, best_adv = current_clean, current_adv
            trainer.save_model(WEIGHTS_BEST)
        trainer.save_model(WEIGHTS_LAST)

        logger.log(f'Time taken: {format_time(time.time() - t0)}')
        metrics_df = pd.concat([metrics_df, pd.DataFrame(row, index=[0])], ignore_index=True)
        metrics_df.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)
        _wandb_log(row)

    _wandb_summary("final_train_acc", train_res.get('clean_acc', train_res.get('acc', None)))
    _wandb_summary("final_test_clean_acc", best_clean)
    _wandb_summary("final_test_adv_acc", best_adv)
    _wandb_finish()

# ---------------- run all ----------------
if __name__ == "__main__":
    run_training(desc="400_10-class",        num_classes=10, filter_classes=None,           eval_metrics=["animal", "vehicle", "bi"])
    run_training(desc="400_6-class-animal",  filter_classes=animal_classes,                 eval_metrics=["animal"])
    run_training(desc="400_4-class-vehicle", filter_classes=vehicle_classes,                eval_metrics=["vehicle"])
