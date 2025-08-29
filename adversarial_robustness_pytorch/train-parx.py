"""
Adversarial Training. (par_x network)

must require unknown classes flag 
""" 

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core.data import get_data_info
from core.data import load_data

from core.utils import format_time
from core.utils import Logger
from core.utils import parser_train

from par_x.train import ParEnsemble
from par.individual_train import run_training, setup_data


from core.utils import seed
import wandb
import subprocess  # 用于调用评估脚本

from core import animal_classes, vehicle_classes

_WANDB_USERNAME = "yhe106-johns-hopkins-university"
_WANDB_PROJECT = "ablation_test"

# Setup

parse = parser_train()
# 添加参数
parse.add_argument('--decay_factor', type=float, default=0.98, help='Decay factor for alpha values.')
parse.add_argument('--strategy', type=str, default='constant', choices=['exponential', 'linear', 'constant'], help='Strategy for alpha decay.')
# alpha1 alpha2 alpha3
parse.add_argument('--alpha1', type=float, default=1.0, help='Initial value for alpha1.')
parse.add_argument('--alpha2', type=float, default=1.0, help='Initial value for alpha2.')
parse.add_argument('--alpha3', type=float, default=1.0, help='Initial value for alpha3.')

parse.add_argument('--train_submodels', type=bool, default=False, help='Train sub-models independently before ensemble training.')
parse.add_argument('--unknown_classes', type=bool, default=True, help='Use unknown classes for the tree ensemble.')
args = parse.parse_args()

DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)
logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(vars(args), f, indent=4)

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
NUM_ADV_EPOCHS = args.num_adv_epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))
if args.debug:
    NUM_ADV_EPOCHS = 1

torch.backends.cudnn.benchmark = True

# Load data
seed(args.seed)
train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
    DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, shuffle_train=True, 
    aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, 
)
del train_dataset, test_dataset

# Train sub-models if required
if args.train_submodels:
    from par.individual_train import run_training, setup_data
    logger.log("Training sub-models independently...")
    
    # Train subroot animal model
    pseudo_label_classes = animal_classes
    desc = "7-class-animal_parx"
    num_classes = 7
   

    animal_train_dataloader, animal_test_dataloader = setup_data(
        DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, args, filter_classes=None, pseudo_label_classes=pseudo_label_classes
    )
    wandb.init(
        project=_WANDB_PROJECT, entity=_WANDB_USERNAME,
        name=f"{args.desc}-{desc}",
        reinit=True,
    )
    run_training(
        info=info, temp_args=args, logger=logger, train_dataloader=animal_train_dataloader, test_dataloader=animal_test_dataloader,
        desc=desc, num_classes=num_classes, eval_metrics=["animal"], other_label=vehicle_classes , 
        weighted_loss=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2] 
    )
    # Clean up dataloaders
    del animal_train_dataloader, animal_test_dataloader
    wandb.finish()

    # Train subroot vehicle model
    pseudo_label_classes = vehicle_classes
    desc = "5-class-vehicle_parx"
    num_classes = 5

    vehicle_train_dataloader, vehicle_test_dataloader = setup_data(
        DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, args, filter_classes=None, pseudo_label_classes=pseudo_label_classes
    )
    wandb.init(
        project=_WANDB_PROJECT, entity=_WANDB_USERNAME,
        name=f"{args.desc}-{desc}",
        reinit=True,
    )
    run_training(
        info=info, temp_args=args, logger=logger, train_dataloader=vehicle_train_dataloader, test_dataloader=vehicle_test_dataloader,
        desc=desc, num_classes=num_classes, eval_metrics=["vehicle"], other_label=animal_classes , 
        weighted_loss=[1.0, 1.0, 1.0, 1.0, 0.2] 
    )
    # Clean up dataloaders
    del vehicle_train_dataloader, vehicle_test_dataloader
    wandb.finish()


# Initialize wandb at the end to log final metrics
wandb.init(
    project=_WANDB_PROJECT, entity=_WANDB_USERNAME,
    name=args.desc,
    reinit=True,
)

# Initialize ParEnsemble
trainer = ParEnsemble(info, args, loss_weights_animal=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2] ,
                      loss_weights_vehicle=[1.0, 1.0, 1.0, 1.0, 0.2] )

# Load pre-trained sub-models if specified
logger.log("Loading pre-trained sub-models...")

animal_model_path = os.path.join(args.log_dir, "7-class-animal_parx", "weights-best.pt")
vehicle_model_path = os.path.join(args.log_dir, "5-class-vehicle_parx", "weights-best.pt")
trainer.load_individual_models(
    animal_path=animal_model_path, vehicle_path=vehicle_model_path
)

logger.log("Model Summary:")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

logger.log(f"Total parameters: {count_parameters(trainer.model)}")
last_lr = args.lr

# save initial model weights
if not os.path.exists(WEIGHTS):
    logger.log('Saving initial model weights to {}'.format(WEIGHTS))
    trainer.save_model(WEIGHTS)

# breakpoint()

# Adversarial Training
if NUM_ADV_EPOCHS > 0:
    logger.log('\n\n')
    metrics = pd.DataFrame()
    logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader)['acc']*100))
    # 记录初始对抗准确率
    logger.log('Adversarial Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader, adversarial=True)['acc']*100))
    
    old_score = [0.0, 0.0]
    logger.log('Adversarial training for {} epochs'.format(NUM_ADV_EPOCHS))
    trainer.init_optimizer(args.num_adv_epochs)
    test_adv_acc = 0.0    

for epoch in range(1, NUM_ADV_EPOCHS+1):
    start = time.time()
    logger.log('======= Epoch {} ======='.format(epoch))
    
    # 提取学习率
    subroot_animal_lr = trainer.optimizer.param_groups[0]['lr']
    subroot_vehicle_lr = trainer.optimizer.param_groups[1]['lr']

    if args.scheduler:
        last_lr = trainer.scheduler.get_last_lr()[0]
    
    res = trainer.train(train_dataloader, epoch=epoch, adversarial=True)
    test_res = trainer.eval(test_dataloader)
    test_acc = test_res['acc']
    acc_animal = test_res['acc_animal']
    acc_vehicle = test_res['acc_vehicle']

    alpha1, alpha2, alpha3 = trainer.update_alphas(epoch, args.decay_factor)

    logger.log('Loss: {:.4f}.\tSubroot Animal LR: {:.6f}.\tSubroot Vehicle LR: {:.6f}'.format(
        res['loss'], subroot_animal_lr, subroot_vehicle_lr
    ))
    
    if 'clean_acc' in res:
        logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['clean_acc']*100, test_acc*100))
    else:
        logger.log('Standard Accuracy-\tTest: {:.2f}%.'.format(test_acc*100))
    
    epoch_metrics = {'train_'+k: v for k, v in res.items()}
    epoch_metrics.update({
        'epoch': epoch,
        'lr_subroot_animal': subroot_animal_lr,
        'lr_subroot_vehicle': subroot_vehicle_lr,
        'test_clean_acc': test_acc,
        'test_clean_acc_animal': acc_animal,
        'test_clean_acc_vehicle': acc_vehicle,
        'test_adversarial_acc': None,
        'test_adversarial_acc_animal': None,
        'test_adversarial_acc_vehicle': None,
    })
    
    if epoch % args.adv_eval_freq == 0 or epoch > (NUM_ADV_EPOCHS-5) or (epoch >= (NUM_ADV_EPOCHS-10) and NUM_ADV_EPOCHS > 90):
        test_adv_res = trainer.eval(test_dataloader, adversarial=True)
        test_adv_acc = test_adv_res['acc']
        test_adv_acc_animal = test_adv_res['acc_animal']
        test_adv_acc_vehicle = test_adv_res['acc_vehicle']

        logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100, 
                                                                                   test_adv_acc*100))
        epoch_metrics.update({'test_adversarial_acc': test_adv_acc})
        epoch_metrics.update({'test_adversarial_acc_animal': test_adv_acc_animal})
        epoch_metrics.update({'test_adversarial_acc_vehicle': test_adv_acc_vehicle})

    else:
        logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.'.format(res['adversarial_acc']*100))
    
    logger.log('Alpha1: {:.4f}.\tAlpha2: {:.4f}.\tAlpha3: {:.4f}'.format(alpha1, alpha2, alpha3))
    epoch_metrics.update({'alpha1': alpha1, 'alpha2': alpha2, 'alpha3': alpha3})

    if test_adv_acc >= old_score[1]:
        old_score[0], old_score[1] = test_acc, test_adv_acc
        trainer.save_model(WEIGHTS)
    trainer.save_model(os.path.join(LOG_DIR, 'weights-last.pt'))

    logger.log('Time taken: {}'.format(format_time(time.time()-start)))
    
    # 检查 epoch_metrics 是否为空或包含全为 NA 的列
    epoch_metrics_df = pd.DataFrame(epoch_metrics, index=[0])
    if not epoch_metrics_df.empty and not epoch_metrics_df.isna().all(axis=None):
        metrics = pd.concat([metrics, epoch_metrics_df], ignore_index=True)

    metrics.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)
    wandb.log(epoch_metrics)

# Record metrics
train_acc = res['clean_acc'] if 'clean_acc' in res else trainer.eval(train_dataloader)['acc']
logger.log('\nTraining completed.')
logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(train_acc*100, old_score[0]*100))
if NUM_ADV_EPOCHS > 0:
    logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100, old_score[1]*100)) 

logger.log('Script Completed.')


wandb.summary["final_train_acc"] = train_acc
wandb.summary["final_test_clean_acc"] = old_score[0]
wandb.summary["final_test_adv_acc"] = old_score[1]

# Ensure AutoAttack evaluation subprocess runs successfully
try:
    logger.log('Starting AutoAttack evaluation...')
    aa_result = subprocess.run(
        ['python', 'eval-aa.py', '--desc', args.desc, '--log-dir', args.log_dir, '--data-dir', args.data_dir, '--softroute', True, '--unknown_classes', True],
        capture_output=True, text=True
    )
    logger.log(aa_result.stdout)

    # Parse and log AutoAttack results to wandb
    for line in aa_result.stdout.splitlines():
        if "autoattack_clean_accuracy" in line:
            clean_acc = float(line.split(":")[1].strip().replace("%", "")) / 100
            wandb.summary["autoattack_clean_acc"] = clean_acc
        if "autoattack_robust_accuracy" in line:
            robust_acc = float(line.split(":")[1].strip().replace("%", "")) / 100
            wandb.summary["autoattack_robust_acc"] = robust_acc
except Exception as e:
    logger.log(f"AutoAttack evaluation failed: {e}")

wandb.finish()

