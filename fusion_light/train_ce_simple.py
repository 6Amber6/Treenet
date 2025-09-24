#!/usr/bin/env python3
"""
Simplified CE training script for the tree model
"""

import os
import json
import time
import argparse
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
sys.path.append('./adversarial_robustness_pytorch')

from core.data import get_data_info, load_data
from core.utils import format_time, Logger, seed
from par_x.train import ParEnsemble
from core import animal_classes, vehicle_classes

def main():
    parser = argparse.ArgumentParser(description='Simplified CE Training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--log-dir', type=str, default='./log_ce_optimization', help='Log directory')
    parser.add_argument('--desc', type=str, default='ce_simple', help='Description')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.4, help='Learning rate')
    parser.add_argument('--alpha1', type=float, default=0.4, help='Alpha1 weight')
    parser.add_argument('--alpha2', type=float, default=0.3, help='Alpha2 weight')
    parser.add_argument('--alpha3', type=float, default=0.3, help='Alpha3 weight')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    DATA_DIR = os.path.join(args.data_dir, 'cifar10')
    LOG_DIR = os.path.join(args.log_dir, args.desc)
    WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
    
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))
    
    with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    info = get_data_info(DATA_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log('Using device: {}'.format(device))
    
    # Set random seed
    seed(args.seed)
    
    # Load data
    train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
        DATA_DIR, args.batch_size, args.batch_size, use_augmentation=True, shuffle_train=True
    )
    
    # Create a simple args object for ParEnsemble
    class SimpleArgs:
        def __init__(self):
            self.model = 'lighttreeresnet20'
            self.normalize = True
            self.num_adv_epochs = args.num_epochs
            self.lr = args.lr
            self.weight_decay = 0.0005
            self.scheduler = 'cosinew'
            self.nesterov = True
            self.attack = 'linf-pgd'
            self.attack_eps = 0.03137254901960784
            self.attack_step = 0.00784313725490196
            self.attack_iter = 10
            self.beta = 6.0
            self.unknown_classes = True
            self.pretrained_file = None
            self.log_dir = LOG_DIR
            self.clip_grad = None
            self.keep_clean = False
            self.mart = False
    
    simple_args = SimpleArgs()
    
    # Initialize trainer
    trainer = ParEnsemble(
        info, simple_args,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        alpha3=args.alpha3,
        loss_weights_animal=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2],
        loss_weights_vehicle=[1.0, 1.0, 1.0, 1.0, 0.2]
    )
    
    logger.log("Model Summary:")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.log(f"Total parameters: {count_parameters(trainer.model)}")
    
    # Training loop
    logger.log('Starting training for {} epochs'.format(args.num_epochs))
    metrics = pd.DataFrame()
    
    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        logger.log('======= Epoch {} ======='.format(epoch))
        
        # Train
        res = trainer.train(train_dataloader, epoch=epoch, adversarial=True)
        
        # Evaluate
        test_res = trainer.eval(test_dataloader)
        test_adv_res = trainer.eval(test_dataloader, adversarial=True)
        
        # Log results
        logger.log('Loss: {:.4f}'.format(res['loss']))
        logger.log('Clean Accuracy - Train: {:.2f}%, Test: {:.2f}%'.format(
            res.get('clean_acc', 0)*100, test_res['acc']*100))
        logger.log('Adversarial Accuracy - Train: {:.2f}%, Test: {:.2f}%'.format(
            res.get('adversarial_acc', 0)*100, test_adv_res['acc']*100))
        
        # Save metrics
        epoch_metrics = {
            'epoch': epoch,
            'loss': res['loss'],
            'train_clean_acc': res.get('clean_acc', 0),
            'train_adv_acc': res.get('adversarial_acc', 0),
            'test_clean_acc': test_res['acc'],
            'test_adv_acc': test_adv_res['acc'],
            'alpha1': trainer.alpha1,
            'alpha2': trainer.alpha2,
            'alpha3': trainer.alpha3,
        }
        
        metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
        metrics.to_csv(os.path.join(LOG_DIR, 'stats.csv'), index=False)
        
        # Save best model
        if test_adv_res['acc'] > 0:  # Simple condition for now
            trainer.save_model(WEIGHTS)
        
        logger.log('Time taken: {}'.format(format_time(time.time()-start)))
    
    logger.log('Training completed!')
    logger.log('Final Test Clean Accuracy: {:.2f}%'.format(test_res['acc']*100))
    logger.log('Final Test Adversarial Accuracy: {:.2f}%'.format(test_adv_res['acc']*100))

if __name__ == '__main__':
    main()
