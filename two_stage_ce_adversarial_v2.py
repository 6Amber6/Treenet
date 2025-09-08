"""
Two-Stage CE Optimization with Adversarial Training
Based on train-parx.py framework but with separate M1/M2 training and embedding fusion
"""

import json
import time
import argparse
import shutil
import os
import sys
import copy
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

import torchvision
import torchvision.transforms as T

# Ensure subpackage 'core' is importable when models/treeresnet references it
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARP_ROOT = os.path.join(PROJECT_ROOT, 'adversarial_robustness_pytorch')
if ARP_ROOT not in sys.path:
    sys.path.insert(0, ARP_ROOT)

from core.data import get_data_info, load_data
from core.models.resnet import LightResnet, BasicBlock
from core.attacks import create_attack
from core.utils.context import ctx_noparamgrad_and_eval
from core.metrics import accuracy, subclass_accuracy
from core.utils import format_time, Logger, parser_train, seed
from core.utils.mart import mart_loss
from core.utils.trades import trades_loss
from core import animal_classes, vehicle_classes

import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 class ids
ANIMAL_CLASSES = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
VEHICLE_CLASSES = [0, 1, 8, 9]       # airplane, automobile, ship, truck


def build_lightresnet20(num_classes: int) -> LightResnet:
    model = LightResnet(BasicBlock, [2, 2, 2], num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)


class HeadG(nn.Module):
    def __init__(self, in_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TwoStageTrainer:
    def __init__(self, info, args, beta=6.0):
        self.info = info
        self.args = args
        self.beta = beta
        self.device = DEVICE
        
        # Initialize models
        self.m1 = build_lightresnet20(num_classes=len(ANIMAL_CLASSES))  # 6-class animal
        self.m2 = build_lightresnet20(num_classes=len(VEHICLE_CLASSES))  # 4-class vehicle
        self.g = HeadG(in_dim=128, num_classes=10).to(DEVICE)  # 10-class fusion
        
        # Initialize optimizers and schedulers
        self.m1_optimizer = torch.optim.SGD(self.m1.parameters(), lr=args.lr, momentum=0.9, 
                                           weight_decay=args.weight_decay, nesterov=True)
        self.m2_optimizer = torch.optim.SGD(self.m2.parameters(), lr=args.lr, momentum=0.9, 
                                           weight_decay=args.weight_decay, nesterov=True)
        self.g_optimizer = torch.optim.Adam(self.g.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Initialize attacks
        self.m1_attack, self.m1_eval_attack = self.init_attack(self.m1, nn.CrossEntropyLoss(), 
                                                              args.attack, args.attack_eps, 
                                                              args.attack_iter, args.attack_step)
        self.m2_attack, self.m2_eval_attack = self.init_attack(self.m2, nn.CrossEntropyLoss(), 
                                                              args.attack, args.attack_eps, 
                                                              args.attack_iter, args.attack_step)
        self.g_attack, self.g_eval_attack = self.init_attack(self.g, nn.CrossEntropyLoss(), 
                                                            args.attack, args.attack_eps, 
                                                            args.attack_iter, args.attack_step)
        
        # Initialize schedulers
        self.m1_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.m1_optimizer, milestones=[35, 45], gamma=0.1)
        self.m2_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.m2_optimizer, milestones=[35, 45], gamma=0.1)
        self.g_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.g_optimizer, milestones=[35, 45], gamma=0.1)

    @staticmethod
    def init_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step):
        """Initialize adversary."""
        attack = create_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform')
        if attack_type in ['linf-pgd', 'l2-pgd']:
            eval_attack = create_attack(model, criterion, attack_type, attack_eps, 2*attack_iter, attack_step)
        elif attack_type in ['fgsm', 'linf-df']:
            eval_attack = create_attack(model, criterion, 'linf-pgd', 8/255, 20, 2/255)
        elif attack_type in ['fgm', 'l2-df']:
            eval_attack = create_attack(model, criterion, 'l2-pgd', 128/255, 20, 15/255)
        return attack, eval_attack

    def register_penultimate_hook(self, model: LightResnet):
        """Register a forward hook on fc to capture its input (penultimate features)."""
        feats = {}

        def hook_fn(module, input, output):
            feats["feat"] = input[0].detach()

        handle = model.fc.register_forward_hook(hook_fn)
        return feats, handle

    def extract_embeddings(self, m1_model, m2_model, dataloader):
        """Extract penultimate embeddings from M1 and M2 models."""
        m1_model.eval()
        m2_model.eval()

        feats_m1, h1 = self.register_penultimate_hook(m1_model)
        feats_m2, h2 = self.register_penultimate_hook(m2_model)

        all_feats, all_labels = [], []
        for x, y in dataloader:
            x = x.to(self.device)
            _ = m1_model(x)  # fills feats_m1["feat"]
            f1 = feats_m1["feat"].cpu()
            _ = m2_model(x)  # fills feats_m2["feat"]
            f2 = feats_m2["feat"].cpu()
            all_feats.append(torch.cat([f1, f2], dim=1))
            all_labels.append(y)

        h1.remove()
        h2.remove()
        return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

    def train_model_adversarial(self, model, optimizer, scheduler, attack, train_loader, test_loader, 
                               epochs, model_name, logger):
        """Train a single model with adversarial training."""
        model.train()
        
        for epoch in range(1, epochs + 1):
            total, correct, running_loss = 0, 0, 0.0
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # TRADES adversarial training
                loss = trades_loss(model, x, y, optimizer, beta=self.beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                with torch.no_grad():
                    logits = model(x)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            scheduler.step()
            
            if epoch % 5 == 0 or epoch == 1:
                clean_acc, adv_acc = self.eval_model_adversarial(model, test_loader, attack)
                logger.log(f'{model_name} Epoch {epoch:03d} | Train Loss {(running_loss/total):.4f} | Clean Acc {clean_acc:.4f} | Adv Acc {adv_acc:.4f}')

    def eval_model_adversarial(self, model, loader, attack):
        """Evaluate model with clean and adversarial accuracy."""
        model.eval()
        
        # Clean accuracy
        correct_clean, total = 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct_clean += (preds == y).sum().item()
                total += y.size(0)
        clean_acc = correct_clean / max(total, 1)
        
        # Adversarial accuracy
        correct_adv, total = 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with ctx_noparamgrad_and_eval(model):
                x_adv, _ = attack.perturb(x, y)
            with torch.no_grad():
                logits = model(x_adv)
                preds = logits.argmax(dim=1)
                correct_adv += (preds == y).sum().item()
                total += y.size(0)
        adv_acc = correct_adv / max(total, 1)
        
        return clean_acc, adv_acc

    def train_g_adversarial(self, X_train, y_train, X_test, y_test, epochs, logger):
        """Train model G with adversarial training on embeddings."""
        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.args.batch_size, shuffle=False)

        for epoch in range(1, epochs + 1):
            self.g.train()
            total, correct, run_loss = 0, 0, 0.0
            
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                # TRADES adversarial training for embeddings
                loss = trades_loss(self.g, xb, yb, self.g_optimizer, beta=self.beta)
                self.g_optimizer.zero_grad()
                loss.backward()
                self.g_optimizer.step()
                
                run_loss += loss.item() * xb.size(0)
                with torch.no_grad():
                    logits = self.g(xb)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)

            self.g_scheduler.step()
            
            if epoch % 5 == 0 or epoch == 1:
                clean_acc, adv_acc = self.eval_model_adversarial(self.g, test_loader, self.g_eval_attack)
                logger.log(f'G Epoch {epoch:03d} | Train Loss {(run_loss/total):.4f} | Clean Acc {clean_acc:.4f} | Adv Acc {adv_acc:.4f}')

    def save_models(self, save_dir):
        """Save all models."""
        torch.save({'model_state_dict': self.m1.state_dict()}, os.path.join(save_dir, 'M1_animal_6cls.pt'))
        torch.save({'model_state_dict': self.m2.state_dict()}, os.path.join(save_dir, 'M2_vehicle_4cls.pt'))
        torch.save({'model_state_dict': self.g.state_dict()}, os.path.join(save_dir, 'G_head_10cls.pt'))


def filter_and_remap_indices(dataset, keep_labels):
    """Filter dataset indices and create label remapping."""
    indices = [i for i, (_, y) in enumerate(dataset) if y in keep_labels]
    remap = {old: new for new, old in enumerate(keep_labels)}
    return indices, remap


class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices, remap):
        self.base = base
        self.indices = indices
        self.remap = remap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base[self.indices[idx]]
        y = self.remap[int(y)]
        return x, y


def build_filtered_loaders(data_dir, keep_labels, batch_size, train=True):
    """Build filtered dataloaders for specific classes."""
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]) if train else T.Compose([T.ToTensor()])

    ds = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    indices, remap = filter_and_remap_indices(ds, keep_labels)
    sub = RemappedSubset(ds, indices, remap)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
    return loader


def main():
    # Parse arguments using the same parser as train-parx.py
    parse = parser_train()
    parse.add_argument('--epochs-m', type=int, default=50, help='epochs for M1/M2 training')
    parse.add_argument('--epochs-g', type=int, default=50, help='epochs for G training')
    parse.add_argument('--beta', type=float, default=6.0, help='TRADES beta parameter')
    args = parse.parse_args()

    # Setup directories and logging
    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR = os.path.join(args.log_dir, args.desc)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

    with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Get data info
    info = get_data_info(DATA_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log('Using device: {}'.format(device))

    # Set random seed
    seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Load full dataset
    train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
        DATA_DIR, args.batch_size, args.batch_size_validation, use_augmentation=args.augment, 
        shuffle_train=True, aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction
    )

    # Build filtered loaders for M1 (6-class animal) and M2 (4-class vehicle)
    m1_train_loader = build_filtered_loaders(DATA_DIR, ANIMAL_CLASSES, args.batch_size, train=True)
    m1_test_loader = build_filtered_loaders(DATA_DIR, ANIMAL_CLASSES, args.batch_size, train=False)
    m2_train_loader = build_filtered_loaders(DATA_DIR, VEHICLE_CLASSES, args.batch_size, train=True)
    m2_test_loader = build_filtered_loaders(DATA_DIR, VEHICLE_CLASSES, args.batch_size, train=False)

    # Initialize trainer
    trainer = TwoStageTrainer(info, args, beta=args.beta)

    logger.log("Starting Two-Stage CE Optimization...")
    logger.log(f"Training M1 (6-class animal) for {args.epochs_m} epochs...")
    
    # Train M1 (6-class animal) with adversarial training
    trainer.train_model_adversarial(
        trainer.m1, trainer.m1_optimizer, trainer.m1_scheduler, trainer.m1_attack,
        m1_train_loader, m1_test_loader, args.epochs_m, "M1", logger
    )

    logger.log(f"Training M2 (4-class vehicle) for {args.epochs_m} epochs...")
    
    # Train M2 (4-class vehicle) with adversarial training
    trainer.train_model_adversarial(
        trainer.m2, trainer.m2_optimizer, trainer.m2_scheduler, trainer.m2_attack,
        m2_train_loader, m2_test_loader, args.epochs_m, "M2", logger
    )

    logger.log("Extracting embeddings for G...")
    
    # Extract embeddings for full train/test sets
    X_train, y_train = trainer.extract_embeddings(trainer.m1, trainer.m2, train_dataloader)
    X_test, y_test = trainer.extract_embeddings(trainer.m1, trainer.m2, test_dataloader)
    
    # Move to device for training head G
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    logger.log(f"Training G (10-class) for {args.epochs_g} epochs...")
    
    # Train head G with adversarial training
    trainer.train_g_adversarial(X_train, y_train, X_test, y_test, args.epochs_g, logger)

    # Final evaluation
    logger.log('\n=== Final Evaluation ===')
    
    logger.log('M1 (Animal 6-class):')
    clean_acc_m1, adv_acc_m1 = trainer.eval_model_adversarial(trainer.m1, m1_test_loader, trainer.m1_eval_attack)
    logger.log(f'  Clean Acc: {clean_acc_m1:.4f} | Adv Acc: {adv_acc_m1:.4f}')
    
    logger.log('M2 (Vehicle 4-class):')
    clean_acc_m2, adv_acc_m2 = trainer.eval_model_adversarial(trainer.m2, m2_test_loader, trainer.m2_eval_attack)
    logger.log(f'  Clean Acc: {clean_acc_m2:.4f} | Adv Acc: {adv_acc_m2:.4f}')
    
    logger.log('G (10-class ensemble):')
    test_ds = TensorDataset(X_test, y_test)
    test_loader_g = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    clean_acc_g, adv_acc_g = trainer.eval_model_adversarial(trainer.g, test_loader_g, trainer.g_eval_attack)
    logger.log(f'  Clean Acc: {clean_acc_g:.4f} | Adv Acc: {adv_acc_g:.4f}')

    # Save models
    trainer.save_models(LOG_DIR)
    logger.log(f'\nDone. Saved models to {LOG_DIR}')

    # Log final metrics to wandb if available
    try:
        wandb.init(project="two-stage-ce", name=args.desc, reinit=True)
        wandb.summary["final_m1_clean_acc"] = clean_acc_m1
        wandb.summary["final_m1_adv_acc"] = adv_acc_m1
        wandb.summary["final_m2_clean_acc"] = clean_acc_m2
        wandb.summary["final_m2_adv_acc"] = adv_acc_m2
        wandb.summary["final_g_clean_acc"] = clean_acc_g
        wandb.summary["final_g_adv_acc"] = adv_acc_g
        wandb.finish()
    except:
        pass


if __name__ == '__main__':
    main()
