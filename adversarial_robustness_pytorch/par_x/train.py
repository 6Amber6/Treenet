from core import animal_classes, vehicle_classes, torch_isin
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks import create_attack
from core.metrics import accuracy, binary_accuracy, subclass_accuracy
from .model import create_model

from core.utils.mart import mart_loss, mart_tree_loss
from core.utils.rst import CosineLR
from core.utils.trades import trades_loss, trades_tree_loss

from core.models.treeresnet import lighttreeresnet

from core.utils.context import ctx_noparamgrad_and_eval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def focal_loss_with_pt(logits, targets, gamma=2.0, reduction='mean', weights = None):
    """
    Multi-class focal loss with pt recording.
    logits: [N, C]
    targets: [N]
    gamma: Focal loss gamma parameter
    reduction: 'mean', 'sum', or 'none'
    weights: Optional class weights for focal loss
    Returns:
        loss: Computed focal loss
        pt: Probability of the true class
    """
    log_probs = F.log_softmax(logits, dim=1)  # [N, C]
    probs = torch.exp(log_probs)              # [N, C]
    targets_one_hot = F.one_hot(targets, num_classes=logits.size(1))  # [N, C]

    pt = (probs * targets_one_hot).sum(dim=1)      # [N]
    log_pt = (log_probs * targets_one_hot).sum(dim=1)  # [N]

    focal_term = (1 - pt) ** gamma
    #loss = -focal_term * log_pt                    # [N]

    if weights is not None:
        class_weight = weights[targets]              # [N]
        loss = -class_weight * focal_term * log_pt  # [N]
    else:
        loss = -focal_term * log_pt                 # [N]

    if reduction == 'mean':
        return loss.mean(), pt
    elif reduction == 'sum':
        return loss.sum(), pt
    else:
        return loss, pt  # no reduction
    

class ParEnsemble(object):
    def __init__(self, 
        info, args,
        alpha1: float = 1,
        alpha2: float = 1,
        alpha3: float = 1,
        max_epochs: int = 100,  # Total number of training epochs
        gamma: float = 2.0,  # Focal loss gamma parameter
        loss_weights_animal = None,
        loss_weights_vehicle = None,

    ):
        # default using unknown_classes flags for create_model
        self.model = create_model(args.model, args.normalize, info, device, unknown_classes=args.unknown_classes)
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.gamma = gamma  # Focal loss gamma parameter

        self.max_epochs = args.num_adv_epochs

        self.params = args
        #self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.criterion = focal_loss_with_pt  # Use focal loss with pt recording
        # for kendall loss weight, set as trainable parameter 
        # self.s_r = nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        # self.s_a = nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        # self.s_v = nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)

        self.init_optimizer(self.params.num_adv_epochs)
        if self.params.pretrained_file is not None:
            self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'weights-best.pt'))

        self.attack, self.eval_attack = self.init_attack(self.model, self.wrap_loss_fn, self.params.attack, self.params.attack_eps, 
                                                         self.params.attack_iter, self.params.attack_step)
        
        self.loss_weights_animal = loss_weights_animal
        self.loss_weights_vehicle = loss_weights_vehicle

    @staticmethod
    def init_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step):
        """
        Initialize adversary.
        """
        attack = create_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform')
        if attack_type in ['linf-pgd', 'l2-pgd']:
            eval_attack = create_attack(model, criterion, attack_type, attack_eps, 2*attack_iter, attack_step)
        elif attack_type in ['fgsm', 'linf-df']:
            eval_attack = create_attack(model, criterion, 'linf-pgd', 8/255, 20, 2/255)
        elif attack_type in ['fgm', 'l2-df']:
            eval_attack = create_attack(model, criterion, 'l2-pgd', 128/255, 20, 15/255)
        return attack,  eval_attack
    
        
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and scheduler with different learning rates for root and subroot models.
        """
        self.optimizer = torch.optim.SGD([
            {"params": self.model.subroot_animal.parameters(), "lr": self.params.lr},    # subroot-animal: 正常
            {"params": self.model.subroot_vehicle.parameters(), "lr": self.params.lr },  # vehicle 学得慢一点，可以提速
            #{"params": [self.s_r, self.s_a, self.s_v], "lr": self.params.lr }  # kendall loss weight
        ],
            weight_decay=self.params.weight_decay, 
            momentum=0.9, 
            nesterov=self.params.nesterov
        )

        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
        
    def init_scheduler(self, num_epochs):
        """
        Initialize scheduler for different parameter groups.
        """
        if self.params.scheduler == 'cyclic':
            num_samples = 50000 if 'cifar10' in self.params.data else 73257
            num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
            update_steps = int(np.floor(num_samples / self.params.batch_size) + 1)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[self.params.lr , self.params.lr  ],
                pct_start=0.25,
                steps_per_epoch=update_steps,
                epochs=int(num_epochs),
            )
        elif self.params.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, gamma=0.1, milestones=[100, 105]
            )
        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[self.params.lr , self.params.lr],
                pct_start=0.025,
                total_steps=int(num_epochs),
            )
        else:
            self.scheduler = None

    def update_alphas(self, current_epoch: int, decay_factor: float=0.98, strategy: str = 'constant'):
        """
        Dynamically update alpha1, alpha2, and alpha3 based on the current epoch.
        """
        if strategy == 'constant':
            return self.alpha1, self.alpha2, self.alpha3

        elif strategy == 'linear':
            progress = current_epoch / self.max_epochs
            
            self.alpha2 = self.alpha2*(1-progress)
            self.alpha3 = self.alpha3*(1-progress)
            
        elif strategy == 'decay':
            self.alpha1 *= decay_factor

        return self.alpha1, self.alpha2, self.alpha3

    def forward(self, x, logits=False):
        if not logits:
            subroot_animal_logits, subroot_vehicle_logits = self.model(x)
        else:
            subroot_animal_logits, subroot_vehicle_logits = x

        # Ensure indices are on the same device as the logits
        device = subroot_animal_logits.device if logits else x.device
        animal_classes_index = torch.tensor(animal_classes, device=device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=device)
        
        conf_animal = 1 - F.softmax(subroot_animal_logits, dim=1)[:, -1]  # scalar confidence
        conf_vehicle = 1 - F.softmax(subroot_vehicle_logits, dim=1)[:, -1]

        animal_logits = torch.full((subroot_animal_logits.shape[0], 10), fill_value=-5.0, device=device)
        vehicle_logits = torch.full((subroot_vehicle_logits.shape[0], 10), fill_value=-5.0, device=device)

        animal_logits[:, animal_classes_index] = subroot_animal_logits[:, :-1]
        vehicle_logits[:, vehicle_classes_index] = subroot_vehicle_logits[:, :-1]

        logits_final = conf_animal.unsqueeze(1) * animal_logits + \
                    conf_vehicle.unsqueeze(1) * vehicle_logits
        return logits_final 

    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Train each trainer on a given (sub)set of data.
        """
        
        metrics = pd.DataFrame()  # Initialize metrics
        self.model.train()

        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            # each batch
            x, y = data
            x, y = x.to(device), y.to(device)
            #print(f"Training batch with shape {x.shape} and labels {y.shape}")
            if adversarial:
                if self.params.beta is not None and self.params.mart:
                    loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                elif self.params.beta is not None:
                    loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                else:
                    loss, batch_metrics = self.adversarial_loss(x, y)
            else:
                loss, batch_metrics = self.standard_loss(x, y) 
            loss.backward()

            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()
            
            metrics = pd.concat([metrics, pd.DataFrame(batch_metrics, index=[0])], ignore_index=True)
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        return dict(metrics.mean())
    
    
    def loss_fn(self, logits_set, y):
        """
        Loss function for the model with pt recording.
        """
        # Focal loss with pt recording
        final_logits = self.forward(logits_set, logits=True)
        loss_fuss, _ = self.criterion(final_logits, y) # nn.CrossEntropyLoss(reduction='mean')


        logits_animal, logits_vehicle = logits_set
        y_animal = y.clone()
        y_vehicle = y.clone()

        # Map `y` to subroot_animal and subroot_vehicle targets
        y_animal = map_labels_to_subroots(y_animal, animal_classes)
        y_vehicle = map_labels_to_subroots(y_vehicle, vehicle_classes)

        subroot_loss_animal, subroot_pt_animal = self.criterion(logits_animal, y_animal, gamma=self.gamma, weights=torch.tensor(self.loss_weights_animal).to(device))
        subroot_loss_vehicle, subroot_pt_vehicle = self.criterion(logits_vehicle, y_vehicle, gamma=self.gamma, weights=torch.tensor(self.loss_weights_vehicle).to(device))

        loss = self.alpha1 * loss_fuss + \
               self.alpha2 * subroot_loss_animal + \
               self.alpha3 * subroot_loss_vehicle

        return loss, subroot_loss_animal, subroot_loss_vehicle, subroot_pt_animal, subroot_pt_vehicle

    def wrap_loss_fn(self, logits_set, y):
        """
        Wrapper for loss function to return only the loss value.
        """
        loss, _, _, _, _ = self.loss_fn(logits_set, y)
        return loss
    
    def KL_loss_fn(self, logits_adv, logits_natural, y):
        """
        KL divergence loss function for TRADES.
        """
        # For tree models, we need to handle the logits properly
        if isinstance(logits_adv, tuple):
            # Tree model returns tuple of logits
            final_logits_adv = self.forward(logits_adv, logits=True)
            final_logits_natural = self.forward(logits_natural, logits=True)
        else:
            # Single model returns single logits
            final_logits_adv = logits_adv
            final_logits_natural = logits_natural
        
        # KL divergence between clean and adversarial logits
        kl_loss = F.kl_div(
            F.log_softmax(final_logits_adv, dim=1),
            F.softmax(final_logits_natural, dim=1),
            reduction='batchmean'
        )
        return kl_loss
    
    def mart_loss_fn(self, adv_logits_set, logits_set, y):
        """
        MART loss function.
        """
        logits_animal, logits_vehicle = logits_set
        adv_logits_animal, adv_logits_vehicle = adv_logits_set
        
        final_logits = self.forward(logits_set, logits=True)
        adv_final_logits = self.forward(adv_logits_set, logits=True)
        
        # MART loss components
        ce_loss = F.cross_entropy(adv_final_logits, y)
        kl_loss = F.kl_div(
            F.log_softmax(adv_final_logits, dim=1),
            F.softmax(final_logits, dim=1),
            reduction='batchmean'
        )
        
        return ce_loss + kl_loss


    def standard_loss(self, x, y):
        """
        Standard training with pt recording.
        """
        self.optimizer.zero_grad()
        out = self.forward(x)
        subroot_animal, subroot_vehicle = self.model(x)
        loss, subroot_loss_animal, subroot_loss_vehicle, subroot_pt_animal, subroot_pt_vehicle = self.loss_fn(
            [subroot_animal, subroot_vehicle], y
        )

        preds = out.detach()
        batch_metrics = {
            'loss': loss.item(),
            'clean_acc': accuracy(y, preds),
        }

        
        batch_metrics.update({
            'subroot_loss_animal': subroot_loss_animal.item() if subroot_loss_animal is not None else 0.0,
            'subroot_loss_vehicle': subroot_loss_vehicle.item() if subroot_loss_vehicle is not None else 0.0,
            'subroot_pt_animal': subroot_pt_animal.mean().item() if subroot_pt_animal is not None and subroot_pt_animal.numel() > 0 else 0.0,
            'subroot_pt_vehicle': subroot_pt_vehicle.mean().item() if subroot_pt_vehicle is not None and subroot_pt_vehicle.numel() > 0 else 0.0,
        })

        return loss, batch_metrics

    def adversarial_loss(self, x, y):
        """
        Adversarial training with pt recording.
        """
        with ctx_noparamgrad_and_eval(self.model):
            x_adv, _ = self.attack.perturb(x, y)

        self.optimizer.zero_grad()
        if self.params.keep_clean:
            x_adv = torch.cat((x, x_adv), dim=0)
            y_adv = torch.cat((y, y), dim=0)
        else:
            y_adv = y
        out = self.forward(x_adv)
        subroot_animal, subroot_vehicle = self.model(x_adv)
        loss, subroot_loss_animal, subroot_loss_vehicle, subroot_pt_animal, subroot_pt_vehicle = self.loss_fn(
            [subroot_animal, subroot_vehicle], y_adv
        )

        preds = out.detach()
        batch_metrics = {
            'loss': loss.item(),
            'subroot_loss_animal': subroot_loss_animal.item() if subroot_loss_animal is not None else 0.0,
            'subroot_loss_vehicle': subroot_loss_vehicle.item() if subroot_loss_vehicle is not None else 0.0,
            'subroot_pt_animal': subroot_pt_animal.mean().item() if subroot_pt_animal is not None and subroot_pt_animal.numel() > 0 else 0.0,
            'subroot_pt_vehicle': subroot_pt_vehicle.mean().item() if subroot_pt_vehicle is not None and subroot_pt_vehicle.numel() > 0 else 0.0,
        }
        if self.params.keep_clean:
            preds_clean, preds_adv = preds[:len(x)], preds[len(x):]
            batch_metrics.update({'clean_acc': accuracy(y, preds_clean), 'adversarial_acc': accuracy(y, preds_adv)})
        else:
            batch_metrics.update({'adversarial_acc': accuracy(y, preds)})
        return loss, batch_metrics
    
    def trades_loss(self, x, y, beta):
        """
        TRADES training. TO DO ... 
        """
        # loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
        #                                   epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
        #                                   beta=beta, attack=self.params.attack)
        loss, batch_metrics = trades_tree_loss(self.model, self.forward, self.KL_loss_fn, self.wrap_loss_fn, x, y, self.optimizer,
                                                  step_size=self.params.attack_step, 
                                                  epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                                  beta=beta, attack=self.params.attack)
        return loss, batch_metrics  

    
    def mart_loss(self, x, y, beta):
        """
        MART training. TO DO ...
        """
        loss, batch_metrics = mart_tree_loss(self.model, self.forward, self.mart_loss_fn, self.wrap_loss_fn, x, y, self.optimizer,
                                                    step_size=self.params.attack_step, 
                                                    epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                                    beta=beta, attack=self.params.attack)

        return loss, batch_metrics  

    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc, acc_animal, acc_vehicle = 0.0, 0.0, 0.0

        self.model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            #print(f"Evaluating batch with shape {x.shape} and labels {y.shape}")
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                out = self.forward(x_adv)
            else:
                out = self.forward(x)
                
            acc += accuracy(y, out)
            temp_acc_animal, temp_acc_vehicle = subclass_accuracy(y, out)
            acc_animal += temp_acc_animal
            acc_vehicle += temp_acc_vehicle

        acc /= len(dataloader)
        acc_animal /= len(dataloader)
        acc_vehicle /= len(dataloader)
        
        return dict(
            acc=acc,
            acc_animal=acc_animal,
            acc_vehicle=acc_vehicle,
        )
    
    def save_model(self, path):
        """
        Save model weights with error handling.
        """
        try:
            torch.save({'model_state_dict': self.model.state_dict()}, path)
        except Exception as e:
            print(f"Failed to save model at {path}: {e}")

    
    def load_model(self, path, load_opt=True):
        """
        Load model weights with error handling.
        """
        try:
            checkpoint = torch.load(path)
            if 'model_state_dict' not in checkpoint:
                raise RuntimeError(f"Model weights not found at {path}.")
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Failed to load model from {path}: {e}")

    def load_individual_models(self, animal_path=None, vehicle_path=None):
        """
        Load pre-trained weights for root_model, subroot_animal, and subroot_vehicle separately.
        """
        def load_model_weights(model, path):
            try:
                checkpoint = torch.load(path)
                if 'model_state_dict' not in checkpoint:
                    raise RuntimeError(f'Model weights not found at {path}.')
                state_dict = checkpoint['model_state_dict']
                model_state_dict = model.state_dict()

                # Filter out mismatched layers
                filtered_state_dict = {
                    k: v for k, v in state_dict.items()
                    if k in model_state_dict and model_state_dict[k].size() == v.size()
                }
                model_state_dict.update(filtered_state_dict)
                model.load_state_dict(model_state_dict)
            except Exception as e:
                print(f"Failed to load model from {path}: {e}")

        if animal_path:
            load_model_weights(self.model.subroot_animal, animal_path)
        if vehicle_path:
            load_model_weights(self.model.subroot_vehicle, vehicle_path)


def map_labels_to_subroots(y, pseudo_label_classes):
    """
    Map labels to subroot targets for soft routing.
    """
    # Validate pseudo_label_classes
    if not pseudo_label_classes or not isinstance(pseudo_label_classes, (list, torch.Tensor)):
        raise ValueError("Invalid pseudo_label_classes provided.")

    # Clone `y` to avoid in-place modifications
    y = y.clone()

    # Assign pseudo-label for unknown classes
    y[~torch_isin(y, torch.tensor(pseudo_label_classes, device=y.device))] = 10

    # Remap pseudo-label classes to a contiguous range starting from 0
    class_mapping = {old_label: new_label for new_label, old_label in enumerate(pseudo_label_classes)}
    class_mapping[10] = len(pseudo_label_classes)
    y = torch.tensor([class_mapping[int(label)] for label in y], device=y.device)  # Convert to int

    return y