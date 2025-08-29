import torch

import torchvision
import torchvision.transforms as transforms


DATA_DESC = {
    'data': 'cifar10',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
    'mean': [0.4914, 0.4822, 0.4465], 
    'std': [0.2023, 0.1994, 0.2010],
}


def load_cifar10(data_dir, use_augmentation=False, filter_classes=None, binary_classes=None, pseudo_label_classes=None):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
        filter_classes (list): List of class indices to keep in the dataset.
        binary_classes (list): List of class indices to map to label 1 for binary classification.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)    

    # Apply class filtering if specified
    if filter_classes is not None:
        train_targets_tensor = torch.tensor(train_dataset.targets)
        test_targets_tensor = torch.tensor(test_dataset.targets)
        filter_classes_tensor = torch.tensor(filter_classes)
        
        train_mask = torch.zeros_like(train_targets_tensor, dtype=torch.bool)
        test_mask = torch.zeros_like(test_targets_tensor, dtype=torch.bool)
        
        for cls in filter_classes_tensor:
            train_mask |= (train_targets_tensor == cls)
            test_mask |= (test_targets_tensor == cls)
            
        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = train_targets_tensor[train_mask].tolist()
        test_dataset.data = test_dataset.data[test_mask]
        test_dataset.targets = test_targets_tensor[test_mask].tolist()

        # Remap labels to a contiguous range starting from 0
        class_mapping = {old_label: new_label for new_label, old_label in enumerate(filter_classes)}
        train_dataset.targets = [class_mapping[label] for label in train_dataset.targets]
        test_dataset.targets = [class_mapping[label] for label in test_dataset.targets]

    # # Apply binary relabeling if specified
    # if binary_classes is not None:
    #     positive_classes = set(binary_classes)
    #     train_dataset.targets = [1 if label in positive_classes else 0 for label in train_dataset.targets]
    #     test_dataset.targets = [1 if label in positive_classes else 0 for label in test_dataset.targets]

    # pseodo-label, all other classes are set to other 
    if pseudo_label_classes is not None:
        train_dataset.targets = torch.tensor(train_dataset.targets)
        test_dataset.targets = torch.tensor(test_dataset.targets)
        pseudo_label_classes_tensor = torch.tensor(pseudo_label_classes)
        
        # Create masks for pseudo-label classes
        train_mask = torch.zeros_like(train_dataset.targets, dtype=torch.bool)
        test_mask = torch.zeros_like(test_dataset.targets, dtype=torch.bool)
        
        for cls in pseudo_label_classes_tensor:
            train_mask |= (train_dataset.targets == cls)
            test_mask |= (test_dataset.targets == cls)
        
        # Set non-pseudo-label classes to 10
        train_dataset.targets[~train_mask] = 10
        test_dataset.targets[~test_mask] = 10
        
        # Remap pseudo-label classes to a contiguous range starting from 0
        class_mapping = {old_label: new_label for new_label, old_label in enumerate(pseudo_label_classes)}
        class_mapping[10] = len(pseudo_label_classes)
        #print(class_mapping, train_dataset.targets.unique())
        train_dataset.targets = [class_mapping[int(label)] for label in train_dataset.targets]  # Convert to int
        test_dataset.targets = [class_mapping[int(label)] for label in test_dataset.targets]    # Convert to int
    
    return train_dataset, test_dataset