import torch
import timm
import numpy as np
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import List, Tuple, Optional
from collections import OrderedDict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re
import os
from tqdm import tqdm 


def load_model(model_name: str, pretrained: bool = True) -> tuple[torch.nn.Module, callable | None]:
    '''
    Load a DNN model from the timm library and return model + preprocessing function
    Args:
        model_name (str): Name of the model to load
        pretrained (bool): Whether to load pretrained weights. Default is True.
    Returns:
        model (torch.nn.Module): The loaded DNN model.
        preprocess (callable | None): The preprocessing function, or None if not applicable.
    '''
    assert model_name in ['vgg16', 'dino', 'clip'], f'Model {model_name} not supported. Choose from "vgg16", "dino", or "clip".'
    preprocess= None

    if model_name == 'vgg16':
        model_str = 'vgg16_bn.tv_in1k'  
        
    elif model_name == 'dino':
        model_str = 'vit_base_patch16_dinov3.lvd1689m'
        
    elif model_name == 'clip':
        model_str = 'vit_base_patch16_clip_224.openai_ft_in1k'

    model = timm.create_model(model_str, pretrained=pretrained)
    model.eval()

    data_cfg = resolve_model_data_config(model)
    preprocess = create_transform(**data_cfg, is_training=False)
    print(f"Loaded model: {model_str} with pretrained={pretrained}. Found preprocessing function: {preprocess is not None}")

    return model, preprocess



def register_vgg16_hooks(model: torch.nn.Module, region: str, layer_percentage: float = 0.2):
    hooks = []
    activations = OrderedDict()


    valid_layers = [] # List of tuples (layer_name, layer_module) for valid layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            valid_layers.append((name, module))

    num_valid_layers = len(valid_layers)
    if num_valid_layers == 0:
        raise ValueError("No valid layers found in the model for hook registration.")
    

    num_layers_to_hook = max(1, int(num_valid_layers * layer_percentage))
    print(f"Total valid layers: {num_valid_layers}. Registering hooks on the top {num_layers_to_hook} {region} layers.")

    

    if region == 'early':
        target_layer_indices = range(0, num_layers_to_hook)
    elif region == 'mid':
        start_idx = num_valid_layers // 2 - num_layers_to_hook // 2
        target_layer_indices = range(start_idx, start_idx + num_layers_to_hook)
    elif region == 'late':
        target_layer_indices = range(num_valid_layers - num_layers_to_hook, num_valid_layers)


    def make_hook(layer_name):
        def hook_fn(module, inputs, output):
            # Conv: [B, C, H, W] -> [B, C]; Linear: [B, D] as-is
            if output.dim() == 4:
                x = output.mean(dim=(-2, -1))
            else:
                x = output
            if layer_name not in activations:
                activations[layer_name] = []
            activations[layer_name].append(x.detach().cpu())
        return hook_fn

    for idx in target_layer_indices:
        layer_name, layer_module = valid_layers[idx]
        h = layer_module.register_forward_hook(make_hook(layer_name))
        hooks.append(h)

    return hooks, activations

def register_vit_hooks(model: torch.nn.Module, region: str, layer_percentage: float = 0.2):
    hooks = []
    activations = OrderedDict()

    # timm ViTs: encoder blocks in model.blocks
    blocks = list(model.blocks)
    n_blocks = len(blocks)
    if n_blocks == 0:
        raise RuntimeError("No transformer blocks found in ViT model.")

    num_layers_to_hook = max(1, int(round(layer_percentage * n_blocks)))
    print(f"Total valid layer blocks: {n_blocks}. Registering hooks on the top {num_layers_to_hook} {region} blocks.")

    if region == 'early':
        block_indices = range(0, num_layers_to_hook)
    elif region == 'mid':
        start = max(0, (n_blocks - num_layers_to_hook) // 2)
        block_indices = range(start, start + num_layers_to_hook)
    elif region == 'late':
        block_indices = range(n_blocks - num_layers_to_hook, n_blocks)

    def make_hook(block_name):
        def hook_fn(module, inputs, output):
            # output: [B, tokens, C]; CLS token at index 0
            cls = output[:, 0, :]
            if block_name not in activations:
                activations[block_name] = []
            activations[block_name].append(cls.detach().cpu())
        return hook_fn

    for idx in block_indices:
        block = blocks[idx]
        name = f"blocks.{idx}"
        h = block.register_forward_hook(make_hook(name))
        hooks.append(h)
    return hooks, activations



def extract_model_activations(model: torch.nn.Module, model_name: str, region: str, dataloader: torch.utils.data.DataLoader, device: torch.device):
    
    assert region in ['early', 'mid', 'late'], f"Region must be one of 'early', 'mid', or 'late'. Got {region}."
    
    if 'vgg' in model_name.lower():
        hooks, activations = register_vgg16_hooks(model, region)
    else:
        hooks, activations = register_vit_hooks(model, region)
    print(f"Registered hooks for layers in the {region} region of the model.")

    model.to(device)
    model.eval()

    with torch.no_grad():
        for imgs in tqdm(dataloader, desc="Extracting features"):
            # imgs is already preprocessed [B, C, H, W]
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)
            imgs = imgs.to(device)

            _ = model(imgs)

    # remove hooks
    for h in hooks:
        h.remove()

    # concatenate across batches per layer/block
    feats_by_layer = {}
    for name, batches in activations.items():
        feats_by_layer[name] = torch.cat(batches, dim=0).numpy()

    return feats_by_layer



class ThingsImageDataset(Dataset):
    def __init__(self, root_dir, img_paths, transform=None, return_paths=False):
        """
        Args:
            root_dir: root directory for images (can be '' if img_paths are absolute)
            img_paths: list of relative or absolute image paths
            transform: torchvision-style transform (e.g. timm's preprocess)
            return_paths: if True, __getitem__ returns (image, path)
        """
        self.root_dir = Path(root_dir)
        self.img_paths = list(img_paths)
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        relpath = self.img_paths[idx]
        object_name = re.split(r'_\d+', relpath)[0]
        if object_name.endswith('_'):
            object_name = object_name[:-1]
        path = self.root_dir / object_name / relpath if not Path(relpath).is_absolute() else Path(relpath)
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            return img
        

def make_things_dataloader(
    root_dir: str,
    img_paths: list[str],
    transform,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    dataset = ThingsImageDataset(
        root_dir=root_dir,
        img_paths=img_paths,
        transform=transform,
       
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Created DataLoader with {len(dataset)} images, batch size {batch_size}, shuffle={shuffle}, num_workers={num_workers}.")
    return loader