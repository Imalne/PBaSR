import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math
from itertools import chain
from collections import OrderedDict
from basicsr.utils.registry import ARCH_REGISTRY

from .__init__ import build_network
import os

from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel

def load_network(net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        # if load_path.startswith('https://'):
        #     pretrain_model_dir = os.path.join('experiments/pretrained_models')
        #     load_path = load_file_from_url(load_path, model_dir=pretrain_model_dir)

        net = get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                print('Loading: params_ema does not exist, use params.')
            if param_key not in load_net and 'params_ema' in load_net:
                param_key = 'params_ema'
                print('Loading: params does not exist, use params_ema.')
            load_net = load_net[param_key]
        print(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        _print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)
    
def get_bare_model(net):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(net, (DataParallel, DistributedDataParallel)):
        net = net.module
    return net

def _print_different_keys_loading(crt_net, load_net, strict=True):
    """Print keys with different name or different size when loading models.

    1. Print keys with different names.
    2. If strict=False, print the same key but with different tensor size.
        It also ignore these keys with different sizes (not load).

    Args:
        crt_net (torch model): Current network.
        load_net (dict): Loaded network.
        strict (bool): Whether strictly loaded. Default: True.
    """
    crt_net = get_bare_model(crt_net)
    crt_net = crt_net.state_dict()
    crt_net_keys = set(crt_net.keys())
    load_net_keys = set(load_net.keys())

    if crt_net_keys != load_net_keys:
        print('Current net - loaded net:')
        for v in sorted(list(crt_net_keys - load_net_keys)):
            print(f'  {v}')
        print('Loaded net - current net:')
        for v in sorted(list(load_net_keys - crt_net_keys)):
            print(f'  {v}')

    # check the size for the same keys
    if not strict:
        common_keys = crt_net_keys & load_net_keys
        for k in common_keys:
            if crt_net[k].size() != load_net[k].size():
                print(f'Size different, ignore [{k}]: crt_net: '
                                f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                load_net[k + '.ignore'] = load_net.pop(k)



@ARCH_REGISTRY.register()
class ParallelNet(nn.Module):
    def __init__(self,
                 general_branch_parameter,
                 defocus_branch_parameter,
                 general_pretrained_weight_path=None,
                 defocus_pretrained_weight_path=None,
                 scale_factor=4,
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.general_branch = build_network(general_branch_parameter)
        if general_pretrained_weight_path is not None:
            load_network(self.general_branch, general_pretrained_weight_path, False)
            frozen_module_keywords = general_branch_parameter.get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.general_branch.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                                print(f"no optimization for : {name}")
                            break        
        
        
        self.defocus_branch = build_network(defocus_branch_parameter)
        if defocus_pretrained_weight_path is not None:
            load_network(self.defocus_branch, defocus_pretrained_weight_path, False)
            frozen_module_keywords = defocus_branch_parameter.get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.defocus_branch.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                                print(f"no optimization for : {name}")
                            break

        self.combinet = build_network(general_branch_parameter)
        # self.combine_params()
    
    @torch.no_grad()
    def combine_params(self):
        # pass
        combinet_params = dict(self.combinet.named_parameters())
        general_params = dict(self.general_branch.named_parameters())
        defocus_params = dict(self.defocus_branch.named_parameters())

        decay=0.5
        for name, param in self.combinet.named_parameters():
            combinet_params[name].data.copy_(general_params[name].data.detach().clone())
            combinet_params[name].data.mul_(decay).add_(defocus_params[name].data.detach().clone(), alpha=1 - decay)
        
        self.combinet.eval()


    @torch.no_grad()
    def branch_weight_cross(self, decay_0=0.995):
        general_params = dict(self.general_branch.named_parameters())
        defocus_params = dict(self.defocus_branch.named_parameters())

        cos_sims=[]
        for name in general_params.keys():
            cos_dis = F.cosine_similarity(general_params[name].data.view(-1), defocus_params[name].data.view(-1), dim=0)
            cos_sims.append(cos_dis_abs)
        
        cos_sim = torch.mean(torch.stack(cos_sims))
        decay = (1-decay_0) * cos_sim + decay_0

        for name in general_params.keys():
            tmp_weight = defocus_params[name].data.detach().clone()
            defocus_params[name].data.mul_(decay).add_(general_params[name].data.detach().clone(), alpha=1 - decay)
            general_params[name].data.mul_(decay).add_(tmp_weight, alpha=1 - decay)

            

    @torch.no_grad()
    def test(self, input, branch_type="combinet"):
        wsz = 8 // self.scale_factor * 8 
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        
        output, _, _, _ = self.forward(input, branch=branch_type)

        output = output
        output = output[..., :h_old * self.scale_factor, :w_old * self.scale_factor]
        return output 
    

    
    def encode_and_decode(self, branch_type="general", input=None, gt_indices=None):
        if branch_type == "general":
            return self.general_branch.encode_and_decode(input, gt_indices)
        elif branch_type == "defocus":
            return self.defocus_branch.encode_and_decode(input, gt_indices)
        elif branch_type == "combinet":
            
            return self.combinet.encode_and_decode(input, gt_indices)
        else:
            raise NotImplementedError(f"branch type {branch_type} is not implemented")

    def forward(self, input, gt_indices=None, branch='general'):
        
        return self.encode_and_decode(branch, input, gt_indices)


# para arch for no codebook models
@ARCH_REGISTRY.register()
class ParallelNet_ncb(nn.Module):
    def __init__(self,
                 general_branch_parameter,
                 defocus_branch_parameter,
                 general_pretrained_weight_path=None,
                 defocus_pretrained_weight_path=None,
                 scale_factor=4,
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.general_branch = build_network(general_branch_parameter)
        if general_pretrained_weight_path is not None:
            load_network(self.general_branch, general_pretrained_weight_path, False)
            frozen_module_keywords = general_branch_parameter.get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.general_branch.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                                print(f"no optimization for : {name}")
                            break        
        
        
        self.defocus_branch = build_network(defocus_branch_parameter)
        if defocus_pretrained_weight_path is not None:
            load_network(self.defocus_branch, defocus_pretrained_weight_path, False)
            frozen_module_keywords = defocus_branch_parameter.get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.defocus_branch.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                                print(f"no optimization for : {name}")
                            break

        self.combinet = build_network(general_branch_parameter)
        self.combine_params()
    
    @torch.no_grad()
    def combine_params(self):
        # pass
        combinet_params = dict(self.combinet.named_parameters())
        general_params = dict(self.general_branch.named_parameters())
        defocus_params = dict(self.defocus_branch.named_parameters())

        decay=0.5
        for name, param in self.combinet.named_parameters():
            combinet_params[name].data.copy_(general_params[name].data.detach().clone())
            combinet_params[name].data.mul_(decay).add_(defocus_params[name].data.detach().clone(), alpha=1 - decay)
        
        self.combinet.eval()
        # del combinet_params, general_params, defocus_params


    @torch.no_grad()
    def branch_weight_cross(self, decay=0.999):
        # pass
        general_params = dict(self.general_branch.named_parameters())
        defocus_params = dict(self.defocus_branch.named_parameters())
        for name in general_params.keys():
            tmp_weight = defocus_params[name].data.detach().clone()
            defocus_params[name].data.mul_(decay).add_(general_params[name].data.detach().clone(), alpha=1 - decay)
            general_params[name].data.mul_(decay).add_(tmp_weight, alpha=1 - decay)

        # del general_params, defocus_params

    @torch.no_grad()
    def test(self, input, branch_type="combinet"):
        wsz = 8 // self.scale_factor * 8 
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        
        output = self.forward(input, branch=branch_type)

        output = output
        output = output[..., :h_old * self.scale_factor, :w_old * self.scale_factor]
        return output 
    

    
    def encode_and_decode(self, branch_type="general", input=None):
        if branch_type == "general":
            return self.general_branch(input)
        elif branch_type == "defocus":
            return self.defocus_branch(input)
        elif branch_type == "combinet":
            return self.combinet(input)
        else:
            raise NotImplementedError(f"branch type {branch_type} is not implemented")

    def forward(self, input, branch='general'):
        
        return self.encode_and_decode(branch, input)