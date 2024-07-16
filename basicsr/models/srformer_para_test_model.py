from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch
import torchvision.utils as tvu

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import copy

import pyiqa


@MODEL_REGISTRY.register()
class SRFormerParaTestModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

         # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        # define metric functions 
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for name, opt in self.opt['val']['metrics'].items(): 
                mopt = opt.copy()
                metric_type = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(metric_type, device=self.device, **mopt)

        # load pre-trained HQ ckpt, frozen decoder and codebook 
        # self.LQ_stage = self.opt['network_g'].get('LQ_stage', False) 
        # if self.LQ_stage:

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])



    def feed_data(self, data):
        if not isinstance(data, dict):
            general_data, defocus_data = data

            # general data
            self.general_lq = general_data['lq'].to(self.device)
            if 'gt' in general_data:
                self.general_gt = general_data['gt'].to(self.device)
            if 'blur_mask' in general_data:
                self.general_blur_mask = general_data['blur_mask'].to(self.device)
            
            # defocus data
            self.defocus_lq = defocus_data['lq'].to(self.device)
            if 'gt' in defocus_data:
                self.defocus_gt = defocus_data['gt'].to(self.device)
            if 'blur_mask' in defocus_data:
                self.defocus_blur_mask = defocus_data['blur_mask'].to(self.device)
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
            if 'blur_mask' in data:
                self.blur_mask = data['blur_mask'].to(self.device)

        
    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000 # use smaller min_size with limited GPU memory
        lq_input = self.lq
        _, _, h, w = lq_input.shape
        if h*w < min_size:
            self.output = net_g.test(lq_input)
        else:
            self.output = net_g.test_tile(lq_input)
        self.net_g.train()
        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir=save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, save_as_dir=None, img_save_max_num=100):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        # if with_metrics:
        #     self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        
        self.net_g.module.combine_params()


        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
                self.metric_results_valid_count = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.metric_results_valid_count = {metric: 0 for metric in self.metric_results_valid_count}
            self.key_metric = self.opt['val'].get('key_metric') 

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            
            sr_img = tensor2img(self.output)
            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train'] and idx < img_save_max_num :
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}', 
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_result = self.metric_funcs[name](*metric_data)
                    if tmp_result > 0:
                        self.metric_results[name] += tmp_result.item()
                        self.metric_results_valid_count[name] += 1
        

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()
            
        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= self.metric_results_valid_count[metric] if self.metric_results_valid_count[metric] > 0 else 0
            
            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric, self.metric_results[self.key_metric], current_iter)
            
                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
            else:
                # update each metric separately 
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    updated.append(tmp_updated)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        
        torch.cuda.empty_cache()
    
    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
                return True
            else:
                return False
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
                return True
            else:
                return False
            
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    
    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx) 
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['general_lq'] = self.general_lq.detach().cpu()[:vis_samples]
        out_dict['general_result'] = self.general_output.detach().cpu()[:vis_samples]
        if hasattr(self, 'general_gt_rec'):
            out_dict['general_gt_rec'] = self.general_gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'general_gt'):
            out_dict['general_gt'] = self.general_gt.detach().cpu()[:vis_samples]
        
        out_dict['defocus_lq'] = self.defocus_lq.detach().cpu()[:vis_samples]
        out_dict['defocus_result'] = self.defocus_output.detach().cpu()[:vis_samples]
        if hasattr(self, 'defocus_gt_rec'):
            out_dict['defocus_gt_rec'] = self.defocus_gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'defocus_gt'):
            out_dict['defocus_gt'] = self.defocus_gt.detach().cpu()[:vis_samples]

        return out_dict
    

    def _update_metric_result(self, dataset_name, metric, val, current_iter):
        self.best_metric_results[dataset_name][metric]['val'] = val
        self.best_metric_results[dataset_name][metric]['iter'] = current_iter
