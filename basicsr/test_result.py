import os.path
import sys
import torch
from utils import get_root_logger, imwrite, tensor2img, img2tensor
import pyiqa
from os import path as osp
import glob, cv2
import tqdm
from argparse import ArgumentParser 
import yaml
import numpy as np


root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
parser = ArgumentParser()
parser.add_argument('--gt_dir', '-g', type=str)
parser.add_argument('--lq_dir', '-l', type=str)
parser.add_argument('--mask_dir', '-m', type=str)
parser.add_argument('--option','-opt', type=str)

args = parser.parse_args()

hr_img_dir = args.gt_dir
sr_img_dir = args.lq_dir
mask_dir = args.mask_dir
with open(args.option, 'r', encoding='utf-8') as f:
    opt = yaml.load(f.read(), Loader=yaml.FullLoader)


sr_img_paths = sorted(glob.glob(os.path.join(sr_img_dir, "*"))) if os.path.isdir(sr_img_dir) else sorted(glob.glob(sr_img_dir))
hr_img_paths = sorted(glob.glob(os.path.join(hr_img_dir, "*"))) if os.path.isdir(hr_img_dir) else sorted(glob.glob(hr_img_dir))
if mask_dir is not None:
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*"))) if os.path.isdir(mask_dir) else sorted(glob.glob(mask_dir))
    assert len(sr_img_paths) == len(hr_img_paths) and len(sr_img_paths) == len(mask_paths), "The number of images in the input directories should be the same, but got sr: {}, hr: {}, mask: {}".format(len(sr_img_paths), len(hr_img_paths), len(mask_paths))
else:
    assert len(sr_img_paths) == len(hr_img_paths), "The number of images in the input directories should be the same, but got sr: {}, hr: {}".format(len(sr_img_paths), len(hr_img_paths))
    mask_paths = [None for _ in sr_img_paths]



with_metrics = opt['val'].get('metrics') is not None
device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
metric_results = {
                metric: []
                for metric in opt['val']['metrics'].keys()
            }
metric_results_valid_count = {metric: 0 for metric in opt['val']['metrics'].keys()}

if opt['val'].get('metrics') is not None:
    metric_funcs = {}
    for name, nopt in opt['val']['metrics'].items():
        mopt = nopt.copy()
        metric_type = mopt.pop('type', None)
        mopt.pop('better', None)
        metric_funcs[name] = pyiqa.create_metric(metric_type, device=device, **mopt)

print('sample num:', len(sr_img_paths))

for idx, (sr_path, mask_path, hr_path) in tqdm.tqdm(enumerate(zip(sr_img_paths, mask_paths, hr_img_paths)), total=len(sr_img_paths)):

    sr_img = cv2.imread(sr_path)
    gt = cv2.imread(hr_path)
    mask = img2tensor(cv2.imread(mask_path), bgr2rgb=False).unsqueeze(0).cuda()/255 if mask_path is not None else None
    
    metric_data = [img2tensor(sr_img).unsqueeze(0)/255, img2tensor(gt).unsqueeze(0)/255]
    if with_metrics:
        # calculate metrics
        with torch.no_grad():
            for name, opt_ in opt['val']['metrics'].items():
                if mask is not None:
                    tmp_result = metric_funcs[name](*metric_data,**{"mask":mask}).item()
                else:
                    tmp_result = metric_funcs[name](*metric_data).item()
                metric_results[name].append(tmp_result)
                if tmp_result > 0:
                    metric_results_valid_count[name] += 1
                torch.cuda.empty_cache()


if with_metrics:
    # calculate average metric
    for metric in metric_results.keys():
        metric_results[metric] = np.array(metric_results[metric])
        metric_results[metric] = np.sum(metric_results[metric][metric_results[metric]>0])/np.sum(metric_results_valid_count[metric]) if metric_results_valid_count[metric] > 0 else 0
    


metrics = ("".join([k+' | ' for k,v in metric_results.items()]))[:-1]+":"
values = ("".join(["{:.4f}".format(v) + ' | ' if v < 1 else "{:.2f}".format(v) +' | ' for k,v in metric_results.items()]))[:-2]
print(metrics, values)
