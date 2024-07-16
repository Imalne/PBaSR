import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def extract_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    assert args.save_combinet_weight_path is not None, 'Please specify the path to save combinet weight in args, used --help to see the options'


    # create model
    model = build_model(opt)

    model.net_g.module.combine_params()

    weights = model.net_g.module.state_dict()
    combinet_weight = {key[9:]:value for key , value in weights.items() if 'combinet' in key}
    torch.save({'params': combinet_weight}, args.save_combinet_weight_path)



if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    extract_pipeline(root_path)
