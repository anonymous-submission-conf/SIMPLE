import os
import time
import torch
import numpy as np
import torch.nn as nn
import random
import os
from tqdm import tqdm
import os
import time
import math
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def make_exp_dirs(exp_name):
    day_logs_root = 'generation_logs/' + time.strftime("%Y-%m%d", time.localtime())
    os.makedirs(day_logs_root, exist_ok=True)
    exp_log_path = os.path.join(day_logs_root, exp_name)
    
    os.makedirs(exp_log_path, exist_ok=True)  # log dir make

    return exp_log_path


def send_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def send_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


def writr_gt(test_dataloader, log_dir, tkr):
    gt_file_name_test = os.path.join(log_dir, ('reference_exp.txt'))
    gt_txt_test = open(gt_file_name_test, 'w', encoding='utf-8')

    gt_with_id_file_name_test = os.path.join(log_dir, ('hypothesis_exp.txt'))
    gt_with_id_txt_test = open(gt_with_id_file_name_test, 'w', encoding='utf-8')

    for idx, test_data in tqdm(enumerate(test_dataloader)):
        for i in range(len(test_data['input_ids'])):
            context = tkr.decode(test_data['input_ids'][i], skip_special_tokens=True)

            #Sarcasm Explanation
            label_pad = test_data['target_ids'][i].masked_fill(test_data['target_ids'][i] == -100, 0)
            label = tkr.decode(label_pad, skip_special_tokens=True)

            gt_with_id_txt_test.write(f"{context} \t\n")
            gt_txt_test.write(label + '\n')

    for txt in [gt_txt_test, gt_with_id_txt_test]:
        txt.flush()
        txt.close()