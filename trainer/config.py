"""
Copyright Zisheng Liang 2019 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
import argparse
from pathlib import Path

time_string = str(time.strftime("%Y%m%d-%H%M%S"))
model_name = 'srgan_deeplesion_16x'

def get_config():
    parser = argparse.ArgumentParser()
    
    # dataset Params
    parser.add_argument('--data_dir',  default=None, help='Directory to store downloaded and serialized data')
    parser.add_argument('--im_h', default=512, type=int, help='height of ground truth image')
    parser.add_argument('--im_w', default=512, type=int, help='width of ground truth image')

    # model Params
    parser.add_argument('--model_name', default=model_name)
    parser.add_argument('--upsampling_rate', default=16, type=int, help='unsampling rate of the model')
    
    
    # training Params
    parser.add_argument('--lr',       default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size',       default=1,    type=int, help='batch size')
    parser.add_argument('--num_epochs',   default=20,  type=int, help='number of epochs')

    
    # GAN Params
    parser.add_argument('--L1_LOSS_ALPHA', default=1000, type=float, help='importance of fine l1')
    parser.add_argument('--GAN_LOSS_ALPHA', default=0.001, type=float, help='importance of GAN loss')

  
    # log Params
    parser.add_argument('--job_dir',
                        default='../jobs/{}_{}'.format(model_name.upper(), time_string),
                        help='Job directory for tensorboard logging')
    parser.add_argument('--model_dir',
                        default='./trained_models/{}_{}'.format(model_name.upper(), time_string),
                        help='Directory for trained models')
    
    # pretrained weight
    parser.add_argument('--g_weight', default=None, help='pretrained generator weight path')
    parser.add_argument('--d_weight', default=None, help='pretrained discriminator weight path')

    
    parsed, unknown = parser.parse_known_args()
    

    return parsed

config = get_config()
