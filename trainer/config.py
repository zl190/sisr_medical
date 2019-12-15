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

def get_config():
    parser = argparse.ArgumentParser()

    # Input parser
    parser.add_argument('--bs',       default=8,    type=int, help='batch size')
    parser.add_argument('--in_h',     default=224,  type=int, help='image input size height')
    parser.add_argument('--in_w',     default=224,  type=int, help='image input size width')
    parser.add_argument('--in_lh',     default=56,  type=int, help='low resolution image size height')
    parser.add_argument('--in_lw',     default=56,  type=int, help='low resolution image size width')
    parser.add_argument('--epochs',   default=100,  type=int, help='number of epochs')
    parser.add_argument('--lr',       default=1e-4, type=float, help='learning rate')
    parser.add_argument('--m',        default=True, type=bool, help='manual run or hp tuning')

    
    # dataset
    parser.add_argument('--dataset',  default='oxford_iiit_pet', help='dataset for training')

    
    # GAN Params
    parser.add_argument('--L1_LOSS_ALPHA', default=1000, type=float, help='importance of fine l1')
    parser.add_argument('--GAN_LOSS_ALPHA', default=0.001, type=float, help='importance of GAN loss')

  
    # Cloud ML Params
    parser.add_argument('--job-dir', default='gs://duke-MML/sisr/tmp/{}'.format(str(time.time())), help='Job directory for Google Cloud ML')
    parser.add_argument('--model_dir', default='./trained_models', help='Directory for trained models')
    parser.add_argument('--image_dir', default=None, help='Local image directory')
    
    # pretrained weight
    parser.add_argument('--g_weight', default=None, help='pretrained generator weight path')
    parser.add_argument('--d_weight', default=None, help='pretrained discriminator weight path')

    
    parsed, unknown = parser.parse_known_args()
    
    print('Unknown args:', unknown)
    print('Parsed args:', parsed.__dict__)
    
    return parsed

config = get_config()