import time

config = type('', (), {})()

config.bs = 4
config.in_h = 256
config.in_w = 256
config.in_lh = 128
config.in_lw = 128

config.WGAN_GP_LAMBDA = 10.0 # gradient penalty importance
config.COARSE_L1_ALPHA = 1.2 # importance of coarse l1
config.L1_LOSS_ALPHA = 1.2 # importance of fine l1
config.AE_LOSS_ALPHA = 1.2 # importance of full reconstruction
config.GAN_LOSS_ALPHA = 0.001 # importance of GAN loss
config.LOCAL = 1
config.NUM_ITER = 5

config.epochs = 100
config.m = True
config.lr = 1e-5

config.job_dir = 'gs://gradient-datascience/zisheng/debug/sisr/{}'.format(str(time.time()))
config.model_dir = './trained_models'

config.image_dir = None
