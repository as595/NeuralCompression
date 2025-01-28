import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.model_summary import ModelSummary

import numpy as np
import random
import csv
import os, sys
from PIL import Image
import psutil

from utils import parse_args, parse_config
from compressor import Compressor

from torchvision.datasets import MNIST, CIFAR10
from datasets import RGZ108k

import platform

if platform.system()=='Darwin':
	os.environ["GLOO_SOCKET_IFNAME"] = "en0"

quiet = False

torch.set_float32_matmul_precision('medium')


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if torch.cuda.is_available():
	device='cuda'
elif torch.backends.mps.is_built():
	device='mps'
else:
	device='cpu'

# -----------------------------------------------------------------------------

if __name__ == "__main__":

# -----------------------------------------------------------------------------
# extract config info & set the random seed

	args = parse_args()
	
	random_state = args['random_state']
	pl.seed_everything(random_state)

	config_dict, config = parse_config(args['config'])
	model_dir = config_dict['top level']['model_dir']

# -----------------------------------------------------------------------------
# lightning stuff

	batch_size = config_dict['training']['batch_size']
	learning_rate = config_dict['optimizer']['lr']
	beta = config_dict['model']['beta']

	config = {
			'beta': beta,
			'learning_rate': learning_rate,
			'batch_size': batch_size,
			'seed': random_state
			}

	# initialise the wandb logger
	wandb_logger = pl.loggers.WandbLogger(project='neural_compression', log_model=True, config=config)
	wandb_config = wandb.config

# -----------------------------------------------------------------------------

	os.makedirs(model_dir, exist_ok=True)
	num_cpus = psutil.cpu_count(logical=True)
	
# -----------------------------------------------------------------------------

	# data transforms
	totensor = transforms.ToTensor()
	normalise= transforms.Normalize(config_dict['data']['datamean'], config_dict['data']['datastd'])
	crop = transforms.CenterCrop(config_dict['data']['imagesize'])

	transform = transforms.Compose([
		totensor, 
		normalise,
		crop
		])

	print("Data: {}".format(config_dict['data']['dataset']))
	if config_dict['data']['dataset']=="MNIST":
		train_data = MNIST(root=config_dict['data']['datadir'], train=True, download=True, transform=transform)
		test_data = MNIST(root=config_dict['data']['datadir'], train=False, download=True, transform=transform)
	elif config_dict['data']['dataset']=="CIFAR":
		train_data = CIFAR10(root=config_dict['data']['datadir'], train=True, download=True, transform=transform)
		test_data = CIFAR10(root=config_dict['data']['datadir'], train=False, download=True, transform=transform)
	else:
		train_data = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], train=True, transform=transform)
		test_data = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], train=False, transform=transform)
	

	# split out a validation set:
	f_train = 1. - config_dict['data']['frac_valid']
	n_train = int(f_train*len(train_data))
	indices = list(range(len(train_data)))
	
	train_sampler = Subset(train_data, indices[:n_train]) 
	valid_sampler = Subset(train_data, indices[n_train:])   

	# specify data loaders for training and validation:
	train_loader = torch.utils.data.DataLoader(train_sampler, 
												batch_size=config_dict['training']['batch_size'], 
												shuffle=True, 
												num_workers=num_cpus-1,
												persistent_workers=True
												)

	val_loader = torch.utils.data.DataLoader(valid_sampler, 
												batch_size=15,
												shuffle=False, 
												num_workers=num_cpus-1,
												persistent_workers=True
												)

	test_loader = torch.utils.data.DataLoader(test_data, 
												batch_size=len(test_data), 
												shuffle=False, 
												num_workers=num_cpus-1,
												persistent_workers=True
												)

	
	
# -----------------------------------------------------------------------------

	print("Model: {} ({})".format(config_dict['model']['model_name'], device))
	model = Compressor(
						config_dict['model']['model_name'],
						config_dict['data']['nchan'],
						config_dict['data']['imagesize'],
						config_dict['model']['hidden'],
						config_dict['model']['latent_dim'],
						config_dict['optimizer']['lr'],
						).to(device)

	summary = ModelSummary(model, max_depth=-1)
	#print(summary)
	
# -----------------------------------------------------------------------------

	lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
	trainer = pl.Trainer(max_epochs=config_dict['training']['num_epochs'],
						 callbacks=[lr_monitor],
 						 num_sanity_val_steps=0, # 0 : turn off validation sanity check
						 accelerator=device, 
						 devices=1,
						 logger=wandb_logger) 

	# train the model
	trainer.fit(model, train_loader, val_dataloaders=val_loader)
	
# -----------------------------------------------------------------------------


	trainer.test(model, test_loader, ckpt_path=None) # test final epoch model

# -----------------------------------------------------------------------------

	wandb.finish()
