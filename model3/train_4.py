from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from datetime import datetime
import os

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import logging
import dataset
from model import *
from utils import *

def main(args):

	print(f"Using device: {args.device}")
	args.ckpt_dir = os.path.join(args.ckpt_dir, datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
	os.mkdir(args.ckpt_dir)
	logger = get_loggings(args.ckpt_dir)

	# dataset, split
	d = dataset.WindProtonDatasetToDST(args.data_path, args.start_year, args.end_year)
	logger.info(f"Dataset length: {len(d)}")
	data_num = len(d)
	train_ratio, valid_ratio, test_ratio = 0.7, 0.2, 0.1
	train_num, valid_num = int(data_num*train_ratio), int(data_num*valid_ratio)
	test_num = data_num - train_num - valid_num
	train_dataset, valid_dataset, test_dataset = random_split(d, [train_num, valid_num, test_num])
	# train_dataset, valid_dataset, test_dataset = d.split()
	logger.info(f"train, valid, test dataset len: {len(train_dataset)}, {len(valid_dataset)}, {len(test_dataset)}")

	# crecate DataLoader for train / dev datasets
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

	model_args = {"input_dim": 3, "hidden_dim": args.hidden_dim, "output_dim": 1, "num_layers": args.num_layers}
	model = Val2Val(**model_args).to(args.device)
	logger.info(model)

	# init optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=5e-5)
	# criterion = torch.nn.MSELoss()
	# criterion = torch.nn.BCELoss()
	best_eval_loss = np.inf

	for epoch in range(1, args.num_epoch+1):
		# Training loop - iterate over train dataloader and update model weights
		model.train()
		loss_train, acc_train, iter_train = 0, 0, 0
		for x_prime, dst in train_loader:
			x_prime, dst = x_prime.to(args.device), dst.to(args.device)
			outputs = model(x_prime)

			# calculate loss and update parameters
			# loss = criterion(outputs, dst)
			weight = torch.where(dst==1, 20., 1.)
			loss = F.binary_cross_entropy(outputs, dst, weight)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# accumulate loss, accuracy
			iter_train += 1
			loss_train += loss.item()
			acc_train += cls_accuracy(outputs, dst)
		
		loss_train /= iter_train
		acc_train /= iter_train

		# Evaluation loop - calculate accuracy and save model weights
		model.eval()
		with torch.no_grad():
			loss_eval, acc_eval, iter_eval = 0, 0, 0
			for x_prime, dst in valid_loader:
				x_prime, dst = x_prime.to(args.device), dst.to(args.device)
				outputs = model(x_prime)

				# calculate loss and update parameters
				# loss = criterion(outputs, dst)
				weight = torch.where(dst==1, 20., 1.)
				loss = F.binary_cross_entropy(outputs, dst, weight)

				# accumulate loss, accuracy
				iter_eval += 1
				loss_eval += loss.item()
				acc_eval += cls_accuracy(outputs, dst)
			
			loss_eval /= iter_eval
			acc_eval /= iter_eval

		logger.info(f"epoch: {epoch:4d}, train_acc: {acc_train:.4f}, eval_acc: {acc_eval:.4f}, train_loss: {loss_train:.4f}, eval_loss: {loss_eval:.4f}")
		scheduler.step(loss_eval)

		# save model
		if loss_eval < best_eval_loss:
			best_eval_loss = loss_eval
			logger.info(f"Trained model saved, eval loss: {best_eval_loss:.4f}")
			torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "model.pt"))
		

	# Inference on test set
	# first load-in best model
	model = Val2Val(**model_args).to(args.device)
	model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "model.pt")))
	model.eval()
	with torch.no_grad():
		loss_test, acc_test, prec_test, recall_test, iter_test = 0, 0, 0, 0, 0 
		for x_prime, dst in test_loader:
			x_prime, dst = x_prime.to(args.device), dst.to(args.device)
			outputs = model(x_prime)

			# calculate loss and update parameters
			# loss = criterion(outputs, dst)
			weight = torch.where(dst==1, 20., 1.)
			loss = F.binary_cross_entropy(outputs, dst, weight)

			# accumulate loss, accuracy
			iter_test += 1
			loss_test += loss.item()
			acc_test += cls_accuracy(outputs, dst)
		
			p, r =  cls_metrics(outputs, dst)
			prec_test += p
			recall_test += r

		loss_test /= iter_test
		acc_test /= iter_test
		prec_test /= iter_test
		recall_test /= iter_test

	logger.info("-----------------------------------------------")
	logger.info(f"Test, test_acc: {acc_test:.4f}, test_prec: {prec_test:.4f}, test_recall: {recall_test:.4f}, test_loss: {loss_test:.4f}")
		


def parse_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument(
		"--data_path",
		type=Path,
		help="Directory to the dataset.",
		default="./Data/processed-date/",
	)

	parser.add_argument(
		"--ckpt_dir",
		type=Path,
		help="Directory to save the model file.",
		default="./Results",
	)

	# year
	parser.add_argument("--start_year", type=int, default=2015)
	parser.add_argument("--end_year", type=int, default=2016)

	# model
	parser.add_argument("--hidden_dim", type=int, default=128)
	parser.add_argument("--num_layers", type=int, default=3)

	# optimizer
	parser.add_argument("--lr", type=float, default=1e-3)

	# data loader
	parser.add_argument("--batch_size", type=int, default=256)

	# training
	parser.add_argument(
		"--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
	)
	parser.add_argument("--num_epoch", type=int, default=1000)

	args = parser.parse_args()
	return args


def get_loggings(ckpt_dir):
	logger = logging.getLogger(name='TASK1-Val2Val')
	logger.setLevel(level=logging.INFO)
	# set formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# console handler
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	# file handler
	file_handler = logging.FileHandler(os.path.join(ckpt_dir, "record.log"))
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger


if __name__ == "__main__":
	args = parse_args()
	args.ckpt_dir.mkdir(parents=True, exist_ok=True)
	main(args)