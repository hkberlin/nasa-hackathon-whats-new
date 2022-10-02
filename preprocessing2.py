import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from argparse import ArgumentParser, Namespace
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d

PYCDF_PATH = "/Applications/cdf/cdf38_1-dist"
os.environ["CDF_LIB"] = PYCDF_PATH
from spacepy import pycdf



# def load_data(year: int, folder: str):
# 	# x (mfi) [931251 * N_days, 3]
# 	x = np.load(os.path.join(folder, f"{year}-BGSE.npy"), allow_pickle=True)
# 	x_epoch = np.load(os.path.join(folder, f"{year}-Epoch.npy"), allow_pickle=True)
# 	x_epoch = [e[0] for e in x_epoch]

# 	# x_prime (mag) [86400 * N_days, 3]
# 	x_prime = np.load(os.path.join(folder, f"{year}-B1GSE.npy"), allow_pickle=True)

# 	# y (swe) [570, 3]
# 	n = np.expand_dims(np.load(os.path.join(folder, f"{year}-Proton_Np_moment.npy"), allow_pickle=True), axis=1)
# 	v = np.expand_dims(np.load(os.path.join(folder, f"{year}-Proton_V_moment.npy"), allow_pickle=True), axis=1)
# 	w = np.expand_dims(np.load(os.path.join(folder, f"{year}-Proton_W_moment.npy"), allow_pickle=True), axis=1)
# 	y = np.concatenate((n, v, w), axis=1)
# 	y_epoch = np.load(os.path.join(folder, f"{year}-swe-Epoch.npy"), allow_pickle=True)

# 	return x, x_epoch, x_prime, y, y_epoch


def load_data(year: int, folder: str):
	date = "20220101"
	mfi_path = f'Data/wind-mfi-mfi_h2/wi_h2_mfi_{date}_v04.cdf'
	dscovr_path = f'Data/dscovr-h0-mag/dscovr_h0_mag_{date}_v01.cdf'
	swe_path = f'Data/wind-swe-swe_h1/wi_h1_swe_{date}_v01.cdf'
	x = pycdf.CDF(mfi_path)["BGSE"][:]
	x_epoch = pycdf.CDF(mfi_path)["Epoch"][:]
	x_epoch = [e[0] for e in x_epoch]
	x_prime = pycdf.CDF(dscovr_path)["B1GSE"][:]
	n = np.expand_dims(pycdf.CDF(swe_path)["Proton_Np_moment"][:], axis=1)
	v = np.expand_dims(pycdf.CDF(swe_path)["Proton_V_moment"][:], axis=1)
	w = np.expand_dims(pycdf.CDF(swe_path)["Proton_W_moment"][:], axis=1)
	y = np.concatenate((n, v, w), axis=1)
	y_epoch = pycdf.CDF(swe_path)["Epoch"][:]

	x_prime[np.abs(x_prime) > 1e+10] = 0

	return x, x_epoch, x_prime, y, y_epoch


def interpolate(epoch, data, N):
	print("start interpolating:", data.shape)
	epoch = [e.timestamp()*1000 for e in tqdm(epoch)]
	interp_f = interp1d(epoch, data, axis=0)
	u_epoch = np.linspace(epoch[0], epoch[-1], N)
	return interp_f(u_epoch)


def smoothening(data:np.array, window_size:int, method:str='SMA', **kwargs) -> np.array:
	""" 
	data smoothening using `method` with `window_size`

	Args: 
		method: `SMA`, simple moving average, 
				`EMA`, exponent moving average
	"""
	print("smoothing data:", data.shape)
	smth_data = None

	if method=='SMA':
		idx = np.arange(0, len(data))
		start_idx = idx-window_size//2
		end_idx = start_idx+window_size

		mask_neg = start_idx<0
		shift = abs(start_idx[mask_neg])
		start_idx[mask_neg] += shift
		end_idx[mask_neg] += shift

		mask_over = end_idx>len(data)-1
		shift = end_idx[mask_over]-len(data)
		start_idx[mask_over] -= shift
		end_idx[mask_over] -= shift

		windows = ([list(range(s,e)) for s,e in tqdm(zip(start_idx, end_idx))])
		smth_data = data[windows].mean(axis=1)
	
	if method=='EMA':
		alpha = kwargs.get('alpha', 2/(window_size+1))
		EMA = data[0]
		for i in tqdm(range(len(data))):            
			EMA = alpha*data[i] + (1-alpha)*EMA
			data[i] = EMA
		smth_data = data

	return smth_data



def downsample(data:np.array, num_of_day: int=365, frequency: int=24) -> np.array:
	print("start downsampling:", data.shape)
	N = num_of_day*frequency
	epoch = np.arange(0, len(data))
	interp_f = interp1d(epoch, data, axis=0)
	u_epoch = np.linspace(epoch[0], epoch[-1], N)   				# uniform epoch in milliseconds
	return interp_f(u_epoch)


# calculate R2 function
def calculate_R2_score(arr1: np.array, arr2: np.array) -> float: # [0, 1]
	y_bar = np.sum(np.mean(arr2), axis=0)
	SS_tot = np.sum((arr2 - y_bar)**2, axis=0)
	SS_res = np.sum((arr2 - arr1)**2, axis=0)
	R2 = 1 - SS_res / SS_tot
	R2[R2 < 0] = 0
	return R2

# calculate correlation function
def calculate_correlation(arr1: np.array, arr2: np.array) -> float: # [0, 1]
	correlation_score = np.corrcoef(arr1, arr2)[0, 1] # This is a two-by-two matrix --> float
	if correlation_score < 0:
		correlation_score = 0
	return correlation_score


# Main_Function
def similarity(x: np.array, x_prime: np.array ,algorithm='R2'):
	if algorithm == 'R2':
		R2_score = calculate_R2_score(x_prime, x)
		# print('The R2 score is: ', R2_score)
		return R2_score

	elif algorithm == 'correlation':
		correlation_score = calculate_correlation(x_prime, x)
		# print('The correlation score is: ', correlation_score)
		return correlation_score



def cut(x: np.array, x_prime: np.array, y: np.array,
		seq_len: int,
		threshold: float,
		similarity_metric: str) -> List[Tuple[np.array, np.array, np.array]]:
	# slide the window through the sequence
	total_length = x.shape[0]
	dataset = []
	for i in range(0, total_length - seq_len):
		x_sequence = x[i:i+seq_len, :]
		x_prime_sequence = x_prime[i:i+seq_len, :]
		scores = similarity(x_sequence, x_prime_sequence, similarity_metric)	# [3]
		# if similarity high enough, add to data list
		if np.all(scores >= threshold):
			y_sequence = y[i:i+seq_len]
			dataset.append((x_sequence, x_prime_sequence, y_sequence))

	return dataset


def save(inputData: List[Tuple[np.array, np.array, np.array]], save_path: str, year: int) :
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	save_data_path = os.path.join(save_path, f"data_{year}.pt")
	# concat
	if len(inputData) == 1:
		outputDataxx = inputData[0][0]
		outputDataX = inputData[0][1]
		outputDataY = inputData[0][2][-1]
	else:
		outputDataxx = np.stack( (inputData[0][0],inputData[1][0]), axis = 0)
		outputDataX =  np.stack( (inputData[0][1],inputData[1][1]), axis = 0)
		outputDataY =  np.stack( (inputData[0][2][-1],inputData[1][2][-1]), axis = 0)
		n_seq = inputData[0][0].shape[0]
		n_features = inputData[0][0].shape[1]
		for i in range(len(inputData)-2):
			outputDataxx = np.concatenate((outputDataxx,inputData[i+2][0].reshape(1,n_seq,n_features) ), axis = 0)
			outputDataX =  np.concatenate( (outputDataX,inputData[i+2][1].reshape(1,n_seq,n_features) ), axis = 0)
			outputDataY =  np.concatenate( (outputDataY,inputData[i+2][2][-1].reshape(1,n_features) ), axis = 0)

	# save to torch.tensor
	outputDataxx = torch.tensor(outputDataxx)
	outputDataX =  torch.tensor(outputDataX)
	outputDataY =  torch.tensor(outputDataY)
	dic = dict({
		'xx':outputDataxx,
		'X': outputDataX,
		'Y': outputDataY
	})
	with open(save_data_path, "wb") as file:
		torch.save(dic, file)
		print("save file to:", save_data_path)


def plot(dataset: List[Tuple[np.array, np.array, np.array]], plot_path: str, year:int, similarity_metric: str) -> None:
	if not os.path.exists(plot_path):
		os.mkdir(plot_path)
	for i, (x, x_prime, _) in enumerate(dataset):
		save_plot_path = os.path.join(plot_path, f"{year}_sequence_{i}.png")
		scores = similarity(x, x_prime, similarity_metric)
		fig, axs = plt.subplots(3, 1, figsize=(12, 7))
		# Bx
		axs[0].plot(x[:, 0], linewidth=1, color='k', label='Wind')
		axs[0].plot(x_prime[:, 0], linewidth=1, color='red', label='DSCOVR')
		axs[0].grid()
		axs[0].legend()
		axs[0].set_title(f"{similarity_metric} = {scores[0]:.4f}")
		axs[0].set_ylabel("Bx")

		# By
		axs[1].plot(x[:, 1], linewidth=1, color='k', label='Wind')
		axs[1].plot(x_prime[:, 1], linewidth=1, color='red', label='DSCOVR')
		axs[1].grid()
		axs[1].legend()
		axs[1].set_title(f"{similarity_metric} = {scores[1]:.4f}")
		axs[1].set_ylabel("By")

		# Bz
		axs[2].plot(x[:, 2], linewidth=1, color='k', label='Wind')
		axs[2].plot(x_prime[:, 2], linewidth=1, color='red', label='DSCOVR')
		axs[2].grid()
		axs[2].legend()
		axs[2].set_title(f"{similarity_metric} = {scores[2]:.4f}")
		axs[2].set_ylabel("Bz")

		fig.supxlabel('Timestep')
		plt.savefig(save_plot_path)
		plt.close()



def parse_args() -> Namespace:
	parser = ArgumentParser()

	# preprocessing
	# parser.add_argument("--num_of_day", type=int, default=365, help="day")

	# load_data
	parser.add_argument("--data_path", type=str, default="Data/npyFiles")

	# smoothing
	parser.add_argument("--window_size", type=int, default=5)
	parser.add_argument("--smoothing_method", type=str, default="EMA")

	# downsample
	parser.add_argument("--frequency", type=int, default=24, help="timestep per day")

	# similarity
	parser.add_argument("--similarity_metric", type=str, default='R2')

	# cut
	parser.add_argument("--seq_len", type=int, default=8)
	parser.add_argument("--similarity_threshold", type=float, default=0.5)

	# paths
	parser.add_argument("--save_path", type=str, default="Data/processed")
	parser.add_argument("--plot_path", type=str, default="Data/plot")

	args = parser.parse_args()
	return args



def main(args):
	for year in range(2017, 2017+1):

		x, x_epoch, x_prime, y, y_epoch = load_data(year, args.data_path)

		# get how many days in the period
		num_of_day = int(x_prime.shape[0] / 86400)
		print("num of day:", num_of_day)

		# interpolate to make uniform
		x = interpolate(x_epoch, x, x.shape[0])
		y = interpolate(y_epoch, y, y.shape[0])

		# smooth
		x = smoothening(x, args.window_size, args.smoothing_method)
		x_prime = smoothening(x_prime, args.window_size, args.smoothing_method)

		# downsample
		x = downsample(x, num_of_day, args.frequency)
		x_prime = downsample(x_prime, num_of_day, args.frequency)
		y = downsample(y, num_of_day, args.frequency)
		print("after downsample, x:", x.shape)
		print("after downsample, x_prime:", x_prime.shape)
		print("after downsample, y:", y.shape)
		
		# cut
		dataset = cut(x, x_prime, y, args.seq_len, args.similarity_threshold, args.similarity_metric)
		print("dataset len:", len(dataset))

		# save
		save(dataset, args.save_path, year)

		# plot
		plot(dataset, args.plot_path, year, args.similarity_metric)





if __name__ == "__main__":
	args = parse_args()
	main(args)




	