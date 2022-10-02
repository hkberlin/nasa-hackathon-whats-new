from typing import Dict, List, Tuple
import os
import torch
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

"""
DST format: 

RECORD FORMAT (LENGTH: 120 BYTE FIXED)
COLUMN	FORMAT	SHORT DESCRIPTION

4-5     The last two digits of the year
6-7     Month
9-10	Date
15-16	Top two digits of the year (19 or space for 19XX, 20 from 2000)
21-116	24 hourly values, 4 digit number, unit 1 nT, value 9999 for the missing data.
		First data is for the first hour of the day, and Last data is for the last hour of the day.

"""

# DST data
def get_dst(dst_path):
	print('-'*20, f'{"get dst data":^50s}', '-'*20)
	with open(dst_path) as f:
		timestamps = []
		dsts = []
		for line in f:
			year = f'{line[14:16]}{line[3:5]}'
			month = line[5:7]
			date = line[8:10]
			
			for hr, col in enumerate(range(20, 116, 4)):
				dst = int(line[col:col+4])
				timestamp = int(datetime.strptime(f'{year}-{month}-{date} {hr}-00-00.00', '%Y-%m-%d %H-%M-%S.%f').timestamp())
				if dst=='9999':
					continue
				dsts.append(dst)
				timestamps.append(timestamp)
	return timestamps, dsts

def save_dst(epoch_paths:List[str], dst_data:Tuple[int, int]):
	print('-'*20, f'{"save dst data":^50s}', '-'*20)
	for epoch_path in epoch_paths:
		timestamps, dsts = dst_data

		data = torch.load(f'Data/processed-date/{epoch_path}')
		epoch = data['date'] 													# (num_win, seq_len)
		epoch /= 1000 															# millisec -> sec

		interp_f = interp1d(timestamps, dsts, axis=0)
		interp_dst = np.array([interp_f(e) for e in epoch])						# (num_win, seq_len)
		interp_dst = torch.tensor(interp_dst).float()
		data['dst'] = interp_dst
		torch.save(data, f'Data/processed-date/dst_{epoch_path}')

if __name__=='__main__':
	dst_path = 'Data/DST_2000_2022.txt'
	dst_data = get_dst(dst_path)

	epoch_paths = next(os.walk('Data/processed-date'))[2]
	epoch_paths = [n for n in epoch_paths if 'dst' not in n]
	save_dst(epoch_paths, dst_data)
		

	
	