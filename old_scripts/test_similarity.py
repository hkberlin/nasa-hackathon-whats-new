import wget
import cdflib
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


PYCDF_PATH = "/Applications/cdf/cdf38_1-dist"
os.environ["CDF_LIB"] = PYCDF_PATH
from spacepy import pycdf

date = "20220101"

mfi_path = f'Data/wind-mfi-mfi_h2/wi_h2_mfi_{date}_v04.cdf'
dscvr_path = f'Data/dscovr-h0-mag/dscovr_h0_mag_{date}_v01.cdf'

mfi_cdf = pycdf.CDF(mfi_path)
dscvr_cdf = pycdf.CDF(dscvr_path)


def downsample(data:np.array, num_of_day: int=3, frequency: int=34) -> np.array:
	print("start downsampling:", data.shape)
	N = num_of_day*frequency
	epoch = np.arange(0, len(data))
	interp_f = interp1d(epoch, data, axis=0)
	u_epoch = np.linspace(epoch[0], epoch[-1], N)   				# uniform epoch in milliseconds
	return interp_f(u_epoch)



# data
data_mfi = mfi_cdf["BGSE"][:]
data_dscvr = dscvr_cdf["B1GSE"][:]

# filter
data_dscvr[data_dscvr < -1e+10] = 0
data_dscvr[data_dscvr > 1e+10] = 0

# downsampling
data_mfi = downsample(data_mfi, 1, 1440)
data_dscvr = downsample(data_dscvr, 1, 1440)



fig, axs = plt.subplots(3, 1, figsize=(12, 7))
# Bx
axs[0].plot(data_mfi[:, 0], linewidth=1, color='k', label='Wind')
axs[0].plot(data_dscvr[:, 0], linewidth=1, color='red', label='DSCOVR')
axs[0].grid()
axs[0].legend()
axs[0].set_ylabel("x")

# By
axs[1].plot(data_mfi[:, 1], linewidth=1, color='k', label='Wind')
axs[1].plot(data_dscvr[:, 1], linewidth=1, color='red', label='DSCOVR')
axs[1].grid()
axs[1].legend()
axs[1].set_ylabel("By")

# Bz
axs[2].plot(data_mfi[:, 2], linewidth=1, color='k', label='Wind')
axs[2].plot(data_dscvr[:, 2], linewidth=1, color='red', label='DSCOVR')
axs[2].grid()
axs[2].legend()
axs[2].set_ylabel("Bz")

fig.supxlabel('Year')
fig.supylabel('Percent Degrees Awarded To Women')

plt.show()
