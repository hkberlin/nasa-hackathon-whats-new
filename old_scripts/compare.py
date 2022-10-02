import wget
import cdflib
import os
import matplotlib.pyplot as plt
import numpy as np

PYCDF_PATH = "/Applications/cdf/cdf38_1-dist"
os.environ["CDF_LIB"] = PYCDF_PATH
from spacepy import pycdf

date = "20220101"

mfi_path = f'Data/wind-mfi-mfi_h2/wi_h2_mfi_{date}_v04.cdf'
dscvr_path = f'Data/dscovr-h0-mag/dscovr_h0_mag_{date}_v01.cdf'

mfi_cdf = pycdf.CDF(mfi_path)
dscvr_cdf = pycdf.CDF(dscvr_path)

# data
# data_mfi = mfi_cdf["BGSE"][:, 0]
# data_dscvr = dscvr_cdf["B1GSE"][:, 0]
data_mfi = mfi_cdf["BGSE"][:]
data_dscvr = dscvr_cdf["B1GSE"][:]

# filter
data_dscvr[data_dscvr < -1e+10] = 0
data_dscvr[data_dscvr > 1e+10] = 0

# timescale
time_mfi = np.arange(data_mfi.shape[0]) / data_mfi.shape[0]
time_dscvr = np.arange(data_dscvr.shape[0]) / data_dscvr.shape[0]



# fig, axs = plt.subplots(2, 1, figsize=(12, 6))
# axs[0].plot(time_mfi, data_mfi)
# axs[1].plot(time_dscvr, data_dscvr)

# axs[0].grid()
# axs[1].grid()
# axs[0].set_title(f"Wind (mfi) timestep={data_mfi.shape[0]}")
# axs[1].set_title(f"DSCOVR  timestep={data_dscvr.shape[0]}")
# # # plt.ylim([-500, 500])
# plt.show()




fig, axs = plt.subplots(3, 1, figsize=(12, 7))
# Bx
axs[0].plot(time_mfi, data_mfi[:, 0], linewidth=1, color='k', label='Wind')
axs[0].plot(time_dscvr, data_dscvr[:, 0], linewidth=1, color='red', label='DSCOVR')
axs[0].grid()
axs[0].legend()
axs[0].set_ylabel("x")

# By
axs[1].plot(time_mfi, data_mfi[:, 1], linewidth=1, color='k', label='Wind')
axs[1].plot(time_dscvr, data_dscvr[:, 1], linewidth=1, color='red', label='DSCOVR')
axs[1].grid()
axs[1].legend()
axs[1].set_ylabel("By")

# Bz
axs[2].plot(time_mfi, data_mfi[:, 2], linewidth=1, color='k', label='Wind')
axs[2].plot(time_dscvr, data_dscvr[:, 2], linewidth=1, color='red', label='DSCOVR')
axs[2].grid()
axs[2].legend()
axs[2].set_ylabel("Bz")

fig.supxlabel('Year')
fig.supylabel('Percent Degrees Awarded To Women')

plt.show()