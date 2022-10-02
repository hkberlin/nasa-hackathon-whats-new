import wget
import cdflib
import os
import matplotlib.pyplot as plt

PYCDF_PATH = "/Applications/cdf/cdf38_1-dist"
os.environ["CDF_LIB"] = PYCDF_PATH
from spacepy import pycdf


cdf_path = 'Data/wind-mfi-mfi_h2/wi_h2_mfi_20220101_v04.cdf'

cdf_pycdf = pycdf.CDF(cdf_path)
print(cdf_pycdf)
# print(cdf_pycdf["Time"][:])
print(cdf_pycdf["Time1_PB5"][:])


# plt.plot(cdf_pycdf["BGSE"][:, 0], linewidth=2)
# plt.plot(cdf_pycdf["BGSM"][:, 0], linewidth=1)
# # plt.ylim([-500, 500])
# plt.show()
# print(cdf_pycdf["BGSE"].type())

# for attr in dir(cdf_pycdf):
    # print(attr)


