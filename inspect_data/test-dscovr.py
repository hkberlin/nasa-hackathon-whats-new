import wget
import cdflib
import os
import matplotlib.pyplot as plt

PYCDF_PATH = "/Applications/cdf/cdf38_1-dist"
os.environ["CDF_LIB"] = PYCDF_PATH
from spacepy import pycdf


cdf_path = '../Data/dscovr-h0-mag/dscovr_h0_mag_20220101_v01.cdf'

cdf_pycdf = pycdf.CDF(cdf_path)
# print(cdf_pycdf)
print(cdf_pycdf["B1GSE"].attrs)
# print(cdf_pycdf["label_bgse"][:])


data = cdf_pycdf["B1GSE"][:, 0]
data[data < -1e+10] = 0
data[data > 1e+10] = 0
plt.plot(data, linewidth=2)
# plt.plot(cdf_pycdf["BGSM"][:, 0], linewidth=1)
# plt.ylim([-2, 2])
plt.show()

# for attr in dir(cdf_pycdf):
    # print(attr)


