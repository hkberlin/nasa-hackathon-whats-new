import wget
import cdflib
import os
import matplotlib.pyplot as plt

PYCDF_PATH = "/Applications/cdf/cdf38_1-dist"
os.environ["CDF_LIB"] = PYCDF_PATH
from spacepy import pycdf


cdf_path = 'Data/wind-swe-swe_h1/wi_h1_swe_20220102_v01.cdf'

cdf_pycdf = pycdf.CDF(cdf_path)
print(cdf_pycdf)
# print(cdf_pycdf["BX"][:])


plt.plot(cdf_pycdf["Alpha_VZ_nonlin"])
plt.plot(cdf_pycdf["Proton_VZ_nonlin"])
# plt.ylim([-500, 500])
plt.show()
print(cdf_pycdf["Proton_Np_moment"].type())

# for attr in dir(cdf_pycdf):
    # print(attr)


