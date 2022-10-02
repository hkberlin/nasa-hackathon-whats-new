import wget
import cdflib
import os
import matplotlib.pyplot as plt

PYCDF_PATH = "/Applications/cdf/cdf38_1-dist"
os.environ["CDF_LIB"] = PYCDF_PATH
from spacepy import pycdf


cdf_path = '../Data/dscovr-h1-farady_cup/dscovr_h1_fc_20190101_v12.cdf'

cdf_pycdf = pycdf.CDF(cdf_path)
print(cdf_pycdf)
# print(cdf_pycdf["FLAG1"][:].sum())
# print(cdf_pycdf["label_bgse"][:])


