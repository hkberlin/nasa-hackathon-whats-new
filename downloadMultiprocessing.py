import wget
import os
import time
import multiprocessing
from multiprocessing import Pool


# download_dataset = "wind/swe/swe_h1"
# data_name = "wi_h1_swe"
# min_version = 1
# max_version = 1

# download_dataset = "wind/mfi/mfi_h2"
# data_name = "wi_h2_mfi"
# min_version = 3
# max_version = 5

download_dataset = "dscovr/h0/mag"
data_name = "dscovr_h0_mag"
min_version = 1
max_version = 1

# download_dataset = "dscovr/h1/farady_cup"
# data_name = "dscovr_h1_fc"
# data_version = "v12"

start_year = 2015
end_year = 2016

processes = 1

folder = download_dataset.replace("/", "-")
if not os.path.exists(folder):
    os.mkdir(folder)


def download(args):
    cdf_url, cdf_save_path = args
    print("downloading:", cdf_url)
    try:
        wget.download(cdf_url, cdf_save_path)
    except:
        print(f"---Found no date")

def debug():
    p_pool = []
    for year in range(start_year, end_year+1):
        for month in range(1, 13):
            for day in range(1, 32):
                for version in range(min_version, max_version+1):
                    year, month, day = str(year), str(month), str(day)
                    month = "0" + month if len(month) == 1 else month
                    day = "0" + day if len(day) == 1 else day
                    data_version = f"v0{version}"
                    
                    cdf_url = f"https://cdaweb.gsfc.nasa.gov/pub/data/{download_dataset}/{year}/{data_name}_{year}{month}{day}_{data_version}.cdf"
                    cdf_save_path = os.path.join(folder, cdf_url.split("/")[-1])

                    if not os.path.exists(cdf_save_path):
                        # print(cdf_save_path)
                        args = (cdf_url, cdf_save_path)
                        p_pool.append(args)
    return p_pool

# collect processes
def collect_process():
    p_pool = []
    for year in range(start_year, end_year+1):
        for month in range(1, 13):
            for day in range(1, 32):
                for version in range(min_version, max_version+1):
                    year, month, day = str(year), str(month), str(day)
                    month = "0" + month if len(month) == 1 else month
                    day = "0" + day if len(day) == 1 else day
                    data_version = f"v0{version}"
                    
                    cdf_url = f"https://cdaweb.gsfc.nasa.gov/pub/data/{download_dataset}/{year}/{data_name}_{year}{month}{day}_{data_version}.cdf"
                    cdf_save_path = os.path.join(folder, cdf_url.split("/")[-1])

                    args = (cdf_url, cdf_save_path)
                    p_pool.append(args)
    return p_pool



if __name__ == "__main__":
    
    multiprocessing.freeze_support()

    p_pool = collect_process()
    # start running processes
    with Pool(processes=processes) as p:
        for i in p.imap_unordered(download, p_pool):
            time.sleep(0.1)
