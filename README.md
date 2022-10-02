## downloadMultiprocessing.py
> This python file can help you to download the .cdf files from official website. Besides, to speedup the whole process, you may raise the ```processess```. ```download_dataset, data_name, min_version and max_version``` can help you specify the file you want to download.

## preparingData.py
> The time-related data should have long enough time to demonstrate its features. Thus, this file can that you concatenate the .cdf files in to one and return one numerical equivalent .npy files. You can also using ```key``` to choose the features you (or your research interested.)

## prepocessing.py

> Due to there's lots of uncertainty when collecting data in space. People usually do some tricks to avoid the result be distorted when the data contain some noise or some unreasonable value. Then this file is what we designed for dealing with it. The method include data smoothening such as SMA or EMA or estimating the similarity between data. Plus, DSCOVR/mag, Wind/mfi and Wind/swe have different time resolution, we also adjust in this file.

## model1 & model2
> This algorithm can predict the Proton (solar wind ion) density, thermal speed, and velocity vector (n, w, v) from DSCOVR/mag Magnetic Field Data Sets information. The simplified pipeline will be descripted below.
>### model1
>>  First, using DSCOVR/mag data to predict Wind/mfi data since they both detect magnetic field data and have some complicated time-related, position-related relation.
>### model2
>> Secondly, using Wind/mfi data to predict Wind/swe data (n, w, v.) Very simple and clear.

## model3
> We found that DST (Disturbance Storm Time) is somehow related to solar storm. "A negative Dst value means that Earth's magnetic field is weakened. This is particularly the case during solar storms." from Wikipedia said. Thus, we want to use DSCOVR/mag magnetic field data to forecast the probability of the extreme solar storm.