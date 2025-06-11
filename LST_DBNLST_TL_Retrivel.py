
import os
from osgeo import gdal
from sklearn.preprocessing import MinMaxScaler
from numpy import transpose

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")
import torch
from utils.DBN import DBN
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
seed=0

os.environ['PROJ_LIB'] = r'D:\Anacond3\envs\gdal\Library\share\proj'
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.UseExceptions()


def regularit(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    return newDataFrame


def write(save_path, image, geo, pro):
    dtype = 6
    height = image.shape[1]
    width = image.shape[2]
    im_bands = 1
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(save_path, width, height, im_bands, dtype)
    dataset.SetProjection(pro)
    dataset.SetGeoTransform(geo)
    im_data = np.array(image)
    if im_data.ndim == 2:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def lon_lat_index(geo_dnb, geo_ps):
    M_x_0 = int(abs((geo_dnb[0] - geo_ps[0]) / geo_ps[1]))
    M_y_0 = int(abs((geo_dnb[3] - geo_ps[3]) / geo_ps[5]))
    print(M_x_0, M_y_0)
    return M_x_0, M_y_0


if __name__ == '__main__':

    fanyan_result = r'G:\actual_inversion\result_new\\'

    fanyan_file = r'G:\actual_inversion\picture\LC09_L1TP_115035_20220622_20230410_02_T1\\'

    model_file = r'G:\actual_inversion\model\GuiYihua_mode\\'

    timing = '20220622_'

    Quan = 'Korea_LST_DBN_all_'

    Q = 'all_'


    column = ['elevation', 'slope', 'aspect', 'BT', 'ndvi', 'ndbi', 'mndwi']



    BT_file = fanyan_file + 'QY_BT.tif'
    ndvi_file = fanyan_file + 'QY_ndvi.tif'
    ndbi_file = fanyan_file + 'QY_ndbi.tif'
    mndwi_file = fanyan_file + 'QY_mndwi.tif'


    model_filea = model_file + 'all-Korea_WT.pt'

    save_tiffa = fanyan_result + timing + Quan + 'all-1.tif'


    dataset = gdal.Open(BT_file)
    geo_bt = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    x_bt = dataset.RasterXSize
    y_bt = dataset.RasterYSize
    xx = np.arange(x_bt, dtype=np.float64)
    yy = np.arange(y_bt, dtype=np.float64)
    lon_bt = geo_bt[0] + xx * geo_bt[1]
    lat_bt = geo_bt[3] + yy * geo_bt[5]



    data_all = np.zeros((7, y_bt, x_bt), dtype=np.float64)


    dataset_dem = gdal.Open("G:\\DEM-Korea\\all_Korean_DSM.tif")
    geo_dem = dataset_dem.GetGeoTransform()
    x_1, y_1 = lon_lat_index(geo_bt, geo_dem)
    data_all[0] = (dataset_dem.ReadAsArray())[y_1:y_1 + y_bt, x_1:x_1 + x_bt]

    dataset_slope = gdal.Open("G:\\DEM-Korea\\all_Korean_Slope.tif")

    data_all[1] = (dataset_slope.ReadAsArray())[y_1:y_1 + y_bt, x_1:x_1 + x_bt]

    dataset_aspect = gdal.Open("G:\\DEM-Korea\\all_Korean_Aspect.tif")

    data_all[2] = (dataset_aspect.ReadAsArray())[y_1:y_1 + y_bt, x_1:x_1 + x_bt]

    data_all[3] = dataset.ReadAsArray()

    dataset_ndvi = gdal.Open(ndvi_file)

    data_all[4] = dataset_ndvi.ReadAsArray()

    dataset_ndbi = gdal.Open(ndbi_file)

    data_all[5] = dataset_ndbi.ReadAsArray()

    dataset_mndwi = gdal.Open(mndwi_file)

    data_all[6] = dataset_mndwi.ReadAsArray()

    data_list = []

    for i in range(7):
        da = data_all[i]
        data_list.append(da.reshape(x_bt * y_bt, 1))

    dat = transpose(np.array(data_list))
    text_x = pd.DataFrame(dat.reshape(x_bt * y_bt, 7), columns=column)
    text_x = text_x.to_numpy()

    GYHdata = pd.read_csv('dataset/GYH.csv', header=None)
    GYH_input_data = GYHdata.iloc[:, :-1].values
    GYH_output_data = GYHdata.iloc[:, -1].values.reshape(-1, 1)
    ss_X = MinMaxScaler(feature_range=(0, 1)).fit(GYH_input_data)
    ss_y = MinMaxScaler(feature_range=(0, 1)).fit(GYH_output_data)
    text_x = ss_X.transform(text_x)


    learning_rate = 0.1
    epoch_pretrain = 10

    hidden_units = [20, 15]

    batch_size = 64

    learning_rate_finetune = 0.1
    epoch_finetune = 100
    momentum = 0.9
    tf = 'Sigmoid'
    dbn = DBN(hidden_units, 7, 1,learning_rate=learning_rate,activate=tf, device=device)
    dbn.load_state_dict(torch.load('G:\\actual_inversion\\all-Korea_WT.pt', map_location=device))


    rea = dbn.predict(text_x,batch_size,types=1)
    rea = rea.cpu().numpy()
    rea = ss_y.inverse_transform(rea)


    data_tiff = (np.array(rea)).reshape(1, y_bt, x_bt)
    write(save_tiffa, data_tiff, geo_bt, projection)

