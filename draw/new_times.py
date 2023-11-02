import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from config import get_args
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.basemap import Basemap
import os


def lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :, :int(x.shape[2] / 2)] = x[:, :, int(x.shape[2] / 2):]
    x_new[:, :, int(x.shape[2] / 2):] = x[:, :, :int(x.shape[2] / 2)]
    return x_new


def two_dim_lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :int(x.shape[1] / 2)] = x[:, int(x.shape[1] / 2):]
    x_new[:, int(x.shape[1] / 2):] = x[:, :int(x.shape[1] / 2)]
    return x_new

modelname1 = 'LSTM'
modelname2 = 'MCLSTM'
modelname3 = 'WBLSTM'
cfg = get_args()
PATH = cfg['inputs_path']+cfg['product']+'/' + str(cfg["spatial_resolution"]) +'/'+str(cfg['selected_year'][0])+'/'
file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
sea_mask = np.load(PATH + file_name_mask)
sea_mask = two_dim_lon_transform(sea_mask)
out_path_lstm = cfg['inputs_path']+cfg['product']+'/' + str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname1+"/focast_time " + str(cfg["forcast_time"]) + "/"
y_pred = np.load(out_path_lstm + '_predictions.npy')
y_pred = lon_transform(y_pred)

y_test = np.load(out_path_lstm + 'observations.npy')
y_test = lon_transform(y_test)

out_path_kde =  cfg['inputs_path']+cfg['product']+'/'+ str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname2+"/focast_time " + str(cfg["forcast_time"]) + "/"
y_pred1 = np.load(out_path_kde + '_predictions.npy')
y_pred1 = lon_transform(y_pred1)

out_path_wb =  cfg['inputs_path']+cfg['product']+'/'+ str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname3+"/focast_time " + str(cfg["forcast_time"]) + "/"
y_pred2 = np.load(out_path_wb + '_predictions.npy')
y_pred2 = lon_transform(y_pred2)

sea_mask[-int(sea_mask.shape[0] / 5.4):, :] = 0
min_map = np.min(y_test, axis=0)
max_map = np.max(y_test, axis=0)
sea_mask[min_map == max_map] = 0

PATH = cfg['inputs_path']+cfg['product']+'/' + str(cfg["spatial_resolution"]) +'/'+str(cfg['selected_year'][0])+'/'

lat_file_name = 'lat_{s}.npy'.format(s=cfg['spatial_resolution'])
lon_file_name = 'lon_{s}.npy'.format(s=cfg['spatial_resolution'])

# gernate lon and lat
lat_ = np.load(PATH + lat_file_name)
lon_ = np.load(PATH + lon_file_name)
lon_ = np.linspace(-180, 179, int(y_pred.shape[2]))

climates_mask = np.load(PATH+"climates_mask_" + str(cfg["spatial_resolution"]) + ".npy")
mask = sea_mask * climates_mask
sites_lon_index = []
sites_lat_index = []

for i in range(5):
    climates = i + 1
    a = np.where(mask == climates)
    lat = a[0]
    lon = a[1]
    high = int(lon.size) + 1
    index = np.random.randint(0, high, 1)
    sites_lon_index.append(lon[index])
    sites_lat_index.append(lat[index])

plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap()
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(-90., 90, 18.)
meridians = np.arange(-180., 180., 36.)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
xi, yi = m(lon, lat)
for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
    # ndarray
    lon = lon_[int(lon_index)]
    lat = lat_[int(lat_index)]
    plt.plot(lon, lat, marker='*', color='red', markersize=9)
plt.legend(loc=0)
plt.tight_layout()
plt.show()

data_all = [y_test, y_pred, y_pred1, y_pred2]  # y_pred_process
color_list = ['black', 'blue', 'red', 'green']  # red
# name_plt5 = ['ERA5-Land values',cfg['modelname'],'process-based']
fig, axs = plt.subplots(5, 1,figsize=(15, 10))
# plt.subplots_adjust(top=0.95)

# fig, axs = plt.subplots(5, 1, figsize=(15, 7.5))
count1 = 0
for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
    count2 = 0
    # fig, axs = plt.subplots(1, 1, figsize=(15, 2))
    print('lat is {lat_v} and lon is {ln_v:.1f}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
    for data_f5plt in (data_all):
        axs[count1].plot(data_f5plt[:, lat_index, lon_index], color=color_list[count2])  # label=name_plt5[count]
        axs[count1].legend(loc=1)
        count2 = count2 + 1
    axs[count1].set_title('lat is {lat_v} and lon is {ln_v:.1f}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]),fontsize = 16)
    # axs[count1].text('lat is {lat_v} and lon is {ln_v:.1f}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
    count1 = count1 + 1
fig.tight_layout()
plt.show()
print('Figure 6： time series plot completed!')


print("----")
