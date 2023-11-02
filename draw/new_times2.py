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
def r2(obs,pre,r2):
    obs[np.isnan(obs)]=0
    pre[np.isnan(pre)]=0
    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            # 真实 - 预测
            a = obs[:,i, j] - pre[:,i, j]
            # 真实 - 平均真实
            b = obs[:,i, j] - np.mean(pre[:,i, j])
            # 预测 - 平均预测
            # r2分子
            f = np.sum(a ** 2)
            # r2分母
            g = np.sum(b ** 2)
            if g != 0:
                r2[i, j] = 1 - f / g
    return r2

modelname1 = 'LSTM'
modelname2 = 'PHYLSTM'
modelname3 = 'PHYsLSTM'
cfg = get_args()
PATH = cfg['inputs_path']+cfg['product']+'/' + str(cfg["spatial_resolution"]) +'/'+str(cfg['selected_year'][0])+'/'
file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
sea_mask = np.load(PATH + file_name_mask)
sea_mask = two_dim_lon_transform(sea_mask)
out_path_lstm = cfg['inputs_path']+cfg['product']+'/' + str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname1+"/focast_time " + str(cfg["forcast_time"]) + "/"
y_pred = np.load(out_path_lstm + '_predictions.npy')
y_pred = lon_transform(y_pred)/ 7 * 100 / 1

y_r2 = np.load(out_path_lstm+'r2_'+modelname1 + '.npy')
y_r2 = two_dim_lon_transform(y_r2)

y_test = np.load(out_path_lstm + 'observations.npy')
y_test = lon_transform(y_test)/ 7 * 100 / 1

out_path_kde =  cfg['inputs_path']+cfg['product']+'/'+ str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname2+"/focast_time " + str(cfg["forcast_time"]) + "/"
y_pred1 = np.load(out_path_kde + '_predictions.npy')
y_pred1 = lon_transform(y_pred1)/ 7 * 100 / 1

y_r21 = np.load(out_path_kde+'r2_'+modelname2 + '.npy')
y_r21 = two_dim_lon_transform(y_r21)

out_path_wb =  cfg['inputs_path']+cfg['product']+'/'+ str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname3+"/focast_time " + str(cfg["forcast_time"]) + "/"
y_pred2 = np.load(out_path_wb + '_predictions.npy')
y_pred2 = lon_transform(y_pred2)/ 7 * 100 / 1

y_r22 = np.load(out_path_wb+'r2_'+modelname3 + '.npy')
y_r22 = two_dim_lon_transform(y_r22)

sea_mask[-int(sea_mask.shape[0] / 5.4):, :] = 0
min_map = np.min(y_test, axis=0)
max_map = np.max(y_test, axis=0)
sea_mask[min_map == max_map] = 0

r21 = np.zeros(y_r2.shape)
r22 = np.zeros(y_r2.shape)
r23 = np.zeros(y_r2.shape)
r21 = r2(y_test,y_pred,r21)
r22 = r2(y_test,y_pred1,r22)
r23 = r2(y_test,y_pred2,r23)
# np.savetxt('r21.csv',r21,delimiter=',')
# np.savetxt('r22.csv',r22,delimiter=',')
# np.savetxt('r23.csv',r23,delimiter=',')

R21 = r22-r21
R22 = r23-r21
ilist1 = []
jlist1 = []
ilist2 = []
jlist2 = []
ilist3 = []
jlist3 = []
for i in range(r21.shape[0]):
    for j in range(r21.shape[1]):
        # if r21[i,j]>0.98:
        #     ilist1.append(i)
        #     jlist1.append(j)
        if r22[i, j] > 0.90:
            if R22[i,j]>0.04:
                ilist2.append(i)
                jlist2.append(j)
        # if r23[i,j]>0.90:
        #     ilist3.append(i)
        #     jlist3.append(j)

PATH = cfg['inputs_path']+cfg['product']+'/' + str(cfg["spatial_resolution"]) +'/'+str(cfg['selected_year'][0])+'/'

lat_file_name = 'lat_{s}.npy'.format(s=cfg['spatial_resolution'])
lon_file_name = 'lon_{s}.npy'.format(s=cfg['spatial_resolution'])

# r2_mc = y_r21 - y_r2
# r2_wb = y_r22 - y_r2

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
    # high = int(lon.size) + 1
    # index = np.random.randint(0, high, 1)
    # sites_lon_index.append(lon[index])
    # sites_lat_index.append(lat[index])
    high = int(len(ilist2)) + 1
    index = np.random.randint(0, high)
    sites_lon_index.append(jlist2[index])
    sites_lat_index.append(ilist2[index])
# sites_lon_index = jlist2
# sites_lat_index = ilist2


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
fig, axs = plt.subplots(5, 1,figsize=(13, 11))
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
