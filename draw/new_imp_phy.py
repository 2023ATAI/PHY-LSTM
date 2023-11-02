import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.basemap import Basemap
import os

import numpy as np
from config import get_args
# ---------------------------------# ---------------------------------

def lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:,:int(x.shape[2]/2)] = x[:,:,int(x.shape[2]/2):] 
  x_new[:,:,int(x.shape[2]/2):] = x[:,:,:int(x.shape[2]/2)] 
  return x_new

def two_dim_lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:int(x.shape[1]/2)] = x[:,int(x.shape[1]/2):] 
  x_new[:,int(x.shape[1]/2):] = x[:,:int(x.shape[1]/2)] 
  return x_new

# ~~~~~~~~~~~~~~~~~~~~modelname~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CNN、PHYCNN、LSTM、PHYLSTM、ConvLSTM、PHYConvLSTM
modelname1 = 'LSTM'
modelname2 = 'PHYLSTM'
modelname3 = 'PHYsLSTM'

# configures
cfg = get_args()
PATH = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/'
file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
mask = np.load(PATH+file_name_mask)
# np.savetxt('mask1.csv',mask,delimiter=',')
# 平移地图
mask = two_dim_lon_transform(mask)
# np.savetxt('mask1.csv',mask,delimiter=',')
# 除去南极大陆
mask[-int(mask.shape[0]/5.4):,:]=0
# np.savetxt('mask2.csv',mask,delimiter=',')
out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/' + modelname1 +'/focast_time '+ str(0) +'/'
# 读取结果并且平移地图
y_pred = np.load(out_path+'_predictions.npy')
y_pred = lon_transform(y_pred)
y_test = np.load(out_path+'observations.npy')
y_test = lon_transform(y_test)
print('y_pred is',y_pred[0])
# 删除没有变化的点
min_map = np.min(y_test,axis=0)
max_map = np.max(y_test,axis=0)
mask[min_map==max_map] = 0
# 读取dw,old_sm
out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/'
dww_train = np.load(out_path + '/' + 'dww_test.npy', mmap_mode='r')
if 'LandBench1' in cfg['workname']:
  dw_test = dww_train[0, :, :, :]
elif 'LandBench2' in cfg['workname']:
  dw_test = dww_train[1, :, :, :]
elif 'LandBench3' in cfg['workname']:
  dw_test = dww_train[2, :, :, :]
elif 'LandBench4' in cfg['workname']:
  dw_test = dww_train[3, :, :, :]
elif 'LandBench5' in cfg['workname']:
  dw_test = dww_train[4, :, :, :]
elif 'LandBench6' in cfg['workname']:
  dw_test = dww_train[5, :, :, :]
elif 'LandBench7' in cfg['workname']:
  dw_test = dww_train[6, :, :, :]
elif 'LandBench8' in cfg['workname']:
  dw_test = dww_train[7, :, :, :]

dw_test = dw_test[-(y_test.shape[0]):,:,:,0]
dw_test = lon_transform(dw_test)
dw_test[0,:,:] = 0
out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/'
old_sm = np.zeros(y_test.shape)
old_sm[1:,:,:] = y_test[:-1,:,:]
# 计算lstm phy
phy_lstm = dw_test*(old_sm-y_pred)
phy_lstm[:,mask==0]=0
phy_lstm[phy_lstm>0]=1
phy_lstm[phy_lstm!=1]=0
phy_lstm = np.sum(phy_lstm,axis=0)
name_test = 'Observations'
# pltday =  135 # used for plt spatial distributions at 'pltday' day

print('the average ubrmse of '+modelname1+' model is :',np.nanmedian(phy_lstm[mask==1]))
out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/'+modelname2+'/focast_time '+ str(0) +'/'
# 计算mclstm phy
y_pred = np.load(out_path+'_predictions.npy')
y_pred = lon_transform(y_pred)
phy_mc = dw_test*(old_sm-y_pred)
phy_mc[:,mask==0]=0
phy_mc[phy_mc>0]=1
phy_mc[phy_mc!=1]=0
phy_mc = np.sum(phy_mc,axis=0)
print('the average ubrmse of '+ modelname2 +' model is :',np.nanmedian(phy_mc[mask==1]))


out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/'+modelname3+'/focast_time '+ str(0) +'/'
# 计算mclstm phy
y_pred = np.load(out_path+'_predictions.npy')
y_pred = lon_transform(y_pred)
phy_wb = dw_test*(old_sm-y_pred)
phy_wb[:,mask==0]=0
phy_wb[phy_wb>0]=1
phy_wb[phy_wb!=1]=0
phy_wb = np.sum(phy_wb,axis=0)
print('the average ubrmse of '+ modelname3 +' model is :',np.nanmedian(phy_wb[mask==1]))

PATH = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'+str(cfg['selected_year'][0])+'/'

lat_file_name = 'lat_{s}.npy'.format(s=cfg['spatial_resolution'])
lon_file_name = 'lon_{s}.npy'.format(s=cfg['spatial_resolution'])

# gernate lon and lat
lat_ = np.load(PATH+lat_file_name)
lon_ = np.load(PATH+lon_file_name)
lon_ = np.linspace(-180,179,int(y_pred.shape[2]))
#print(lon_)
# Figure 6： configure for time series plot
sites_lon_index=[100,100,100,100,100]
sites_lat_index=[55,50,45,43,42]

phy_improvement1 = (phy_lstm-phy_mc)/np.abs(phy_lstm)
phy_improvement2 = (phy_lstm-phy_wb)/np.abs(phy_lstm)
out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/' + modelname1 +'/focast_time '+ str(0) +'/'
# 读取结果并且平移地图
y_pred = np.load(out_path+'_predictions.npy')
y_pred = lon_transform(y_pred)
y_test = np.load(out_path+'observations.npy')
y_test = lon_transform(y_test)
print('y_pred is',y_pred[0])
# 删除没有变化的点
min_map = np.min(y_test,axis=0)
max_map = np.max(y_test,axis=0)
mask[min_map==max_map] = 0
# 读取dw,old_sm
# out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/'
# dw_test = np.load(out_path+'dw_test.npy')
# dw_test = dw_test[-(y_test.shape[0]):,:,:,0]
# dw_test = lon_transform(dw_test)
# dw_test[0,:,:] = 0
# out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/'
# old_sm = np.zeros(y_test.shape)
# old_sm[1:,:,:] = y_test[:-1,:,:]
# # 计算lstm phy
# phy_lstm = dw_test*(old_sm-y_pred)
# phy_lstm[:,mask==0]=0
# phy_lstm[phy_lstm>0]=1
# phy_lstm[phy_lstm!=1]=0
# phy_lstm = np.sum(phy_lstm,axis=0)
# name_test = 'Observations'
# # pltday =  135 # used for plt spatial distributions at 'pltday' day
#
# print('the average ubrmse of '+modelname1+' model is :',np.nanmedian(phy_lstm[mask==1]))
# out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/'+modelname2+'/focast_time '+ str(0) +'/'
# # 计算mclstm phy
# y_pred = np.load(out_path+'_predictions.npy')
# y_pred = lon_transform(y_pred)
# phy_mc = dw_test*(old_sm-y_pred)
# phy_mc[:,mask==0]=0
# phy_mc[phy_mc>0]=1
# phy_mc[phy_mc!=1]=0
# phy_mc = np.sum(phy_mc,axis=0)
# print('the average ubrmse of '+ modelname2 +' model is :',np.nanmedian(phy_mc[mask==1]))
#
#
# out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/'+modelname3+'/focast_time '+ str(0) +'/'
# # 计算mclstm phy
# y_pred = np.load(out_path+'_predictions.npy')
# y_pred = lon_transform(y_pred)
# phy_wb = dw_test*(old_sm-y_pred)
# phy_wb[:,mask==0]=0
# phy_wb[phy_wb>0]=1
# phy_wb[phy_wb!=1]=0
# phy_wb = np.sum(phy_wb,axis=0)
# print('the average ubrmse of '+ modelname3 +' model is :',np.nanmedian(phy_wb[mask==1]))
#
# PATH = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'+str(cfg['selected_year'][0])+'/'
#
# lat_file_name = 'lat_{s}.npy'.format(s=cfg['spatial_resolution'])
# lon_file_name = 'lon_{s}.npy'.format(s=cfg['spatial_resolution'])
#
# # gernate lon and lat
# lat_ = np.load(PATH+lat_file_name)
# lon_ = np.load(PATH+lon_file_name)
# lon_ = np.linspace(-180,179,int(y_pred.shape[2]))
# #print(lon_)
# # Figure 6： configure for time series plot
# sites_lon_index=[100,100,100,100,100]
# sites_lat_index=[55,50,45,43,42]
#
# phy_improvement3 = (phy_lstm-phy_mc)/np.abs(phy_lstm)
# phy_improvement4 = (phy_lstm-phy_wb)/np.abs(phy_lstm)
# print('the max rmse_improvement of '+ modelname2 +' model is :',np.nanmax(phy_improvement1[mask==1]))
# print('-------------------------------------------------------------')
# print('the avg rmse_improvement of '+ modelname2 +' model is :',np.nanmean(phy_improvement1[mask==1]))
# print('-------------------------------------------------------------')
# print('the median rmse_improvement of '+ modelname2 +' model is :',np.nanmedian(phy_improvement1[mask==1]))
# ---------------------------------
# 1: phy_improvement
# ---------------------------------
# plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap()
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(-90.,90,18.)
meridians = np.arange(-180.,180.,36.)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
xi, yi = m(lon, lat)

fig, axes = plt.subplots(1, 2, figsize=(16, 11))
fig.subplots_adjust(hspace=0.01, wspace=0.13)
# plot.set_label('Improvement in phy(%)')
cs = axes[0].contourf(xi,yi, phy_improvement1, np.arange(-1, 1.1, 0.1), cmap='RdBu')
axes[0].set_label('Improvement in phy(%)')
cbar = m.colorbar(cs, ax=axes[0], location='bottom', pad="10%")
axes[0].set_title('(a)', loc='left')
axes[0].set_title('PHY-LSTM Model', loc='center')
axes[0].set_ylabel('')
# metrics = '$\\frac{R(Our\\_Model) - R(LSTM)}{R(Our\\_Model)}$' + '$\\times$' + '100%'
# axes[0,0].text(77, 49, metrics,fontsize=12)

cs = axes[1].contourf(xi,yi, phy_improvement2, np.arange(-1, 1.1, 0.1), cmap='RdBu')
axes[1].set_label('Improvement in phy(%)')
cbar = m.colorbar(cs, ax=axes[1], location='bottom', pad="10%")
axes[1].set_title('(b)', loc='left')
axes[1].set_title('PHYs-LSTM Model', loc='center')

# cs = axes[1,0].contourf(xi,yi, phy_improvement3, np.arange(-1, 1.1, 0.1), cmap='RdBu')
# axes[1,0].set_label('Improvement in phy(%)')
# cbar = m.colorbar(cs, ax=axes[1,0], location='bottom', pad="10%")
# axes[1,0].set_title('(c)', loc='left')
# # axes[1,0].set_title('Improvement in phy(%)', loc='center')
# axes[1,0].set_ylabel('1d predicted SM')
#
# cs = axes[1,1].contourf(xi,yi, phy_improvement4, np.arange(-1, 1.1, 0.1), cmap='RdBu')
# axes[1,1].set_label('Improvement in phy(%)')
# cbar = m.colorbar(cs, ax=axes[1,1], location='bottom', pad="10%")
# axes[1,1].set_title('(d)', loc='left')
# axes[1,1].set_title('Improvement in phy(%)', loc='center')

# m.colorbar(cs, location='bottom', pad="10%")
# im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
#                    cmap="YlGn", cbarlabel="harvest [t/year]")
fig.suptitle('Improvement in phy(%)',fontsize=20 )
plt.tight_layout()
plt.show()