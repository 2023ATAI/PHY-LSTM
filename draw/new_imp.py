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
mask = two_dim_lon_transform(mask)


out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/' + modelname1 +'/focast_time '+ str(0) +'/'
y_pred = np.load(out_path+'_predictions.npy')
y_pred = lon_transform(y_pred)
y_test = np.load(out_path+'observations.npy')
y_test = lon_transform(y_test)
print('y_pred is',y_pred[0])

mask[-int(mask.shape[0]/5.4):,:]=0
min_map = np.min(y_test,axis=0)
max_map = np.max(y_test,axis=0)
mask[min_map==max_map] = 0

name_test = 'Observations'
pltday =  135 # used for plt spatial distributions at 'pltday' day
out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/'+modelname1+'/focast_time '+ str(0) +'/'
r_lstm  = np.load(out_path+'r_'+ modelname1 +'.npy')
r_lstm = two_dim_lon_transform(r_lstm)
# urmse_lstm  = np.load(out_path+'urmse_'+ modelname1 +'.npy')
# urmse_lstm = two_dim_lon_transform(urmse_lstm)
rmse_lstm  = np.load(out_path+'rmse_'+ modelname1 +'.npy')
rmse_lstm = two_dim_lon_transform(rmse_lstm)
r2_lstm  = np.load(out_path+'r2_'+ modelname1 +'.npy')
r2_lstm = two_dim_lon_transform(r2_lstm)
# print('the average ubrmse of '+modelname1+' model is :',np.nanmedian(urmse_lstm[mask==1]))
print('the average r of '+modelname1+' model is :',np.nanmedian(r_lstm[mask==1]))
print('the average rmse of '+modelname1+' model is :',np.nanmedian(rmse_lstm[mask==1]))
print('the average r2 of '+modelname1+' model is :',np.nanmedian(r2_lstm[mask==1]))
out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/'+modelname2+'/focast_time '+ str(0) +'/'
r_mc  = np.load(out_path+'r_'+ modelname2 +'.npy')
r_mc = two_dim_lon_transform(r_mc)
# urmse_mc  = np.load(out_path+'urmse_'+ modelname2 +'.npy')
# urmse_mc = two_dim_lon_transform(urmse_mc)
rmse_mc  = np.load(out_path+'rmse_'+ modelname2 +'.npy')
rmse_mc = two_dim_lon_transform(rmse_mc)
r2_mc  = np.load(out_path+'r2_'+ modelname2 +'.npy')
r2_mc = two_dim_lon_transform(r2_mc)
out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/'+modelname3+'/focast_time '+ str(0) +'/'
r_wb  = np.load(out_path+'r_'+ modelname3 +'.npy')
r_wb = two_dim_lon_transform(r_wb)
# urmse_mc  = np.load(out_path+'urmse_'+ modelname2 +'.npy')
# urmse_mc = two_dim_lon_transform(urmse_mc)
rmse_wb  = np.load(out_path+'rmse_'+ modelname3 +'.npy')
rmse_wb = two_dim_lon_transform(rmse_wb)
r2_wb  = np.load(out_path+'r_'+ modelname3 +'.npy')
r2_wb = two_dim_lon_transform(r2_wb)


# print('the average ubrmse of '+ modelname2 +' model is :',np.nanmedian(urmse_mc[mask==1]))
print('the average r of '+ modelname2 +' model is :',np.nanmedian(r_mc[mask==1]))
print('the average rmse of '+ modelname2 +' model is :',np.nanmedian(rmse_mc[mask==1]))
print('the average r2 of '+ modelname2 +' model is :',np.nanmedian(r2_mc[mask==1]))


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

rmse_improvement1 = (rmse_lstm-rmse_mc)/np.abs(rmse_lstm)
# ubrmse_improvement = (urmse_lstm-urmse_mc)/np.abs(urmse_lstm)
r_improvement1 = (r_mc-r_lstm)/np.abs(r_lstm)
r2_improvement1 = (r2_lstm-r2_mc)/np.abs(r2_lstm)
rmse_improvement2 = (rmse_lstm-rmse_wb)/np.abs(rmse_lstm)
# ubrmse_improvement = (urmse_lstm-urmse_mc)/np.abs(urmse_lstm)
r_improvement2 = (r_wb-r_lstm)/np.abs(r_lstm)
r2_improvement2 = (r2_lstm-r2_wb)/np.abs(r2_lstm)

lon, lat = np.meshgrid(lon_, lat_)
m = Basemap()
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(-90.,90,18.)
meridians = np.arange(-180.,180.,36.)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
xi, yi = m(lon, lat)

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.subplots_adjust(hspace=0.01, wspace=0.13)
# plot.set_label('Improvement in phy(%)')
cs = axes[0,0].contourf(xi,yi, r2_improvement1, np.arange(-1, 1.1, 0.1), cmap='RdBu')
axes[0,0].set_label('Improvement in phy(%)')
cbar = m.colorbar(cs, ax=axes[0,0], location='bottom', pad="10%")
axes[0,0].set_title('(a)', loc='left')
axes[0,0].set_title('Improvement in R(%)', loc='center')
axes[0,0].set_ylabel('PHYs-LSTM Model')

cs = axes[0,1].contourf(xi,yi, rmse_improvement1, np.arange(-1, 1.1, 0.1), cmap='RdBu')
axes[0,1].set_label('Improvement in phy(%)')
cbar = m.colorbar(cs, ax=axes[0,1], location='bottom', pad="10%")
axes[0,1].set_title('(b)', loc='left')
axes[0,1].set_title('Improvement in RMSE(%)', loc='center')

cs = axes[1,0].contourf(xi,yi, r2_improvement2, np.arange(-1, 1.1, 0.1), cmap='RdBu')
axes[1,0].set_label('Improvement in phy(%)')
cbar = m.colorbar(cs, ax=axes[1,0], location='bottom', pad="10%")
axes[1,0].set_title('(d)', loc='left')
axes[1,0].set_ylabel('WB-LSTM Model')

cs = axes[1,1].contourf(xi,yi, rmse_improvement2, np.arange(-1, 1.1, 0.1), cmap='RdBu')
axes[1,1].set_label('Improvement in phy(%)')
cbar = m.colorbar(cs, ax=axes[1,1], location='bottom', pad="10%")
axes[1,1].set_title('(e)', loc='left')

# fig.suptitle('Improvement in phy(%)',fontsize=20 )
fig.suptitle('1d prediction',fontsize=20 )
plt.tight_layout()
plt.show()
