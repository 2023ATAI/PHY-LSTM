import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

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

# configures
cfg = get_args()
PATH = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/'
file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
mask = np.load(PATH+file_name_mask)
mask = two_dim_lon_transform(mask)


out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/' + modelname1 +'/focast_time '+ str(cfg['forcast_time']) +'/'
y_pred = np.load(out_path+'_predictions.npy')
y_pred = lon_transform(y_pred)
y_test = np.load(out_path+'observations.npy')
y_test = lon_transform(y_test)
# print('y_pred is',y_pred[0])

mask[-int(mask.shape[0]/5.4):,:]=0
min_map = np.min(y_test,axis=0)
max_map = np.max(y_test,axis=0)
mask[min_map==max_map] = 0

name_test = 'Observations'
pltday =  135 # used for plt spatial distributions at 'pltday' day
out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/'+modelname1+'/focast_time '+ str(cfg['forcast_time']) +'/'
r_lstm  = np.load(out_path+'r_'+ modelname1 +'.npy')
r_lstm = two_dim_lon_transform(r_lstm)
urmse_lstm  = np.load(out_path+'urmse_'+ modelname1 +'.npy')
urmse_lstm = two_dim_lon_transform(urmse_lstm)
rmse_lstm  = np.load(out_path+'rmse_'+ modelname1 +'.npy')
rmse_lstm = two_dim_lon_transform(rmse_lstm)
bias_lstm  = np.load(out_path+'bias_'+ modelname1 +'.npy')
bias_lstm = two_dim_lon_transform(bias_lstm)
# print('the average ubrmse of '+modelname1+' model is :',np.nanmedian(urmse_lstm[mask==1]))
# print('the average r of '+modelname1+' model is :',np.nanmedian(r_lstm[mask==1]))
# print('the average rmse of '+modelname1+' model is :',np.nanmedian(rmse_lstm[mask==1]))
# print('the average bias of '+modelname1+' model is :',np.nanmedian(bias_lstm[mask==1]))
print('the average ubrmse of '+modelname1+' model is :',np.nanmean(urmse_lstm[mask==1]))
print('the average r of '+modelname1+' model is :',np.nanmean(r_lstm[mask==1]))
print('the average rmse of '+modelname1+' model is :',np.nanmean(rmse_lstm[mask==1]))
print('the average bias of '+modelname1+' model is :',np.nanmean(bias_lstm[mask==1]))
out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/'+modelname2+'/focast_time '+ str(cfg['forcast_time']) +'/'
r_mc  = np.load(out_path+'r_'+ modelname2 +'.npy')
r_mc = two_dim_lon_transform(r_mc)
urmse_mc  = np.load(out_path+'urmse_'+ modelname2 +'.npy')
urmse_mc = two_dim_lon_transform(urmse_mc)
rmse_mc  = np.load(out_path+'rmse_'+ modelname2 +'.npy')
rmse_mc = two_dim_lon_transform(rmse_mc)
bias_mc  = np.load(out_path+'bias_'+ modelname2 +'.npy')
bias_mc = two_dim_lon_transform(bias_mc)

# print('the average ubrmse of '+ modelname2 +' model is :',np.nanmedian(urmse_mc[mask==1]))
# print('the average r of '+ modelname2 +' model is :',np.nanmedian(r_mc[mask==1]))
# print('the average rmse of '+ modelname2 +' model is :',np.nanmedian(rmse_mc[mask==1]))
# print('the average bias of '+ modelname2 +' model is :',np.nanmedian(bias_mc[mask==1]))
print('the average ubrmse of '+ modelname2 +' model is :',np.nanmean(urmse_mc[mask==1]))
print('the average r of '+ modelname2 +' model is :',np.nanmean(r_mc[mask==1]))
print('the average rmse of '+ modelname2 +' model is :',np.nanmean(rmse_mc[mask==1]))
print('the average bias of '+ modelname2 +' model is :',np.nanmean(bias_mc[mask==1]))


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

rmse_improvement = (rmse_lstm-rmse_mc)/np.abs(rmse_lstm)
ubrmse_improvement = (urmse_lstm-urmse_mc)/np.abs(urmse_lstm)
r_improvement = (r_mc-r_lstm)/np.abs(r_lstm)
bias_improvement = (bias_lstm-bias_mc)/np.abs(bias_lstm)
print('the max rmse_improvement of '+ modelname2 +' model is :',np.nanmax(rmse_improvement[mask==1]))
print('the max ubrmse_improvement of '+ modelname2 +'model is :',np.nanmax(ubrmse_improvement[mask==1]))
print('the max r_improvement of '+ modelname2 +' model is :',np.nanmax(r_improvement[mask==1]))
print('the max bias_improvement of '+ modelname2 +' model is :',np.nanmax(bias_improvement[mask==1]))
print('-------------------------------------------------------------')
print('the avg rmse_improvement of '+ modelname2 +' model is :',np.nanmean(rmse_improvement[mask==1]))
print('the avg ubrmse_improvement of '+ modelname2 +' model is :',np.nanmean(ubrmse_improvement[mask==1]))
print('the avg r_improvement of '+ modelname2 +' model is :',np.nanmean(r_improvement[mask==1]))
print('the avg bias_improvement of '+ modelname2 +' model is :',np.nanmean(bias_improvement[mask==1]))
print('-------------------------------------------------------------')
print('the median rmse_improvement of '+ modelname2 +' model is :',np.nanmedian(rmse_improvement[mask==1]))
print('the median ubrmse_improvement of '+ modelname2 +' model is :',np.nanmedian(ubrmse_improvement[mask==1]))
print('the median r_improvement of '+ modelname2 +' model is :',np.nanmedian(r_improvement[mask==1]))
print('the median bias_improvement of '+ modelname2 +' model is :',np.nanmedian(bias_improvement[mask==1]))
# ---------------------------------
# 1: rmse_improvement
# ---------------------------------
plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap()
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(-90.,90,18.)
meridians = np.arange(-180.,180.,36.)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
xi, yi = m(lon, lat)



if cfg['label'] == ["volumetric_soil_water_layer_1"]:
	cs = m.contourf(xi,yi, rmse_improvement, np.arange(-1, 1.1, 0.1), cmap='RdBu')
	cbar = m.colorbar(cs, location='bottom', pad="10%")
	cbar.set_label('RMSE(m$^{3}$/m$^{3}$)')
plt.title('Improvement in RMSE(%)')
	#plt.savefig(out_path + 'urmse_'+ cfg['modelname'] + '_spatial distributions.png')
print('Figure 8: spatial distributions for rmse_improvement completed!')
plt.show()



# ---------------------------------
# 2: ubrmse_improvement
# ---------------------------------
plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap()
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(-90.,90,18.)
meridians = np.arange(-180.,180.,36.)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
xi, yi = m(lon, lat)



if cfg['label'] == ["volumetric_soil_water_layer_1"]:
	cs = m.contourf(xi,yi, ubrmse_improvement, np.arange(-1, 1.1, 0.1), cmap='RdBu')
	cbar = m.colorbar(cs, location='bottom', pad="10%")
	cbar.set_label('ubRMSE(m$^{3}$/m$^{3}$)')
plt.title('Improvement in ubRMSE(%)')
	#plt.savefig(out_path + 'urmse_'+ cfg['modelname'] + '_spatial distributions.png')
print('Figure 8: spatial distributions for ubrmse_improvement completed!')
plt.show()

# ---------------------------------
# 3: r_improvement
# ---------------------------------
plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap()
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(-90.,90,18.)
meridians = np.arange(-180.,180.,36.)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
xi, yi = m(lon, lat)



if cfg['label'] == ["volumetric_soil_water_layer_1"]:
	cs = m.contourf(xi,yi, r_improvement, np.arange(-1, 1.1, 0.1), cmap='RdBu')
	cbar = m.colorbar(cs, location='bottom', pad="10%")
	cbar.set_label('R NAN ("1" is NAN in land region )')
plt.title('Improvement in R(%)')
	#plt.savefig(out_path + 'urmse_'+ cfg['modelname'] + '_spatial distributions.png')
print('Figure 8: spatial distributions for r_improvement completed!')
plt.show()

# ---------------------------------
# 4: bias_improvement
# ---------------------------------
plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap()
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(-90.,90,18.)
meridians = np.arange(-180.,180.,36.)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
xi, yi = m(lon, lat)

if cfg['label'] == ["volumetric_soil_water_layer_1"]:
	cs = m.contourf(xi,yi, bias_improvement, np.arange(-1, 1.1, 0.1), cmap='RdBu')
	cbar = m.colorbar(cs, location='bottom', pad="10%")
	cbar.set_label('Bias(m$^{3}$/m$^{3}$)')
plt.title('Improvement in Bias(%)')
	#plt.savefig(out_path + 'urmse_'+ cfg['modelname'] + '_spatial distributions.png')
print('Figure 8: spatial distributions for bias_improvement completed!')
plt.show()
