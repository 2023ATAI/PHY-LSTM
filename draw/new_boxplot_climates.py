import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from config import get_args
def lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:,:int(x.shape[2]/2)] = x[:,:,int(x.shape[2]/2):]
  x_new[:,:,int(x.shape[2]/2):] = x[:,:,:int(x.shape[2]/2)]
  return x_new
def lon_transform_2d(x):
  x_new = np.zeros(x.shape)
  x_new[:,:int(x.shape[1]/2)] = x[:,int(x.shape[1]/2):]
  x_new[:,int(x.shape[1]/2):] = x[:,:int(x.shape[1]/2)]
  return x_new
def two_dim_lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:int(x.shape[1]/2)] = x[:,int(x.shape[1]/2):]
  x_new[:,int(x.shape[1]/2):] = x[:,:int(x.shape[1]/2)]
  return x_new
modelname1 = 'LSTM'
modelname2 = 'PHYLSTM'
modelname3 = 'PHYsLSTM'
cfg = get_args()
PATH = cfg['inputs_path']+cfg['product']+'/' + str(cfg["spatial_resolution"]) +'/'+str(cfg['selected_year'][0])+'/'
file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
sea_mask = np.load(PATH+file_name_mask)
sea_mask = two_dim_lon_transform(sea_mask)

out_path_lstm = cfg['inputs_path']+cfg['product']+'/' + str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname1+"/focast_time " + str(cfg["forcast_time"]) + "/"
y_pred = np.load(out_path_lstm+'_predictions.npy')
y_pred = lon_transform(y_pred)

y_test = np.load(out_path_lstm+'observations.npy')
y_test = lon_transform(y_test)

out_path_kde = cfg['inputs_path']+cfg['product']+'/'+ str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname2+"/focast_time " + str(cfg["forcast_time"]) + "/"
y_pred1 = np.load(out_path_kde+'_predictions.npy')
y_pred1 = lon_transform(y_pred1)

out_path_wb = cfg['inputs_path']+cfg['product']+'/'+ str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname3+"/focast_time " + str(cfg["forcast_time"]) + "/"
y_pred2 = np.load(out_path_kde+'_predictions.npy')
y_pred2 = lon_transform(y_pred2)





sea_mask[-int(sea_mask.shape[0]/5.4):,:]=0
min_map = np.min(y_test,axis=0)
max_map = np.max(y_test,axis=0)
sea_mask[min_map==max_map] = 0

PATH = cfg['inputs_path']+cfg['product']+'/' + str(cfg["spatial_resolution"]) +'/'+str(cfg['selected_year'][0])+'/'

lat_file_name = 'lat_{s}.npy'.format(s=cfg['spatial_resolution'])
lon_file_name = 'lon_{s}.npy'.format(s=cfg['spatial_resolution'])

# gernate lon and lat
lat_ = np.load(PATH+lat_file_name)
lon_ = np.load(PATH+lon_file_name)
lon_ = np.linspace(-180,179,int(y_pred.shape[2]))

climates_mask = np.load(PATH+"climates_mask_"+str(cfg["spatial_resolution"])+".npy")
mask = sea_mask*climates_mask


bias_lstm = np.load(cfg['inputs_path']+cfg['product']+'/'+str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname1+"/focast_time " +str(cfg["forcast_time"])+"/r_"+modelname1+".npy")
bias_lstm = lon_transform_2d(bias_lstm)
bias_kde_lstm = np.load(cfg['inputs_path']+cfg['product']+'/'+str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname2+"/focast_time " +str(cfg["forcast_time"])+"/r_"+modelname2+".npy")
bias_kde_lstm = lon_transform_2d(bias_kde_lstm)
bias_wb_lstm = np.load(cfg['inputs_path']+cfg['product']+'/'+str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname3+"/focast_time " +str(cfg["forcast_time"])+"/r_"+modelname3+".npy")
bias_wb_lstm = lon_transform_2d(bias_kde_lstm)

bias_lstm1 = bias_lstm[mask==1]
bias_lstm2 = bias_lstm[mask==2]
bias_lstm3 = bias_lstm[mask==3]
bias_lstm4 = bias_lstm[mask==4]
bias_lstm5 = bias_lstm[mask==5]
bias_kde_lstm1 = bias_kde_lstm[mask==1]
bias_kde_lstm2 = bias_kde_lstm[mask==2]
bias_kde_lstm3 = bias_kde_lstm[mask==3]
bias_kde_lstm4 = bias_kde_lstm[mask==4]
bias_kde_lstm5 = bias_kde_lstm[mask==5]
bias_wb_lstm1 = bias_wb_lstm[mask==1]
bias_wb_lstm2 = bias_wb_lstm[mask==2]
bias_wb_lstm3 = bias_wb_lstm[mask==3]
bias_wb_lstm4 = bias_wb_lstm[mask==4]
bias_wb_lstm5 = bias_wb_lstm[mask==5]

bias_lstm_all =[bias_lstm1,bias_lstm2,bias_lstm3,bias_lstm4,bias_lstm5]
bias_lstm_all = np.concatenate(bias_lstm_all)

bias_kde_lstm_all =[bias_kde_lstm1,bias_kde_lstm2,bias_kde_lstm3,bias_kde_lstm4,bias_kde_lstm5]
bias_kde_lstm_all = np.concatenate(bias_kde_lstm_all)

bias_wb_lstm_all =[bias_wb_lstm1,bias_wb_lstm2,bias_wb_lstm3,bias_wb_lstm4,bias_wb_lstm5]
bias_wb_lstm_all = np.concatenate(bias_wb_lstm_all)

bias_all = np.append(bias_lstm_all,bias_kde_lstm_all)
bias_all = np.append(bias_all,bias_wb_lstm_all)
model_type = np.empty(bias_all.size,dtype=object)
model_type[:bias_lstm_all.size]='LSTM Model'
model_type[bias_lstm_all.size:2*(bias_lstm_all.size)]='PHY-LSTM Model'
model_type[2*(bias_lstm_all.size):]='PHYs-LSTM Model'
climates_type1 = np.empty(bias_lstm_all.size,dtype=object)
a = bias_lstm1.size
b = a + bias_lstm2.size
c = b + bias_lstm3.size
d = c + bias_lstm4.size
e = d + bias_lstm5.size
climates_type1[:a]='Equatorial'
climates_type1[a:b]='Arid'
climates_type1[b:c]='Warm temperate'
climates_type1[c:d]='Snow'
climates_type1[d:]='Polar'
climates_type2 = climates_type1
climates_type3 = climates_type1
climates_type_all = np.append(climates_type1,climates_type2)
climates_type_all = np.append(climates_type_all,climates_type3)
df = pd.DataFrame({"R": pd.Series(bias_all), "climates values": pd.Series(climates_type_all), 'Model': pd.Series(model_type)})
#--------------------------------------------------------------------------------------------------------------------

bias_lstm = np.load(cfg['inputs_path']+cfg['product']+'/'+str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname1+"/focast_time " +str(cfg["forcast_time"])+"/rmse_"+modelname1+".npy")
bias_lstm = lon_transform_2d(bias_lstm)
bias_kde_lstm = np.load(cfg['inputs_path']+cfg['product']+'/'+str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname2+"/focast_time " +str(cfg["forcast_time"])+"/rmse_"+modelname2+".npy")
bias_kde_lstm = lon_transform_2d(bias_kde_lstm)
bias_wb_lstm = np.load(cfg['inputs_path']+cfg['product']+'/'+str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname3+"/focast_time " +str(cfg["forcast_time"])+"/rmse_"+modelname3+".npy")
bias_wb_lstm = lon_transform_2d(bias_wb_lstm)

bias_lstm1 = bias_lstm[mask==1]
bias_lstm2 = bias_lstm[mask==2]
bias_lstm3 = bias_lstm[mask==3]
bias_lstm4 = bias_lstm[mask==4]
bias_lstm5 = bias_lstm[mask==5]
bias_kde_lstm1 = bias_kde_lstm[mask==1]
bias_kde_lstm2 = bias_kde_lstm[mask==2]
bias_kde_lstm3 = bias_kde_lstm[mask==3]
bias_kde_lstm4 = bias_kde_lstm[mask==4]
bias_kde_lstm5 = bias_kde_lstm[mask==5]
bias_wb_lstm1 = bias_wb_lstm[mask==1]
bias_wb_lstm2 = bias_wb_lstm[mask==2]
bias_wb_lstm3 = bias_wb_lstm[mask==3]
bias_wb_lstm4 = bias_wb_lstm[mask==4]
bias_wb_lstm5 = bias_wb_lstm[mask==5]
bias_lstm_all =[bias_lstm1,bias_lstm2,bias_lstm3,bias_lstm4,bias_lstm5]
bias_lstm_all = np.concatenate(bias_lstm_all)
bias_kde_lstm_all =[bias_kde_lstm1,bias_kde_lstm2,bias_kde_lstm3,bias_kde_lstm4,bias_kde_lstm5]
bias_kde_lstm_all = np.concatenate(bias_kde_lstm_all)
bias_wb_lstm_all =[bias_wb_lstm1,bias_wb_lstm2,bias_wb_lstm3,bias_wb_lstm4,bias_wb_lstm5]
bias_wb_lstm_all = np.concatenate(bias_wb_lstm_all)

bias_all = np.append(bias_lstm_all,bias_kde_lstm_all)
bias_all = np.append(bias_all,bias_wb_lstm_all)
model_type = np.empty(bias_all.size,dtype=object)
model_type[:bias_lstm_all.size]='LSTM Model'
model_type[bias_lstm_all.size:2*(bias_lstm_all.size)]='PHY-LSTM Model'
model_type[2*(bias_lstm_all.size):]='PHYs-LSTM Model'
climates_type1 = np.empty(bias_lstm_all.size,dtype=object)
a = bias_lstm1.size
b = a + bias_lstm2.size
c = b + bias_lstm3.size
d = c + bias_lstm4.size
e = d + bias_lstm5.size
climates_type1[:a]='Equatorial'
climates_type1[a:b]='Arid'
climates_type1[b:c]='Warm temperate'
climates_type1[c:d]='Snow'
climates_type1[d:]='Polar'
climates_type2 = climates_type1
climates_type3 = climates_type1
climates_type_all = np.append(climates_type1,climates_type2)
climates_type_all = np.append(climates_type_all,climates_type3)
df1 = pd.DataFrame({"RMSE": pd.Series(bias_all), "climates values": pd.Series(climates_type_all), 'Model': pd.Series(model_type)})
#--------------------------------------------------------------------------------------------------------------------

bias_lstm = np.load(cfg['inputs_path']+cfg['product']+'/'+str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname1+"/focast_time " +str(cfg["forcast_time"])+"/r2_"+modelname1+".npy")
bias_lstm = lon_transform_2d(bias_lstm)
bias_kde_lstm = np.load(cfg['inputs_path']+cfg['product']+'/'+str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname2+"/focast_time " +str(cfg["forcast_time"])+"/r2_"+modelname2+".npy")
bias_kde_lstm = lon_transform_2d(bias_kde_lstm)
bias_wb_lstm = np.load(cfg['inputs_path']+cfg['product']+'/'+str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname3+"/focast_time " +str(cfg["forcast_time"])+"/r2_"+modelname3+".npy")
bias_wb_lstm = lon_transform_2d(bias_wb_lstm)

bias_lstm1 = bias_lstm[mask==1]
bias_lstm2 = bias_lstm[mask==2]
bias_lstm3 = bias_lstm[mask==3]
bias_lstm4 = bias_lstm[mask==4]
bias_lstm5 = bias_lstm[mask==5]
bias_kde_lstm1 = bias_kde_lstm[mask==1]
bias_kde_lstm2 = bias_kde_lstm[mask==2]
bias_kde_lstm3 = bias_kde_lstm[mask==3]
bias_kde_lstm4 = bias_kde_lstm[mask==4]
bias_kde_lstm5 = bias_kde_lstm[mask==5]
bias_wb_lstm1 = bias_wb_lstm[mask==1]
bias_wb_lstm2 = bias_wb_lstm[mask==2]
bias_wb_lstm3 = bias_wb_lstm[mask==3]
bias_wb_lstm4 = bias_wb_lstm[mask==4]
bias_wb_lstm5 = bias_wb_lstm[mask==5]
bias_lstm_all =[bias_lstm1,bias_lstm2,bias_lstm3,bias_lstm4,bias_lstm5]
bias_lstm_all = np.concatenate(bias_lstm_all)
bias_kde_lstm_all =[bias_kde_lstm1,bias_kde_lstm2,bias_kde_lstm3,bias_kde_lstm4,bias_kde_lstm5]
bias_kde_lstm_all = np.concatenate(bias_kde_lstm_all)
bias_wb_lstm_all =[bias_wb_lstm1,bias_wb_lstm2,bias_wb_lstm3,bias_wb_lstm4,bias_wb_lstm5]
bias_wb_lstm_all = np.concatenate(bias_wb_lstm_all)

bias_all = np.append(bias_lstm_all,bias_kde_lstm_all)
bias_all = np.append(bias_all,bias_wb_lstm_all)
model_type = np.empty(bias_all.size,dtype=object)
model_type[:bias_lstm_all.size]='LSTM Model'
model_type[bias_lstm_all.size:2*(bias_lstm_all.size)]='PHY-LSTM Model'
model_type[2*(bias_lstm_all.size):]='PHYs-LSTM Model'
climates_type1 = np.empty(bias_lstm_all.size,dtype=object)
a = bias_lstm1.size
b = a + bias_lstm2.size
c = b + bias_lstm3.size
d = c + bias_lstm4.size
e = d + bias_lstm5.size
climates_type1[:a]='Equatorial'
climates_type1[a:b]='Arid'
climates_type1[b:c]='Warm temperate'
climates_type1[c:d]='Snow'
climates_type1[d:]='Polar'
climates_type2 = climates_type1
climates_type3 = climates_type1
climates_type_all = np.append(climates_type1,climates_type2)
climates_type_all = np.append(climates_type_all,climates_type3)
df2 = pd.DataFrame({"R2": pd.Series(bias_all), "climates values": pd.Series(climates_type_all), 'Model': pd.Series(model_type)})
# ----------------------------------------------------
kge_lstm = np.load(cfg['inputs_path']+cfg['product']+'/'+str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname1+"/focast_time " +str(cfg["forcast_time"])+"/KGE_"+modelname1+".npy")
kge_lstm = lon_transform_2d(kge_lstm)
bias_kde_lstm = np.load(cfg['inputs_path']+cfg['product']+'/'+str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname2+"/focast_time " +str(cfg["forcast_time"])+"/KGE_"+modelname2+".npy")
bias_kde_lstm = lon_transform_2d(bias_kde_lstm)
bias_wb_lstm = np.load(cfg['inputs_path']+cfg['product']+'/'+str(cfg["spatial_resolution"])+'/' +str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname3+"/focast_time " +str(cfg["forcast_time"])+"/KGE_"+modelname3+".npy")
bias_wb_lstm = lon_transform_2d(bias_kde_lstm)

bias_lstm1 = bias_lstm[mask==1]
bias_lstm2 = bias_lstm[mask==2]
bias_lstm3 = bias_lstm[mask==3]
bias_lstm4 = bias_lstm[mask==4]
bias_lstm5 = bias_lstm[mask==5]
bias_kde_lstm1 = bias_kde_lstm[mask==1]
bias_kde_lstm2 = bias_kde_lstm[mask==2]
bias_kde_lstm3 = bias_kde_lstm[mask==3]
bias_kde_lstm4 = bias_kde_lstm[mask==4]
bias_kde_lstm5 = bias_kde_lstm[mask==5]
bias_wb_lstm1 = bias_wb_lstm[mask==1]
bias_wb_lstm2 = bias_wb_lstm[mask==2]
bias_wb_lstm3 = bias_wb_lstm[mask==3]
bias_wb_lstm4 = bias_wb_lstm[mask==4]
bias_wb_lstm5 = bias_wb_lstm[mask==5]

bias_lstm_all =[bias_lstm1,bias_lstm2,bias_lstm3,bias_lstm4,bias_lstm5]
bias_lstm_all = np.concatenate(bias_lstm_all)

bias_kde_lstm_all =[bias_kde_lstm1,bias_kde_lstm2,bias_kde_lstm3,bias_kde_lstm4,bias_kde_lstm5]
bias_kde_lstm_all = np.concatenate(bias_kde_lstm_all)

bias_wb_lstm_all =[bias_wb_lstm1,bias_wb_lstm2,bias_wb_lstm3,bias_wb_lstm4,bias_wb_lstm5]
bias_wb_lstm_all = np.concatenate(bias_wb_lstm_all)

bias_all = np.append(bias_lstm_all,bias_kde_lstm_all)
bias_all = np.append(bias_all,bias_wb_lstm_all)
model_type = np.empty(bias_all.size,dtype=object)
model_type[:bias_lstm_all.size]='LSTM Model'
model_type[bias_lstm_all.size:2*(bias_lstm_all.size)]='PHY-LSTM Model'
model_type[2*(bias_lstm_all.size):]='PHYs-LSTM Model'
climates_type1 = np.empty(bias_lstm_all.size,dtype=object)
a = bias_lstm1.size
b = a + bias_lstm2.size
c = b + bias_lstm3.size
d = c + bias_lstm4.size
e = d + bias_lstm5.size
climates_type1[:a]='Equatorial'
climates_type1[a:b]='Arid'
climates_type1[b:c]='Warm temperate'
climates_type1[c:d]='Snow'
climates_type1[d:]='Polar'
climates_type2 = climates_type1
climates_type3 = climates_type1
climates_type_all = np.append(climates_type1,climates_type2)
climates_type_all = np.append(climates_type_all,climates_type3)
df3 = pd.DataFrame({"KGE": pd.Series(bias_all), "climates values": pd.Series(climates_type_all), 'Model': pd.Series(model_type)})
#--------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------

print("-------------")
fig, axes = plt.subplots(2, 2, figsize=(11, 14))
# fig.subplots_adjust(hspace=0.13, wspace=0.13)

sns.set(font="SimHei")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.boxplot(x="climates values", y="R", data=df, hue='Model', showfliers=False, order=["Equatorial", "Arid","Warm temperate","Snow","Polar"],ax = axes[0,0]).set_title('(a)', loc='left',)
sns.boxplot(x="climates values", y="RMSE",data=df1,hue='Model',  showfliers=False, order=["Equatorial", "Arid","Warm temperate","Snow","Polar"],ax = axes[0,1]).set_title('(b)', loc='left')
sns.boxplot(x="climates values", y="R2", data=df2, hue='Model', showfliers=False, order=["Equatorial", "Arid","Warm temperate","Snow","Polar"],ax = axes[1,0]).set_title('(c)', loc='left')
sns.boxplot(x="climates values", y="KGE", data=df3, hue='Model', showfliers=False, order=["Equatorial", "Arid","Warm temperate","Snow","Polar"],ax = axes[1,1]).set_title('(d)', loc='left',)
# sns.boxplot(x="climates values", y="RMSE", data=df4, hue='Model', showfliers=False, order=["Equatorial", "Arid","Warm temperate","Snow","Polar"],ax = axes[1,1]).set_title('(d)', loc='left')
# sns.boxplot(x="climates values", y="R", data=df5, hue='Model', showfliers=False, order=["Equatorial", "Arid","Warm temperate","Snow","Polar"],ax = axes[2,1]).set_title('(f)', loc='left')
# sns.tight_layout()
plt.tight_layout()
plt.show()