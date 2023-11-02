import os
import numpy as np
from config import get_args
def nearest_num(num, seq_data):
    """
    modified by Gaosong Shi
    :param num: 某经度 或 纬度的具体值
    :param seq_data: 具体值查找的范围列表
    ：return: index
    """
    near_i = float(np.inf)
    index = 0
    for i in range(len(seq_data)):
        temp_data = abs(num - seq_data[i])
        if temp_data < near_i:
            near_i = temp_data
            index = i
    return index


def resize(data,lon_new,lat_new,lon,lat):
    """
    ### 按照经纬度最相近的位置进行赋值
    """
    ###  根据最近的 经度 与 纬度 ，寻找预处理数据与标准数据间
    data_index_lon = np.array([nearest_num(i,lon) for i in lon_new])
    data_index_lat = np.array([nearest_num(i,lat) for i in lat_new])
    # ————————————————————————————————————————————————————————————————————————————————————————————————————
    ###根据最近的经纬度，调整分辨率后的协变量
    data_new = data[data_index_lat,:][:,data_index_lon]
    return data_new
def lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:,:int(x.shape[2]/2)] = x[:,:,int(x.shape[2]/2):]
  x_new[:,:,int(x.shape[2]/2):] = x[:,:,:int(x.shape[2]/2)]
  return x_new
def main():
    print()
    PATH = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg["spatial_resolution"]) + '/'+str(cfg['selected_year'][0])+'/'
    lon = np.load(PATH+"lon_climates.npy")
    lat = np.load(PATH+"lat_climates.npy")
    lon_new = np.load(PATH+'lon_{s}.npy'.format(s=cfg['spatial_resolution']))
    lat_new = np.load(PATH+'lat_{s}.npy'.format(s=cfg['spatial_resolution']))
    out_path_lstm = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg["spatial_resolution"]) + '/'+str(cfg['selected_year'][0])+ '/' + cfg[
        'workname'] + "/LSTM/focast_time " + str(cfg["forcast_time"]) + "/"
    y_pred = np.load(out_path_lstm + '_predictions.npy')
    y_pred = lon_transform(y_pred)
    lon_new = np.linspace(-180,179,int(y_pred.shape[2]))
    climates_mask = np.load(PATH+"climates_mask.npy")

    climates_mask_2 = resize(climates_mask,lon_new,lat_new,lon,lat)
    climates_mask_2 = np.flipud(climates_mask_2)
    # np.savetxt("climates_mask_2.csv", climates_mask_2, delimiter=",")
    np.save(PATH+'climates_mask_{s}.npy'.format(s=cfg['spatial_resolution']),climates_mask_2)


if __name__ == '__main__':
    cfg = get_args()
    main()
