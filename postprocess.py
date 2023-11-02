import numpy as np
from utils import unbiased_rmse,_rmse,_bias
from config import get_args
# Original author : Qingliang Li,Sen Yan, Cheng Zhang, 1/23/2023
def lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:,:int(x.shape[2]/2)] = x[:,:,int(x.shape[2]/2):] 
  x_new[:,:,int(x.shape[2]/2):] = x[:,:,:int(x.shape[2]/2)] 
  return x_new
def postprocess(cfg):
    PATH = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'+str(cfg['selected_year'][0])+'/'
    file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
    mask = np.load(PATH+file_name_mask)
    if cfg['modelname'] in ['ConvLSTM','PHYConvLSTM']:
        out_path_convlstm = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'+str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
        y_pred_convlstm = np.load(out_path_convlstm+'_predictions.npy')
        y_pred_convlstm = y_pred_convlstm / 7 * 289 / 2.89
        y_test_convlstm = np.load(out_path_convlstm+'observations.npy')
        y_test_convlstm = y_test_convlstm / 7 * 289 / 2.89
        print(y_pred_convlstm.shape, y_test_convlstm.shape)
        # get shape
        nt, nlat, nlon = y_test_convlstm.shape    
        # mask 
        #mask=y_test_lstm==y_test_convlstm
        # cal perf
        r2_convlstm = np.full(( nlat, nlon), np.nan)
        urmse_convlstm = np.full(( nlat, nlon), np.nan)
        r_convlstm = np.full(( nlat, nlon), np.nan)
        rmse_convlstm = np.full(( nlat, nlon), np.nan)
        bias_convlstm = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_convlstm[:, i, j]).any()):
                    urmse_convlstm[i, j] = unbiased_rmse(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])
                    #r2_convlstm[i, j] = r2_score(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])
                    r_convlstm[i, j] = np.corrcoef(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])[0,1]
                    rmse_convlstm[i, j] = _rmse(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])
                    bias_convlstm[i, j] = _bias(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])
        np.save(out_path_convlstm + 'r2_'+cfg['modelname']+'.npy', r2_convlstm)
        np.save(out_path_convlstm + 'r_'+cfg['modelname']+'.npy', r_convlstm)
        np.save(out_path_convlstm + 'rmse_'+cfg['modelname']+'.npy', rmse_convlstm)
        np.save(out_path_convlstm + 'bias_'+cfg['modelname']+'.npy', bias_convlstm)
        np.save(out_path_convlstm + 'urmse_'+cfg['modelname']+'.npy', urmse_convlstm)
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if  cfg['modelname'] in ['LSTM','PHYLSTM','WBLSTM']:
        out_path_lstm = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
        y_pred_lstm = np.load(out_path_lstm+'_predictions.npy')
        y_pred_lstm = y_pred_lstm / 7 * 100 / 1
        y_test_lstm = np.load(out_path_lstm+'observations.npy')
        y_test_lstm = y_test_lstm / 7 * 100 / 1
        print(y_pred_lstm.shape, y_test_lstm.shape)
        # get shape
        nt, nlat, nlon = y_test_lstm.shape 
        # mask
        #mask=y_test_lstm==y_test_lstm
        # cal perf
        r2_lstm = np.full(( nlat, nlon), np.nan)
        urmse_lstm = np.full(( nlat, nlon), np.nan)
        r_lstm = np.full(( nlat, nlon), np.nan)
        rmse_lstm = np.full(( nlat, nlon), np.nan)
        bias_lstm = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_lstm[:, i, j]).any()):
                    #print(' y_pred_lstm[:, i, j] is', y_pred_lstm[:, i, j])
                    #print(' y_test_lstm[:, i, j] is', y_test_lstm[:, i, j])
                    urmse_lstm[i, j] = unbiased_rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    #r2_lstm[i, j] = r2_score(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    #print(' r2_lstm[i, j] is', r2_lstm[i, j])
                    r_lstm[i, j] = np.corrcoef(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])[0,1]
                    rmse_lstm[i, j] = _rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    bias_lstm[i, j] = _bias(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
        np.save(out_path_lstm + 'r2_'+cfg['modelname']+'.npy', r2_lstm)
        np.save(out_path_lstm + 'r_'+cfg['modelname']+'.npy', r_lstm)
        np.save(out_path_lstm + 'rmse_'+cfg['modelname']+'.npy', rmse_lstm)
        np.save(out_path_lstm + 'bias_'+cfg['modelname']+'.npy', bias_lstm)
        np.save(out_path_lstm + 'urmse_'+cfg['modelname']+'.npy', urmse_lstm)
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['CNN','PHYCNN']:
        out_path_cnn = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
        y_pred_cnn = np.load(out_path_cnn+'_predictions.npy')
        y_test_cnn = np.load(out_path_cnn+'observations.npy')
        y_pred_cnn = y_pred_cnn[cfg["seq_len"]:]
        y_test_cnn = y_test_cnn[cfg["seq_len"]:]
        print(y_pred_cnn.shape, y_test_cnn.shape)
        # get shape
        nt, nlat, nlon = y_test_cnn.shape 
        # mask
        #mask=y_test_cnn==y_test_cnn
        # cal perf
        r2_cnn = np.full(( nlat, nlon), np.nan)    
        urmse_cnn = np.full(( nlat, nlon), np.nan)
        r_cnn = np.full(( nlat, nlon), np.nan)
        rmse_cnn = np.full(( nlat, nlon), np.nan)
        bias_cnn = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_cnn[:, i, j]).any()):
                    urmse_cnn[i, j] = unbiased_rmse(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    #r2_cnn[i, j] = r2_score(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    r_cnn[i, j] = np.corrcoef(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])[0,1]
                    rmse_cnn[i, j] = _rmse(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    bias_cnn[i, j] = _bias(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
        np.save(out_path_cnn + 'r2_'+cfg['modelname']+'.npy', r2_cnn)
        np.save(out_path_cnn + 'r_'+cfg['modelname']+'.npy', r_cnn)
        np.save(out_path_cnn + 'rmse_'+cfg['modelname']+'.npy', rmse_cnn)
        np.save(out_path_cnn + 'bias_'+cfg['modelname']+'.npy', bias_cnn)
        np.save(out_path_cnn + 'urmse_'+cfg['modelname']+'.npy', urmse_cnn)
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = get_args()
    postprocess(cfg)






               


