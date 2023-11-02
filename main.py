import os
import numpy as np
from train import train
from eval import test
from data import Dataset
from config import get_args
import torch
# ------------------------------------------------------------------------------ 
# Original author : Qingliang Li,Sen Yan, Cheng Zhang, 1/23/2023
# ------------------------------------------------------------------------------
def main(cfg):
    # 设置PyTorch的随机数种子
    seed = cfg['seed']  # 你可以选择任何你喜欢的种子值
    torch.manual_seed(seed)
    #判断是否有显卡，有显卡用显卡，没显卡用cpu
    device = torch.device(cfg['device']) if torch.cuda.is_available() else torch.device('cpu')

    device_ids = [0,1]
    print('Now we training {d_p} product in {sr} spatial resolution'.format(d_p=cfg['product'],sr=str(cfg['spatial_resolution'])))
    # ------------------------------------------------------------------------------------------------------------------------------
    # x_train: nt,nf,nlat,nlon; y_train:nt,nlat,nlon, static:nlat,nlon
    print('1 step:-----------------------------------------------------------------------------------------------------------------')
    print('[ATAI {d_p} work ] Make & load inputs'.format(d_p=cfg['workname']))
    #创建数据存放的文件夹  spatial_resolution默认为1
    path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'+str(cfg['selected_year'][0])+'/'
    if not os.path.isdir (path):
        os.makedirs(path)
    out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' +str(cfg['selected_year'][0])+'/'+cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
    if not os.path.isdir (out_path):
        os.makedirs(out_path)
    #判断是否有训练数据文件x_train_norm.npy，有则使用 无则创建
    if os.path.exists(path+'x_train_norm.npy'):
        print(' [ATAI {d_p} work ] loading input data'.format(d_p=cfg['workname']))
        x_train_shape = np.load(path+'x_train_norm_shape.npy',mmap_mode='r')
        x_train = np.memmap(path+'x_train_norm.npy',dtype=cfg['data_type'],mode='r+',shape=(x_train_shape[0],x_train_shape[1], x_train_shape[2], x_train_shape[3]))
        x_test_shape = np.load(path+'x_test_norm_shape.npy',mmap_mode='r')
        x_test = np.memmap(path+'x_test_norm.npy',dtype=cfg['data_type'],mode='r+',shape=(x_test_shape[0],x_test_shape[1], x_test_shape[2], x_test_shape[3]))
        y_train = np.load(path+'y_train_norm.npy',mmap_mode='r')
        y_test = np.load(path+'y_test_norm.npy',mmap_mode='r')
        static = np.load(path+'static_norm.npy')
        file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
        mask = np.load(path+file_name_mask)
        dww_train = np.load(path + '/' + 'dww_train.npy', mmap_mode='r')
        # 使用不同的变量组合成地表水平衡
        if 'LandBench1' in cfg['workname']:
            dw_train = dww_train[0, :, :, :]
        elif 'LandBench2' in cfg['workname']:
            dw_train = dww_train[1, :, :, :]
        elif 'LandBench3' in cfg['workname']:
            dw_train = dww_train[2, :, :, :]
        elif 'LandBench4' in cfg['workname']:
            dw_train = dww_train[3, :, :, :]
        elif 'LandBench5' in cfg['workname']:
            dw_train = dww_train[4, :, :, :]
        elif 'LandBench6' in cfg['workname']:
            dw_train = dww_train[5, :, :, :]
        elif 'LandBench7' in cfg['workname']:
            dw_train = dww_train[6, :, :, :]
        elif 'LandBench8' in cfg['workname']:
            dw_train = dww_train[7, :, :, :]
    else:             # 在data.py中重新处理并且加载数据
        print('[ATAI {d_p} work ] making input data'.format(d_p=cfg['workname']))
        cls = Dataset(cfg) #FIXME: saving to input path
        x_train, y_train, x_test, y_test, static, lat, lon,mask,dw_train = cls.fit(cfg)
    # load scaler for inverse
    # 保存归一化所使用的最大最小值
    if cfg['normalize_type'] in ['region']:                                      
        scaler_x = np.memmap(path+'scaler_x.npy',dtype=cfg['data_type'],mode='r+',shape=(2, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
        scaler_y = np.memmap(path+'scaler_y.npy',dtype=cfg['data_type'],mode='r+',shape=(2, y_train.shape[1], y_train.shape[2], y_train.shape[3]))  
    elif cfg['normalize_type'] in ['global']:    
        scaler_x = np.memmap(path+'scaler_x.npy',dtype=cfg['data_type'],mode='r+',shape=(2, x_train.shape[3]))
        scaler_y = np.memmap(path+'scaler_y.npy',dtype=cfg['data_type'],mode='r+',shape=(2, y_train.shape[3]))  
    # ------------------------------------------------------------------------------------------------------------------------------
    mask2 = np.ones(mask.shape)
    for i in range(scaler_x.shape[-1]):
        mask2[scaler_x[0,:,:,i]==scaler_x[1,:,:,i]] = 0
    for i in range(scaler_y.shape[-1]):
        mask2[scaler_y[0, :, :, i] == scaler_y[1, :, :, i]] = 0
    mask = mask * mask2
    file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
    np.save(path + file_name_mask, np.squeeze(mask))
    mask = np.load(path + file_name_mask)
    # ------------------------------------------------------------------------------------------------------------------------------
    print('2 step:-----------------------------------------------------------------------------------------------------------------')
    print('[ATAI {d_p} work ] Train & load {m_n} Model'.format(d_p=cfg['workname'],m_n=cfg['modelname']))
    print('[ATAI {d_p} work ] Wandb info'.format(d_p=cfg['workname']))
# ------------------------------------------------------------------------------------------------------------------------------
    #加载训练好的模型，如果没有则在train.py中进行训练
    if os.path.exists(out_path+cfg['modelname'] +'_para.pkl'):
        print('[ATAI {d_p} work ] loading trained model'.format(d_p=cfg['workname'])) 
        model = torch.load(out_path+cfg['modelname']+'_para.pkl')
    else:
        # train 
        print('[ATAI {d_p} work ] training {m_n} model'.format(d_p=cfg['workname'],m_n=cfg['modelname'])) 
        for j in range(cfg["num_repeat"]):
            train(x_train, y_train, static, mask, scaler_x, scaler_y, cfg, j,path,out_path,device,device_ids,dw_train,dw_train)
	    #加载训练好的模型
            model = torch.load(out_path+cfg['modelname']+'_para.pkl')
        print('[ATAI {d_p} work ] finish training {m_n} model'.format(d_p=cfg['workname'],m_n=cfg['modelname']))   
    # ------------------------------------------------------------------------------------------------------------------------------
    print('3 step:-----------------------------------------------------------------------------------------------------------------')  
    print('[ATAI {d_p} work ] Make predictions by {m_n} Model'.format(d_p=cfg['workname'],m_n=cfg['modelname']))  
# ------------------------------------------------------------------------------------------------------------------------------
    print('x_test shape :',x_test.shape)
    print('y_test shape :',y_test.shape)
    print('static shape :',static.shape)    
    print('scaler_x shape is',scaler_x.shape)
    print('scaler_y shape is',scaler_y.shape)
    #对加载的模型在eval.py中进行测试
    y_pred, y_test = test(x_test, y_test, static, scaler_y, cfg, model,device)
# ------------------------------------------------------------------------------------------------------------------------------   
# 将测试集跑出来的结果保存
    print('[ATAI {d_p} work ] Saving predictions by {m_n} Model and we hope to use "postprocess" and "evaluate" codes for detailed analyzing'.format(d_p=cfg['workname'],m_n=cfg['modelname']))
    np.save(out_path +'_predictions.npy', y_pred)
    np.save(out_path + 'observations.npy', y_test)


if __name__ == '__main__':
    cfg = get_args()
    main(cfg)
