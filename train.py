import time
import numpy as np
import torch
import torch.nn
from tqdm import trange
from data_gen import load_test_data_for_rnn,load_train_data_for_rnn,load_train_data_for_phylstm,load_test_data_for_cnn, load_train_data_for_cnn,load_train_data_for_phycnn,erath_data_transform,sea_mask_rnn,sea_mask_cnn
from loss import  NaNMSELoss,PHYLoss,WBLoss
from model import LSTMModel,CNN,ConvLSTMModel
# Original author : Qingliang Li,Sen Yan, Cheng Zhang, 1/23/2023
def train(x,
          y,
          static,
          mask, 
          scaler_x,
          scaler_y,
          cfg,
          num_repeat,
          PATH,
          out_path,
          device, 
          device_ids,
          dw_train,
          num_task=None,
          valid_split=True):

    patience = cfg['patience']
    wait = 0
    best = 9999
    valid_split = cfg['valid_split']
    print(dw_train.shape)
    print(y.shape)
    print('the device is {d}'.format(d=device))
    #dw_train = np.reshape(y.shape)
    if cfg['modelname'] in ['CNN', 'ConvLSTM','PHYCNN','PHYConvLSTM']:
#	Splice x according to the sphere shape
        lat_index,lon_index = erath_data_transform(cfg, x)
        print('\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg['modelname']))
    if valid_split:
        nt,nf,nlat,nlon = x.shape  #x shape :nt,nf,nlat,nlon
	#划分验证集和训练集
        N = int(nt*cfg['split_ratio'])
        x_valid, y_valid,dw_valid,wb_valid, static_valid = x[N:], y[N:],dw_train[N:],dw_train[N:], static
        x, y ,dw_train,wb_train= x[:N], y[:N], dw_train[:N],dw_train[N:]
        #x_valid, y_valid,dw_valid,wb_valid, static_valid = x[-60:-30], y[-60:-30],dw_train[-60:-30],wb_train[-60:-30], static
        #x, y ,dw_train,wb_train= x[-30:], y[-30:], dw_train[-30:],wb_train[-30:]

    # 损失函数
    lossmse = torch.nn.MSELoss()
    loss_relu = torch.nn.ReLU()
#	filter Antatctica
    print('x_train shape is', x.shape)
    print('y_train shape is', y.shape)
    print('dw_train shape is', dw_train.shape)
    print('static_train shape is', static.shape)
    print('mask shape is', mask.shape)

    # mask see regions
    # 确定陆地边界
    if cfg['modelname'] in ['LSTM','PHYLSTM','WBLSTM']:
        if valid_split:
            x_valid, y_valid, dw_valid,wb_valid, static_valid = sea_mask_rnn(cfg, x_valid, y_valid,dw_valid,wb_valid, static_valid, mask )
        x, y,dw_train,wb_train, static = sea_mask_rnn(cfg, x, y,dw_train,wb_train, static, mask)
    elif cfg['modelname'] in ['CNN', 'ConvLSTM','PHYCNN','PHYConvLSTM']:
        if valid_split:
            x_valid, y_valid, dw_valid, static_valid, mask_index  = sea_mask_cnn(cfg, x_valid, y_valid,dw_valid, static_valid, mask)
        x, y, dw_train, static, mask_index = sea_mask_cnn(cfg, x, y,dw_train, static, mask )

    # train and validate
    # NOTE: We preprare two callbacks for training:
    #       early stopping and save best model.
    for num_ in range(cfg['num_repeat']):
        # prepare models
	#选择模型
        if cfg['modelname'] in ['LSTM','PHYLSTM','WBLSTM']:
            lstmmodel_cfg = {}
            lstmmodel_cfg['input_size'] = cfg["input_size"]
            lstmmodel_cfg['hidden_size'] = cfg["hidden_size"]*1
            lstmmodel_cfg['out_size'] = 1
            model = LSTMModel(cfg,lstmmodel_cfg).to(device)
            # model = torch.nn.DataParallel(model, device_ids=device_ids)
        elif cfg['modelname'] in ['CNN', 'PHYCNN']:
            model = CNN(cfg).to(device)
        elif cfg['modelname'] in ['ConvLSTM', 'PHYConvLSTM']:
            model = ConvLSTMModel(cfg).to(device)

      #  model.train()
	 # Prepare for training
    # NOTE: Only use `Adam`, we didn't apply adaptively
    #       learing rate schedule. We found `Adam` perform
    #       much better than `Adagrad`, `Adadelta`.
	#算法选择Adam
        optim = torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])

        with trange(1, cfg['epochs']+1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg['modelname']+' '+str(num_repeat))
                t_begin = time.time()
                # train
                MSELoss = 0
                for iter in range(0, cfg["niter"]):
 # ------------------------------------------------------------------------------------------------------------------------------
 #  train way for LSTM model
                    if cfg["modelname"] in \
                            ['LSTM']:
                        # generate batch data for Recurrent Neural Network
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_rnn(cfg, x, y, static, scaler_y)
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1)
                        x_batch = torch.cat([x_batch, aux_batch], 2)
                        pred = model(x_batch, aux_batch)
                        pred = torch.squeeze(pred,1)
#  train way for PHYLSTM model
                    elif cfg["modelname"] in \
                            ['PHYLSTM']:
                        # generate batch data for Recurrent Neural Network
                        x_batch, y_batch, aux_batch, _, _, dw, oldsm = \
                            load_train_data_for_phylstm(cfg, x, y, static, scaler_y,dw_train)
                        oldsm = torch.from_numpy(oldsm).to(device)
                        dw = torch.from_numpy(dw).to(device)
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1)
                        x_batch = torch.cat([x_batch, aux_batch], 2)
                        pred = model(x_batch, aux_batch)
                        pred = torch.squeeze(pred,1)
#  train way for WBLSTM model
                    elif cfg["modelname"] in \
                            ['WBLSTM']:
                        # generate batch data for Recurrent Neural Network
                        x_batch, y_batch, aux_batch, _, _, wb, oldsm = \
                            load_train_data_for_phylstm(cfg, x, y, static, scaler_y,wb_train)
                        oldsm = torch.from_numpy(oldsm).to(device)
                        wb = torch.from_numpy(wb).to(device)
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1)
                        x_batch = torch.cat([x_batch, aux_batch], 2)
                        pred = model(x_batch, aux_batch)
                        pred = torch.squeeze(pred,1)
 #  train way for CNN model
                    elif cfg['modelname'] in ['CNN']:
                        # generate batch data for Convolutional Neural Network
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_cnn(cfg, x, y, static, scaler_y,lat_index,lon_index,mask_index)
                        x_batch[np.isnan(x_batch)] = 0  # filter nan values to train cnn model
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        # x_batch = torch.cat([x_batch, aux_batch], 2)
                        x_batch = x_batch.squeeze(dim=1)
                        x_batch = x_batch.reshape(x_batch.shape[0],x_batch.shape[1]*x_batch.shape[2],x_batch.shape[3],x_batch.shape[4])
                        #print('x_batch shape is',x_batch.shape)
                        #print('aux_batch shape is',aux_batch.shape)
                        x_batch = torch.cat([x_batch, aux_batch], 1)
                        pred = model(x_batch, aux_batch)
                    elif cfg['modelname'] in ['PHYCNN']:
                        # generate batch data for Convolutional Neural Network
                        x_batch, y_batch, aux_batch, _, _, dw, oldsm = \
                            load_train_data_for_phycnn(cfg, x, y, static, scaler_y,lat_index,lon_index,mask_index,dw_train)
                        x_batch[np.isnan(x_batch)] = 0  # filter nan values to train cnn model
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        dw = torch.from_numpy(dw).to(device)
                        oldsm = torch.from_numpy(oldsm).to(device)
                        x_batch = x_batch.squeeze(dim=1)
                        x_batch = x_batch.reshape(x_batch.shape[0],x_batch.shape[1]*x_batch.shape[2],x_batch.shape[3],x_batch.shape[4])
                        #print('x_batch shape is',x_batch.shape)
                        #print('aux_batch shape is',aux_batch.shape)
                        x_batch = torch.cat([x_batch, aux_batch], 1)
                        pred = model(x_batch, aux_batch)
                    elif cfg['modelname'] in ['ConvLSTM']:
                        # generate batch data for Convolutional LSTM
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_cnn(cfg, x, y, static, scaler_y,lat_index,lon_index,mask_index) # same as Convolutional Neural Network
                        x_batch[np.isnan(x_batch)] = 0  # filter nan values to train cnn model
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1,1,1)
                        x_batch = x_batch.squeeze(dim=1)
                        pred = model(x_batch, aux_batch,cfg)
                    elif cfg['modelname'] in ['PHYConvLSTM']:
                        # generate batch data for Convolutional LSTM
                        x_batch, y_batch, aux_batch, _, _, dw, oldsm = \
                            load_train_data_for_phycnn(cfg, x, y, static, scaler_y,lat_index,lon_index,mask_index,dw_train) # same as Convolutional Neural Network
                        x_batch[np.isnan(x_batch)] = 0  # filter nan values to train cnn model
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        dw = torch.from_numpy(dw).to(device)
                        oldsm = torch.from_numpy(oldsm).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1,1,1)
                        x_batch = x_batch.squeeze(dim=1)
                        pred = model(x_batch, aux_batch,cfg)
 # ------------------------------------------------------------------------------------------------------------------------------
                    if cfg['modelname'] in ['PHYLSTM', 'PHYCNN', 'PHYConvLSTM']:
                        loss = PHYLoss.fit(cfg, pred.float(), y_batch.float(),oldsm.float(),dw.float(), lossmse,loss_relu)
                        #weight = F.softmax(torch.randn(2),dim=-1)
                        #loss = torch.sum(loss * weight)
                    elif cfg['modelname'] in ['LSTM', 'CNN', 'ConvLSTM']:
                        loss = NaNMSELoss.fit(cfg, pred.float(), y_batch.float(), lossmse)
                    elif cfg['modelname'] in ['WBLSTM']:
                        loss = WBLoss.fit(cfg, pred.float(), y_batch.float(),oldsm.float(),wb.float(), lossmse,loss_relu)
                        # weight = F.softmax(torch.randn(2),dim=-1)
                        # loss = torch.sum(loss * weight)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()                    
                    MSELoss += loss.item()
# ------------------------------------------------------------------------------------------------------------------------------
                t_end = time.time()
                # get loss log
                loss_str = "Epoch {} Train MSE Loss {:.3f} time {:.2f}".format(epoch, MSELoss / cfg["niter"], t_end - t_begin)
                print(loss_str)
                # refresh train if loss equal to NaN. Will build fresh model and 
                # re-train it until it didn't have NaN loss.
                #if np.isnan(MSELoss):
                #    break

                # validate
		#使用验证集对已经训练的模型进行测试
		#每20轮训练用验证集进行一次验证，如果得到的误差比最小误差小 那么就对模型进行保存
                if valid_split:
                    del x_batch, y_batch, aux_batch
                    MSE_valid_loss = 0
                    if epoch % 20 == 0:
                        wait += 1
                        # NOTE: We used grids-mean NSE as valid metrics.
                        t_begin = time.time()
# ------------------------------------------------------------------------------------------------------------------------------
 #  validate way for LSTM model
                        if cfg["modelname"] in ['LSTM','PHYLSTM','WBLSTM']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len'],cfg["stride"])]
                            n = (x_valid.shape[0]-cfg["seq_len"])//cfg["stride"]
                            for i in range(0, n):
                                #mask
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_rnn(cfg, x_valid, y_valid, static_valid, scaler_y,cfg["stride"], i, n)
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                aux_valid_batch = aux_valid_batch.unsqueeze(1)
                                aux_valid_batch = aux_valid_batch.repeat(1,x_valid_batch.shape[1],1)
                                x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2)
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid.squeeze(1), y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()
#  validate way for CNN model
                        elif cfg['modelname'] in ['CNN','PHYCNN']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len']-cfg['forcast_time'],cfg["stride"])]
                            valid_batch_size = cfg["batch_size"]*10
                            for i in gt_list:
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_cnn(cfg, x_valid, y_valid, static, scaler_y,gt_list,lat_index,lon_index, i ,cfg["stride"]) # same as Convolutional Neural Network

                                x_valid_batch[np.isnan(x_valid_batch)] = 0
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                # x_valid_temp = torch.cat([x_valid_temp, static_valid_temp], 2)
                                x_valid_batch = x_valid_batch.squeeze(1)
                                x_valid_batch = x_valid_batch.reshape(x_valid_batch.shape[0],x_valid_batch.shape[1]*x_valid_batch.shape[2],x_valid_batch.shape[3],x_valid_batch.shape[4])
                                x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], axis=1)
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid, y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()
#  validate way for ConvLSTM model，same as CNN model
                        elif cfg['modelname'] in ['ConvLSTM','PHYConvLSTM']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len']-cfg['forcast_time'],cfg["stride"])]
                            valid_batch_size = cfg["batch_size"]*10
                            for i in gt_list:
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_cnn(cfg, x_valid, y_valid, static, scaler_y,gt_list,lat_index,lon_index, i ,cfg["stride"]) # same as Convolutional Neural Network

                                x_valid_batch[np.isnan(x_valid_batch)] = 0
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                aux_valid_batch = aux_valid_batch.unsqueeze(1)
                                aux_valid_batch = aux_valid_batch.repeat(1,x_valid_batch.shape[1],1,1,1)
                                # x_valid_temp = torch.cat([x_valid_temp, static_valid_temp], 2)
                                x_valid_batch = x_valid_batch
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch,cfg)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid, y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()
# ------------------------------------------------------------------------------------------------------------------------------                        

                        t_end = time.time()
                        mse_valid_loss = MSE_valid_loss/(len(gt_list))
                        # get loss log
                        loss_str = '\033[1;31m%s\033[0m' % \
                                "Epoch {} Val MSE Loss {:.3f}  time {:.2f}".format(epoch,mse_valid_loss, 
                                    t_end-t_begin)
                        print(loss_str)
                        val_save_acc = mse_valid_loss

                        # save best model by val loss
                        # NOTE: save best MSE results get `single_task` better than `multi_tasks`
                        #       save best NSE results get `multi_tasks` better than `single_task`
                        if val_save_acc < best:
                        #if MSE_valid_loss < best:
                            torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                            wait = 0  # release wait
                            best = val_save_acc #MSE_valid_loss
                            print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                else:
                    # save best model by train loss
                    if MSELoss < best:
                        best = MSELoss
                        wait = 0
                        torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                        print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')

                # early stopping
                if wait >= patience:
                    return
            return




