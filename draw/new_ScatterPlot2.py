from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import os
from statistics import mean
import matplotlib.colors as colors

from config import get_args


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b
def density_calc(x, y, radius):
    """
    散点密度计算（以便给散点图中的散点密度进行颜色渲染）
    :param x:
    :param y:
    :param radius:
    :return:  数据密度
    """
    res = np.empty(len(x), dtype=np.float32)
    for i in range(len(x)):
        res[i] = np.sum((x > (x[i] - radius)) & (x < (x[i] + radius))
                        & (y > (y[i] - radius)) & (y < (y[i] + radius)))
    return res
def main(cfg):
    modelname1 = 'LSTM'
    modelname2 = 'PHYLSTM'
    modelname3 = 'PHYsLSTM'
    file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
    mask = np.load(cfg['inputs_path']+cfg['product']+'/' + str(cfg["spatial_resolution"]) +'/' +str(cfg['selected_year'][0])+'/'+ file_name_mask)
    out_path = cfg['inputs_path']+cfg['product']+'/' + str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname1+"/focast_time " + str(0) + "/"
    lstm_predicted = np.load(out_path + '_predictions.npy')
    observed = np.load(out_path + 'observations.npy')
    out_path = cfg['inputs_path']+cfg['product']+'/'+ str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname2+"/focast_time " + str(0) + "/"
    kde_lstm_predicted = np.load(out_path + '_predictions.npy')
    out_path = cfg['inputs_path']+cfg['product']+'/'+ str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname3+"/focast_time " + str(0) + "/"
    wb_lstm_predicted = np.load(out_path + '_predictions.npy')
    observed_sm = np.squeeze(observed)
    observed_sm = observed_sm[-365:, :, :]
    observed_sm[:, mask == 0] = -0

    predicted_lstm = np.squeeze(lstm_predicted)
    predicted_lstm = predicted_lstm[-365:, :, :]
    predicted_lstm[:, mask == 0] = -0

    predicted_kde = np.squeeze(kde_lstm_predicted)
    predicted_kde = predicted_kde[-365:, :, :]
    predicted_kde[:, mask == 0] = -0

    predicted_wb = np.squeeze(wb_lstm_predicted)
    predicted_wb = predicted_wb[-365:, :, :]
    predicted_wb[:, mask == 0] = -0

    fig, axes = plt.subplots(1, 2, figsize=(16, 13))
    ax = axes.flatten()

    # 计算散点图
    sizes = np.pi * 2 ** 1
    observed_sm_mean1 = np.average(observed_sm, axis=0)
    predicted_sm_mean1 = np.average(predicted_lstm, axis=0)
    ############################################
    sm_index = observed_sm_mean1 != 0
    observed_sm_mean1 = observed_sm_mean1[sm_index]
    predicted_sm_mean1 = predicted_sm_mean1[sm_index]
    m, b = best_fit_slope_and_intercept(observed_sm_mean1, predicted_sm_mean1)
    regression_line = []
    for a in observed_sm_mean1:
        regression_line.append((m * a) + b)
    radius = 0.1  # 半径

    # axes.ylabel('111111111111111111111111')

    colormap = plt.get_cmap("Blues")
    colormap1 = plt.get_cmap("Reds")
    colormap2 = plt.get_cmap("Greens")

    # 计算散点图
    sizes = np.pi * 2 ** 1
    observed_sm_mean2 = np.average(observed_sm, axis=0)
    predicted_sm_mean2 = np.average(predicted_kde, axis=0)
    ############################################
    sm_index = observed_sm_mean2 != 0
    observed_sm_mean2 = observed_sm_mean2[sm_index]
    predicted_sm_mean2 = predicted_sm_mean2[sm_index]
    m, b = best_fit_slope_and_intercept(observed_sm_mean2, predicted_sm_mean2)
    regression_line = []
    for a in observed_sm_mean2:
        regression_line.append((m * a) + b)
    radius = 0.1  # 半径
    observed_sm_mean3 = np.average(observed_sm, axis=0)
    predicted_sm_mean3 = np.average(predicted_wb, axis=0)
    ############################################
    sm_index = observed_sm_mean3 != 0
    observed_sm_mean3 = observed_sm_mean3[sm_index]
    predicted_sm_mean3 = predicted_sm_mean3[sm_index]
    m, b = best_fit_slope_and_intercept(observed_sm_mean3, predicted_sm_mean3)
    regression_line = []
    for a in observed_sm_mean3:
        regression_line.append((m * a) + b)
    radius = 0.1  # 半径
    Z1 = density_calc(observed_sm_mean1, predicted_sm_mean1, radius)
    Z2 = density_calc(observed_sm_mean2, predicted_sm_mean2, radius)
    Z3 = density_calc(observed_sm_mean3, predicted_sm_mean3, radius)
    ax[0].scatter(observed_sm_mean3, predicted_sm_mean3, c=Z3, s=sizes, cmap=colormap2,
                  norm=colors.LogNorm(vmin=Z3.min(), vmax=Z3.max()))
    ax[0].plot(observed_sm_mean3, regression_line, 'red', lw=0.8)  #####回归线

    ax[0].scatter(observed_sm_mean1, predicted_sm_mean1, c=Z1, s=sizes, cmap=colormap,
                  norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))
    ax[0].plot(observed_sm_mean1, regression_line, 'red', lw=0.8)  #####回归线

    ax[0].scatter(observed_sm_mean2, predicted_sm_mean2, c=Z2, s=sizes, cmap=colormap1,
                  norm=colors.LogNorm(vmin=Z2.min(), vmax=Z2.max()))
    ax[0].plot(observed_sm_mean1, regression_line, 'red', lw=0.8)  #####回归线

    ax[0].plot([0.05, 0.7], [0.05, 0.7], 'black', lw=0.8)  #####y=x
    ax[0].set_title('(c)', loc='left')
    ax[0].set_title('WBLSTM Model')
    ax[0].set_xlabel('observed SM (WBLSTM)')
    ax[0].set_ylabel('0d predicted SM (WBLSTM)')
    metrics = 'y=' + '%.2f' % m + 'x' + '+ %.3f' % b
    ax[0].text(0.05,0.67, metrics)
    print('y=' + '%.2f' % m + 'x' + '+ %.3f' % b)
    # out_path = cfg['inputs_path']+cfg['product']+'/' + str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname1+"/focast_time " + str(1) + "/"
    # lstm_predicted = np.load(out_path + '_predictions.npy')
    # observed = np.load(out_path + 'observations.npy')
    # out_path = cfg['inputs_path']+cfg['product']+'/'+ str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname2+"/focast_time " + str(1) + "/"
    # kde_lstm_predicted = np.load(out_path + '_predictions.npy')
    # out_path = cfg['inputs_path']+cfg['product']+'/'+ str(cfg["spatial_resolution"])+'/'+str(cfg['selected_year'][0])+'/'  + cfg['workname'] + "/"+modelname3+"/focast_time " + str(1) + "/"
    # wb_lstm_predicted = np.load(out_path + '_predictions.npy')
    # observed_sm = np.squeeze(observed)
    # observed_sm = observed_sm[-365:, :, :]
    # observed_sm[:, mask == 0] = -0
    #
    # predicted_lstm = np.squeeze(lstm_predicted)
    # predicted_lstm = predicted_lstm[-365:, :, :]
    # predicted_lstm[:, mask == 0] = -0
    #
    # predicted_kde = np.squeeze(kde_lstm_predicted)
    # predicted_kde = predicted_kde[-365:, :, :]
    # predicted_kde[:, mask == 0] = -0
    #
    # predicted_wb = np.squeeze(wb_lstm_predicted)
    # predicted_wb = predicted_wb[-365:, :, :]
    # predicted_wb[:, mask == 0] = -0
    #
    # # fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    # # ax = axes.flatten()
    #
    # # 计算散点图
    # sizes = np.pi * 2 ** 1
    # observed_sm_mean = np.average(observed_sm, axis=0)
    # predicted_sm_mean = np.average(predicted_lstm, axis=0)
    # ############################################
    # sm_index = observed_sm_mean != 0
    # observed_sm_mean = observed_sm_mean[sm_index]
    # predicted_sm_mean = predicted_sm_mean[sm_index]
    # m, b = best_fit_slope_and_intercept(observed_sm_mean, predicted_sm_mean)
    # regression_line = []
    # for a in observed_sm_mean:
    #     regression_line.append((m * a) + b)
    # radius = 0.1  # 半径
    #
    # # axes.ylabel('111111111111111111111111')
    #
    # colormap = plt.get_cmap("jet")
    # Z1 = density_calc(observed_sm_mean, predicted_sm_mean, radius)
    # ax[3].scatter(observed_sm_mean, predicted_sm_mean, c=Z1, s=sizes, cmap=colormap,
    #             norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))
    # ax[3].plot(observed_sm_mean, regression_line, 'red', lw=0.8,label='aaa')  #####回归线
    # ax[3].plot([0.05, 0.7], [0.05, 0.7], 'black', lw=0.8)#####y=x
    # ax[3].set_title('LSTM Model')
    # ax[3].set_title('(d)',loc = 'left')
    # ax[3].set_xlabel('observed SM (LSTM)')
    # ax[3].set_ylabel('1d predicted SM (LSTM)')
    # metrics = 'y=' + '%.2f' % m + 'x' + '+ %.3f' % b
    # ax[3].text(0.05,0.7, metrics)
    # # plt.plot(X, Y_x3, label=u"sin函数")
    # print('y=' + '%.2f' % m + 'x' + '+ %.3f' % b)
    #
    # # 计算散点图
    # sizes = np.pi * 2 ** 1
    # observed_sm_mean = np.average(observed_sm, axis=0)
    # predicted_sm_mean = np.average(predicted_kde, axis=0)
    # ############################################
    # sm_index = observed_sm_mean != 0
    # observed_sm_mean = observed_sm_mean[sm_index]
    # predicted_sm_mean = predicted_sm_mean[sm_index]
    # m, b = best_fit_slope_and_intercept(observed_sm_mean, predicted_sm_mean)
    # regression_line = []
    # for a in observed_sm_mean:
    #     regression_line.append((m * a) + b)
    # radius = 0.1  # 半径
    #
    # colormap = plt.get_cmap("jet")
    # Z1 = density_calc(observed_sm_mean, predicted_sm_mean, radius)
    # ax[4].scatter(observed_sm_mean, predicted_sm_mean, c=Z1, s=sizes, cmap=colormap,
    #               norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))
    # ax[4].plot(observed_sm_mean, regression_line, 'red', lw=0.8)  #####回归线
    # ax[4].plot([0.05, 0.7], [0.05, 0.7], 'black', lw=0.8)  #####y=x
    # ax[4].set_title('PHYsLSTM Model')
    # ax[4].set_title('(e)', loc='left')
    # ax[4].set_xlabel('observed SM (PHYsLSTM)')
    # ax[4].set_ylabel('1d predicted SM (PHYsLSTM)')
    # metrics = 'y=' + '%.2f' % m + 'x' + '+ %.3f' % b
    # ax[4].text(0.05,0.7, metrics)
    # print('y=' + '%.2f' % m + 'x' + '+ %.3f' % b)
    # # 计算散点图
    # sizes = np.pi * 2 ** 1
    # observed_sm_mean = np.average(observed_sm, axis=0)
    # predicted_sm_mean = np.average(predicted_wb, axis=0)
    # ############################################
    # sm_index = observed_sm_mean != 0
    # observed_sm_mean = observed_sm_mean[sm_index]
    # predicted_sm_mean = predicted_sm_mean[sm_index]
    # m, b = best_fit_slope_and_intercept(observed_sm_mean, predicted_sm_mean)
    # regression_line = []
    # for a in observed_sm_mean:
    #     regression_line.append((m * a) + b)
    # radius = 0.1  # 半径
    #
    # colormap = plt.get_cmap("jet")
    # Z1 = density_calc(observed_sm_mean, predicted_sm_mean, radius)
    # ax[5].scatter(observed_sm_mean, predicted_sm_mean, c=Z1, s=sizes, cmap=colormap,
    #               norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()))
    # ax[5].plot(observed_sm_mean, regression_line, 'red', lw=0.8)  #####回归线
    # ax[5].plot([0.05, 0.7], [0.05, 0.7], 'black', lw=0.8)  #####y=x
    # ax[5].set_title('(f)', loc='left')
    # ax[5].set_title('WBLSTM Model')
    # ax[5].set_xlabel('observed SM (WBLSTM)')
    # ax[5].set_ylabel('1d predicted SM (WBLSTM)')
    # metrics = 'y=' + '%.2f' % m + 'x' + '+ %.3f' % b
    # ax[5].text(0.05,0.67, metrics)
    # print('y=' + '%.2f' % m + 'x' + '+ %.3f' % b)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    cfg = get_args()
    main(cfg)

