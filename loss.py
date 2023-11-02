import torch.nn
import torch
# Original author : Qingliang Li,Sen Yan, Cheng Zhang, 1/23/2023
class NaNMSELoss():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred,y_true,lossmse):
        # mask = y_true == y_true
        # y_true = y_true[mask]
        # y_pred = torch.squeeze(y_pred[mask])
        loss = torch.sqrt(lossmse(y_true, y_pred))
        return loss
class PHYLoss():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred,y_true,old,dw,lossmse,lossrelu):
        # loss = torch.zeros([2])
        mask = y_true == y_true
        y_true = y_true[mask]
        old = old[mask]
        dw = dw[mask]
        y_pred = torch.squeeze(y_pred[mask])
        rmse = torch.sqrt(lossmse(y_true, y_pred))
        mcls = torch.mean(lossrelu(torch.mul(dw, torch.sub(old, y_pred)))**2)
        #mcls = torch.mean(lossrelu(torch.mul(dw, torch.sub(old, y_pred))))
        # loss[0] = rmse
        # loss[1] = mcls
            # s_lwb = torch.mean(lwb ** 2).cuda()
        loss = 0.1*rmse + 0.9*mcls
        #loss = 0.3*rmse + 0.7*mcls
        #loss = 0.5*rmse + 0.5*mcls
        #loss = 0.7*rmse + 0.3*mcls
        # loss = 0.9*rmse + 0.1*mcls
        return loss
class WBLoss():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred,y_true,old,wb,lossmse,lossrelu):
        loss = torch.zeros([2])
        mask = y_true == y_true
        y_true = y_true[mask]
        old = old[mask]
        wb = wb[mask]
        y_pred = torch.squeeze(y_pred[mask])
        rmse = torch.sqrt(lossmse(y_true, y_pred))
        mcls = torch.mean(torch.abs(torch.sub(wb, torch.sub(y_pred, old))))
        # mcls = torch.mean(torch.abs(torch.sub(y_pred, old)))
        loss[0] = rmse
        loss[1] = 0.000001*mcls
        # loss = rmse + 0.01 * mcls
        return loss




    
