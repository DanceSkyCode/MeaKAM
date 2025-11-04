import torch.nn as nn
import torch
import math
class MLHLoss1(nn.Module):
    def __init__(self):
        super(MLHLoss1, self).__init__()
    def forward(self, mea, output, target):
        # MAE Loss
        height, weight = mea[: , 6], mea[: , 7]
        mae_loss = torch.mean(torch.abs(output - target))
        losses = (output - target)**2 # torch.Size([8, 123])
        # MSE Loss
        mse_loss = torch.mean(losses)
        
        # RMSE Loss
        rmse_loss = torch.sqrt(mse_loss)
        
        # rRMSE Loss
        max_target = torch.max(target)
        rrms_loss = rmse_loss / max_target
        
        # ρ (rho) metric
        rho_metric = torch.sum((output - torch.mean(output)) * (target - torch.mean(target))) / (
                torch.sqrt(torch.sum((output - torch.mean(output))**2)) * torch.sqrt(torch.sum((target - torch.mean(target))**2)))
        
        return mae_loss, mse_loss, rmse_loss*100, rrms_loss*100, rho_metric, losses
class MLHLoss2(nn.Module):
    def __init__(self):
        super(MLHLoss2, self).__init__()
        self.rho = 1
    def forward(self, inputs, output, target):
        mae_loss = torch.mean(torch.abs(output - target))
        losses = (output - target) ** 2 # torch.Size([8, 123])
        target_diff = torch.abs(target[:, 3:] - target[:, :-3]) # torch.Size([8, 120])
        avg_grad = torch.mean(target_diff, dim=1) # torch.Size([8])
        weights = torch.ones_like(target) # torch.Size([8, 123])
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1] - 3):
                if target_diff[i, j] < avg_grad[i] :
                    weights[i, j] *= self.rho
        weighted_losses = losses * weights
        weighted_loss = torch.sum(weighted_losses)
        # MSE Loss
        mse_loss = torch.mean(weighted_loss)
        # RMSE Loss
        rmse_loss = torch.sqrt(mse_loss)
        # rRMSE Loss
        max_target = torch.max(target)
        rrms_loss = rmse_loss / max_target
        loss = mae_loss + mse_loss + rmse_loss + rrms_loss
        return loss 
class MLHLoss3(nn.Module):
    def __init__(self):
        super(MLHLoss3, self).__init__()

    def forward(self, inputs, output, target):
        mae_loss = torch.mean(torch.abs(output - target))
        mse_loss = torch.mean((output - target) ** 2)
        rmse_loss = torch.sqrt(mse_loss)
        max_target = torch.max(target)
        rrms_loss = rmse_loss / max_target
        frame_diff = torch.abs(output - target)
        weights = torch.ones_like(frame_diff)
        weights[frame_diff <= 0.25 * target] *= math.e
        weights[(frame_diff > 0.25 * target) & (frame_diff <= 0.5 * target)] *= math.e ** 2
        weights[(frame_diff > 0.5 * target) & (frame_diff <= target)] *= math.e ** 3
        weights[frame_diff > target] *= math.e ** 4
        weighted_losses = (output - target) ** 2 * weights
        weighted_loss = torch.sum(weighted_losses)
        loss = mae_loss + mse_loss + rmse_loss + rrms_loss
        return loss