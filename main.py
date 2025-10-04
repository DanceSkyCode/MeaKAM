import torch
import sys
import os
sys.path.append('C:/fzh/1-KAMcode/KAM_and_KFM_Estimation-main') 
import torch.nn as nn
import math, copy
import torch.nn.functional as F
from Attention import MultiHeadAttention
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataset import load_dataset
from filter import butter_lowpass_filter
from nncore.nn import (build_model, FeedForwardNetwork,
                       Parameter, build_norm_layer)
from loss import MLHLoss1, MLHLoss2, MLHLoss3
from sklearn.model_selection import KFold
import logging
from torch.autograd import Variable
from GuassianDiffusion import GaussianDiffusion, DiffusionOutput
from Diffusion_Denoising import EpsilonTheta
from GaitGraph import GaitHyperGraph
from OutputNet import OutNet
import torch
import matplotlib.pyplot as plt
import seaborn as sns

    

class InertialNet(nn.Module):
    def __init__(self, x_dim, hidden_size, dims,
                 heads=4, num_layers=2,
                 ratio=4,
                 p=0.1,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)
                 ):
        super(InertialNet, self).__init__()
        self.dims = dims
        self.heads = heads
        self.length = 123
        self.relu = nn.ReLU()
        
        # Define convolutional layers with different kernel sizes
        self.conv_layers1 = nn.ModuleList([
            nn.Conv1d(in_channels=x_dim*2, out_channels=hidden_size//4, kernel_size=kernel_size, padding=kernel_size//2)
            for kernel_size in [3, 5, 7]  # Example kernel sizes, you can adjust them
        ])
        self.conv_layers2 = nn.ModuleList([
            nn.Conv1d(in_channels=x_dim*2, out_channels=hidden_size//4, kernel_size=kernel_size, padding=kernel_size//2)
            for kernel_size in [3, 5, 7]  # Example kernel sizes, you can adjust them
        ])    
        self.conv_layers3 = nn.ModuleList([
            nn.Conv1d(in_channels=x_dim*4 - 2, out_channels=hidden_size//4, kernel_size=kernel_size, padding=kernel_size//2)
            for kernel_size in [3, 5, 7]  # Example kernel sizes, you can adjust them
        ])
        self.conv_layers4 = nn.ModuleList([
            nn.Conv1d(in_channels=x_dim*4 - 2, out_channels=hidden_size//4, kernel_size=kernel_size, padding=kernel_size//2)
            for kernel_size in [3, 5, 7]  # Example kernel sizes, you can adjust them
        ])   
        # LSTM layers
        self.rnn_layers = nn.ModuleList([
            nn.LSTM(hidden_size//4, hidden_size, num_layers, batch_first=True, bidirectional=False)
        for _ in range(12)
        ])
        self.denoise_fn = EpsilonTheta(
            target_dim = 1,
            cond_length = 100,
            residual_layers = 8,
            residual_channels = 8,
            dilation_cycle_length = 2,
        )  # dinosing network
        self.diffusion = GaussianDiffusion(
            self.denoise_fn,
            input_size = hidden_size,
            diff_steps = 100,
            loss_type = "l2",
            beta_end = 0.1,
            # share ratio, new argument to control diffusion and sampling
            share_ratio_list = [1,0.6,0.6],
            beta_schedule = "linear",
        )  # diffusion network
        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=hidden_size, cond_size=hidden_size
        )  # distribution output
        self.proj_dist_args = self.distr_output.get_args_proj(
        in_features = hidden_size)  # projection distribution arguments
        self.gaitgraph = GaitHyperGraph(hidden_size)
        self.dropout_layer = nn.Dropout(0.25)
        self.att1 = MultiHeadAttention(heads, 16)
        self.att2 = MultiHeadAttention(heads, 16)
        self.linear_1 = nn.Linear(3, 4, bias=True)
        self.linear_2 = nn.Linear(3, 4, bias=True)
    def distr_args(self, rnn_outputs: torch.Tensor):
        (distr_args,) = self.proj_dist_args(rnn_outputs)
        return distr_args



    def forward(self, sequence, ele, lens):
        # Split sequence and ele
        right_z, left_z, right_x, left_x, right_ele, left_ele, right_max, left_max = (
            sequence[:, :, [2, 5, 8, 11]], sequence[:, :, [14, 17, 20, 23]],
            sequence[:, :, [0, 3, 6, 9]], sequence[:, :, [12, 15, 18, 21]],
            sequence[:, :, 24:27], sequence[:, :, 27:30],
            ele[:, :, 0:3], ele[:, :, 3:6]
        )
        

        lx = torch.cat((right_x, left_x), dim = 2)
        lz = torch.cat((right_z, left_z), dim = 2)
        fx = torch.cat((right_x, left_x, right_ele, left_ele), dim = 2)
        fz = torch.cat((right_z, left_z, right_ele, left_ele), dim = 2)
        lx = [lx,lx,lx]
        lz = [lz,lz,lz]
        fx = [fx,fx,fx]
        fz = [fz,fz,fz]
        # Apply convolutional layers with different kernel sizes to each input
        conv_outputs1 = []
        for input_data, conv_layer in zip(lx , self.conv_layers1):
            input_data = input_data.permute(0, 2, 1)  # Conv1d expects input channels as the second dimension
            conv_output = F.relu(conv_layer(input_data))
            conv_outputs1.append(conv_output.permute(0, 2, 1))  # Reshape back to (batch_size, seq_len, hidden_size)

        conv_outputs2 = []
        for input_data, conv_layer in zip(lz , self.conv_layers2):
            input_data = input_data.permute(0, 2, 1)  # Conv1d expects input channels as the second dimension
            conv_output = F.relu(conv_layer(input_data))
            conv_outputs2.append(conv_output.permute(0, 2, 1))  # Reshape back to (batch_size, seq_len, hidden_size)

        conv_outputs3 = []
        for input_data, conv_layer in zip(fx , self.conv_layers3):
            input_data = input_data.permute(0, 2, 1)  # Conv1d expects input channels as the second dimension
            conv_output = F.relu(conv_layer(input_data))
            conv_outputs3.append(conv_output.permute(0, 2, 1))  # Reshape back to (batch_size, seq_len, hidden_size)

        conv_outputs4 = []
        for input_data, conv_layer in zip(fz , self.conv_layers4):
            input_data = input_data.permute(0, 2, 1)  # Conv1d expects input channels as the second dimension
            conv_output = F.relu(conv_layer(input_data))
            conv_outputs4.append(conv_output.permute(0, 2, 1))  # Reshape back to (batch_size, seq_len, hidden_size)

        # xt1 = self.att1(conv_outputs[2], conv_outputs2[2], conv_outputs2[2], mask=None)
        # xt2 = self.att2(conv_outputs2[2], conv_outputs[2], conv_outputs[2], mask=None)
        rnn_out = [conv_outputs1[0] , conv_outputs4[0] , conv_outputs3[0] , conv_outputs2[0], 
                   conv_outputs1[1] , conv_outputs4[1] , conv_outputs3[1] , conv_outputs2[1],
                     conv_outputs1[2] , conv_outputs4[2] , conv_outputs3[2] , conv_outputs2[2]]
        titles = ["conv_outputs1[0]", "conv_outputs4[0]", "conv_outputs3[0]", "conv_outputs2[0]",
                "conv_outputs1[1]", "conv_outputs4[1]", "conv_outputs3[1]", "conv_outputs2[1]",
                "conv_outputs1[2]", "conv_outputs4[2]", "conv_outputs3[2]", "conv_outputs2[2]"]
        save_dir = r"C:\fzh\3_KAM\pic"
        visualize_and_save_feature_maps(rnn_out, titles,save_dir)        
        # Pass through LSTM layers
        lstm_outputs = []
        for conv_output, rnn_layer in zip(rnn_out, self.rnn_layers):
            lstm_output, _ = rnn_layer(conv_output)
            lstm_outputs.append(lstm_output)
        lstm_outputs = [lstm_outputs[0] * lstm_outputs[1] - lstm_outputs[2] * lstm_outputs[3],
                        lstm_outputs[4] * lstm_outputs[5] - lstm_outputs[6] * lstm_outputs[7],
                        lstm_outputs[8] * lstm_outputs[9] - lstm_outputs[10] * lstm_outputs[11]]
        graph_outputs = self.gaitgraph(lstm_outputs)
        distr_args = [self.distr_args(rnn_output)
                      for rnn_output in graph_outputs]
        output = distr_args[0] + distr_args[1] + distr_args[2] + lstm_outputs[0] + lstm_outputs[1] + lstm_outputs[2]
        return output
def visualize_and_save_feature_maps(feature_maps, titles, save_dir):
    for i, (feature_map, title) in enumerate(zip(feature_maps, titles)):
        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.detach().cpu().numpy()  
            
        avg_feature_map = feature_map.mean(axis=0)  
        plt.figure(figsize=(12, 8), dpi=300)  
        sns.heatmap(avg_feature_map, cmap="viridis", cbar=True)
        plt.title(title)
        plt.xlabel("Hidden Size")
        plt.ylabel("Time Steps")
        save_path = os.path.join(save_dir, f"{title}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  
class DirectNet(nn.Module):
    """ Implemented based on the paper "Efficient low-rank multimodal fusion with modality-specific factors" """
    def __init__(self, input_size, hidden_size, dims, heads=2, num_layers=1):
        super(DirectNet, self).__init__()
        self.attn = InertialNet(input_size, hidden_size, dims,
                 heads=2, num_layers = 1)
        self.out_net = OutNet(globals()['lstm_unit'])
    def __str__(self):
        return 'Direct fusion net'
    def forward(self, acc_x, mea, lens):
        mea = mea.unsqueeze(1).repeat(1, sequence_length, 1)
        ele, height, weight = mea[ : , : , 0 : 6 ], mea[: , : , 6], mea[: ,  : , 7]
        acc_h = self.attn(acc_x, ele, lens)
        sequence = self.out_net(acc_h, ele, height, weight, lens)
        return sequence
# h5_file = 'C:/fzh/1-KAMcode/dataset.h5'
input_data, output_data, measure_data = load_dataset(h5_file)
print("Input data shape:", input_data.shape)
print("Output data shape:", output_data.shape)
print("Output data shape:", measure_data.shape)
X = torch.tensor(input_data, dtype=torch.float)
Y = torch.tensor(output_data, dtype=torch.float)
Z = torch.tensor(measure_data, dtype=torch.float)
input_size = 4
dims = 256
heads = 2
hidden_size = 256
num_layers = 4
batch_size = 8
num_epochs = 100
learning_rate = 0.01
sequence_length= 123
best_param = {'lstm_unit': hidden_size, 'linear1_unit': 64, 'linear2_unit': 16, 'linear3_unit': 4}
globals().update(best_param)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
dataset = TensorDataset(X, Y, Z)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
lens = [sequence_length]*batch_size
lens = torch.tensor(lens, dtype=torch.float)
kf = KFold(n_splits=10, random_state=None)
for train_index, test_index in kf.split(X):
      # print("Train:", train_index, "Validation:",test_index)
      X_train, X_test = X[train_index], X[test_index] 
      y_train, y_test = Y[train_index], Y[test_index] 
      z_train, z_test = Z[train_index], Z[test_index] 
train_losses_all = []
test_losses_all = []
all_outputs = []
max_predictions = []  
max_labels = []
loss_mse = 0
loss_mae = 0
loss_rmse = 0
loss_rrms = 0
length_mse_loss = []
rho = 0
output_files = ['output_fold_{}.csv'.format(i) for i in range(1, kf.n_splits + 1)]
label_files = ['label_fold_{}.csv'.format(i) for i in range(1, kf.n_splits + 1)]
ii = 0
for train_index, test_index in kf.split(X, Y, Z):
    ii = ii + 1
    model = DirectNet(input_size, hidden_size, dims, heads, num_layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion1 = MLHLoss1()
    criterion2 = MLHLoss3()
    train_losses = []
    test_losses = []
    print('\n{} of kfold {}'.format(ii,kf.n_splits))
    X_train, X_test = X[train_index], X[test_index] 
    Y_train, Y_test = Y[train_index], Y[test_index]
    Z_train, Z_test = Z[train_index], Z[test_index]
    
    train_dataset = TensorDataset(X_train, Y_train, Z_train)
    test_dataset = TensorDataset(X_test, Y_test, Z_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size*9, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(num_epochs):
        epoch_losses = []
        with torch.enable_grad():
            model.train()
            for i, data in enumerate(train_dataloader):
                inputs, labels, mea = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                mea = mea.to(device)
                outputs = model(inputs, mea, lens)
                loss = criterion2(inputs, outputs, labels)
                optimizer.zero_grad()            
                loss.backward()
                optimizer.step()
                lr = optimizer.param_groups[0]['lr']  
                if epoch % 70 == 1 and epoch > 2 and i == 1:
                    optimizer.param_groups[0]['lr'] = lr * 0.1
                epoch_losses.append(loss.item())
        epoch_loss_mean = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(epoch_loss_mean)

        if epoch % 10 == 9 and epoch > 2:
            with torch.no_grad():
                model.eval()
                predictions = []
                for i, data in enumerate(test_dataloader):
                    inputs, labels, mea= data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    mea = mea.to(device)
                    mea = mea.to(device)
                    outputs = model(inputs, mea, lens)
                    predictions.extend(outputs.cpu().detach().numpy().tolist())
                    mae_loss, mse_loss, rmse_loss, rrms_loss, rho_metric, losses= criterion1(mea, outputs, labels)
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], MAE Loss: %.7f, MSE Loss: %.7f, RMSE Loss: %.7f, rRMSE Loss: %.7f, (rho) metric: %.7f' %
                  (mae_loss.item(), mse_loss.item(), rmse_loss.item(), rrms_loss.item(), rho_metric.item()))
    with torch.no_grad():
        model.eval()
        predictions = []
        for i, data in enumerate(test_dataloader):
            inputs, labels, mea= data
            inputs = inputs.to(device)
            labels = labels.to(device)
            mea = mea.to(device)
            outputs = model(inputs, mea, lens)
            predictions.extend(outputs.cpu().numpy().tolist())
            all_outputs.extend(outputs.cpu().numpy().tolist())
            max_prediction = torch.max(outputs, dim=1)[0]
            max_label = torch.max(labels, dim=1)[0]
            max_predictions.extend(max_prediction.cpu().numpy().tolist())   
            max_labels.extend(max_label.cpu().numpy().tolist())   
            mae_loss, mse_loss, rmse_loss, rrms_loss, rho_metric, losses = criterion1(mea, outputs, labels)
            loss_mae = loss_mae + mae_loss
            loss_mse = loss_mse + mse_loss
            loss_rmse = loss_rmse + rmse_loss
            loss_rrms = loss_rrms +rrms_loss
            rho = rho + rho_metric
            length_mse_loss.extend(losses.cpu().numpy().tolist())
avg_mae_loss = loss_mae / 10
avg_mse_loss = loss_mse / 10
avg_rmse_loss = loss_rmse / 10
avg_rrms_loss = loss_rrms / 10
avg_rho_metric = rho / 10
print("Average MAE Loss:", avg_mae_loss)
print("Average MSE Loss:", avg_mse_loss)
print("Average RMSE Loss:", avg_rmse_loss)
print("Average rRMSE Loss:", avg_rrms_loss)
print("Average Rho Metric:", avg_rho_metric)
average_output = torch.mean(torch.tensor(all_outputs), dim=0)
average_output_np = average_output.cpu().numpy()
max_list =np.array(max_predictions)
prediction = np.array(all_outputs)
max_label_list = np.array(max_labels)
average_loss_mse = torch.mean(torch.tensor(length_mse_loss), dim=0)# torch.Size([80, 123])
length_mse_loss_np = np.array(length_mse_loss)
length_mse_loss_transposed = np.transpose(length_mse_loss_np)
np.savetxt('C:/fzh/experiment/Difflength_mse_loss_rho1.csv', length_mse_loss_transposed, delimiter=',')
average_loss_mae = torch.sqrt(average_loss_mse)# torch.Size([123])
average_loss_np = average_loss_mae.cpu().numpy()
df1 = pd.DataFrame(average_output_np, columns=['average_output'])
df2 = pd.DataFrame(average_loss_np, columns=['average_loss_np'])
df3 = pd.DataFrame(max_list, columns=['max_batch'])
df4 = pd.DataFrame(max_label_list, columns=['max_label'])
df5 = pd.DataFrame(prediction)
df1.to_csv('C:/fzh/experiment/Diffaverage_output_rho1.csv', index=False)
df2.to_csv('C:/fzh/experiment/Diffaverage_loss_rho1.csv', index=False)
df3.to_csv('C:/fzh/experiment/Diffmax_batch_rho1.csv', index=False)
df4.to_csv('C:/fzh/experiment/Diffmax_label_rho1.csv', index=False)
df5.to_csv('C:/fzh/experiment/Diffprediction.csv', index=False)

file_path = "C:/fzh/experiment/Diffloss_metrics_rho1.txt"
with open(file_path, "w") as file:
    file.write("Average MAE Loss: {}\n".format(avg_mae_loss))
    file.write("Average MSE Loss: {}\n".format(avg_mse_loss))
    file.write("Average RMSE Loss: {}\n".format(avg_rmse_loss))
    file.write("Average rRMSE Loss: {}\n".format(avg_rrms_loss))
    file.write("Average Rho Metric: {}\n".format(avg_rho_metric))

print("Loss metrics saved.", file_path)