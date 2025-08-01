import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import copy

import torch
from torch.utils.tensorboard import SummaryWriter


class NetP(nn.Module):
    def __init__(
        self,
        n_chars ,
        hidden,
        seq_len,
        factor_dim=64,  # 新增：因子表示维度
    ):
        super().__init__()
        assert seq_len == 20
        self.convs = nn.Sequential(
            nn.Conv2d(n_chars, 96, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(96, 128, kernel_size=(1, 4)),
            nn.ReLU(),
        )  # [50, 128, 1, 6]
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 6, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(nn.Linear(256, 1))
        # 新增：因子表示头
        self.factor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, factor_dim),
            nn.Tanh()  # 归一化到[-1, 1]
        )
        self.factor_dim = factor_dim

    def forward(self, x, latent=False, return_factor=False):
        x = x.float()
        x = x.permute(0, 2, 1)[:, :, None]
        x = self.convs(x)
        x = x.reshape([x.shape[0], 128 * 6])
        latent_tensor = self.fc1(x)
        score = self.fc2(latent_tensor)
        
        if return_factor:
            factor_repr = self.factor_head(latent_tensor)
            return score, latent_tensor, factor_repr
        elif latent:
            return score, latent_tensor
        else:
            return score
        
    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1  :
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)
    

    
class NetP_CNN(nn.Module):
    def __init__(self, n_chars, seq_len, hidden):
        super().__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.hidden = hidden
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
        )
        self.conv1d = nn.Conv1d(n_chars, hidden, 1)
        self.linear = nn.Linear(seq_len*hidden, 1)

    def forward(self, input,latent=False):
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.seq_len*self.hidden)
        latent_tensor = output
        output = self.linear(latent_tensor)
        if latent:
            return output, latent_tensor
        else:
            return output
        

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1  :
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


def train_regression_model(
    train_loader,
    valid_loader,
    net,
    loss_fn,
    optimizer,
    num_epochs=10,
    use_tensorboard=True,
    tensorboard_path="logs",
    early_stopping_patience=None,
):
    # Initialize TensorBoard SummaryWriter if requested
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(tensorboard_path)

    # Initialize variables for early stopping
    best_valid_loss = float("inf")
    best_weights = None
    patience_counter = 0

    # Training loop for regression
    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0
        total_samples_train = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = net(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_y.size(0)
            total_samples_train += batch_y.size(0)

        average_train_loss = total_train_loss / total_samples_train

        # Validation
        net.eval()
        with torch.no_grad():
            total_valid_loss = 0
            total_samples_valid = 0
            for batch_x, batch_y in valid_loader:
                outputs = net(batch_x)
                loss = loss_fn(outputs, batch_y)
                total_valid_loss += loss.item() * batch_y.size(0)
                total_samples_valid += batch_y.size(0)

            average_valid_loss = total_valid_loss / total_samples_valid

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Validation Loss: {average_valid_loss:.4f}"
            )

            # Write to TensorBoard if requested
            if use_tensorboard:
                writer.add_scalar("Train Loss", average_train_loss, epoch)
                writer.add_scalar("Validation Loss", average_valid_loss, epoch)

            # Early Stopping
            if (
                early_stopping_patience is not None
                and average_valid_loss < best_valid_loss
            ):
                best_valid_loss = average_valid_loss
                patience_counter = 0
                best_weights = copy.deepcopy(net.state_dict())  # Record the best weights
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load the best weights back to the net
    if best_weights is not None:
        net.load_state_dict(best_weights)

    # Close the TensorBoard SummaryWriter if used
    if use_tensorboard:
        writer.close()



def train_regression_model_with_weight(
    train_loader,
    valid_loader,
    net,
    loss_fn,
    optimizer,
    device = 'cuda:0',
    num_epochs=10,
    use_tensorboard=True,
    tensorboard_path="logs",
    early_stopping_patience=None,
):
    # Initialize TensorBoard SummaryWriter if requested
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(tensorboard_path)

    # Initialize variables for early stopping
    best_valid_loss = float("inf")
    best_weights = None
    patience_counter = 0

    # Training loop for regression
    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0
        total_samples_train = 0
        for batch_x, batch_y, batch_w in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)

            optimizer.zero_grad()
            outputs = net(batch_x)
            loss = loss_fn(outputs, batch_y, batch_w)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_y.size(0)
            total_samples_train += batch_y.size(0)

        average_train_loss = total_train_loss / total_samples_train

        # Validation
        net.eval()
        with torch.no_grad():
            total_valid_loss = 0
            total_samples_valid = 0
            for batch_x, batch_y, batch_w in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_w = batch_w.to(device)
                outputs = net(batch_x)
                loss = loss_fn(outputs, batch_y, batch_w)
                total_valid_loss += loss.item() * batch_y.size(0)
                total_samples_valid += batch_y.size(0)

            average_valid_loss = total_valid_loss / total_samples_valid

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss:.5f}, Validation Loss: {average_valid_loss:.5f}"
            )

            # Write to TensorBoard if requested
            if use_tensorboard:
                writer.add_scalar("Train Loss", average_train_loss, epoch)
                writer.add_scalar("Validation Loss", average_valid_loss, epoch)

            # Early Stopping
            if (
                early_stopping_patience is not None
                and average_valid_loss < best_valid_loss - 1e-5
            ):
                best_valid_loss = average_valid_loss
                patience_counter = 0
                best_weights = copy.deepcopy(net.state_dict())  # Record the best weights
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load the best weights back to the net
    if best_weights is not None:
        net.load_state_dict(best_weights)

    # Close the TensorBoard SummaryWriter if used
    if use_tensorboard:
        writer.close()


import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 重写上面的函数 要求针对不同样本 有权重
def train_net_p_with_weight(cfg, net, x, y, weights, data, target, lr=0.001):
    from alphagen.utils.correlation import batch_pearsonr
    from alphagen.utils.pytorch_utils import normalize_by_day
    
    # 准备目标收益率数据
    target_returns = normalize_by_day(target.evaluate(data))  # (n_days, n_stocks)
    
    x_train, x_valid, y_train, y_valid, weights_train, weights_valid = train_test_split(
        x, y, weights, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train, weights_train),
        batch_size=cfg.batch_size_p, shuffle=True)
    valid_loader = DataLoader(
        TensorDataset(x_valid, y_valid, weights_valid), 
        batch_size=cfg.batch_size_p, shuffle=False)
    
    # 修改损失函数，加入IC损失
    def combined_loss(pred_scores, true_scores, weights, factor_reprs, batch_indices):
        # 原始的加权MSE损失
        mse_loss = ((pred_scores - true_scores)**2 * weights).mean()
        
        # IC损失：通过因子表示计算IC
        ic_loss = 0.0
        if factor_reprs is not None:
            # 将因子表示映射到实际的因子值
            # 这里需要一个简单的解码器，可以是线性层
            batch_size = factor_reprs.shape[0]
            ic_losses = []
            
            for i in range(min(batch_size, 10)):  # 采样计算，避免计算量过大
                # 模拟因子值（实际中可能需要更复杂的解码）
                factor_values = factor_reprs[i].unsqueeze(0).repeat(data.n_days, data.n_stocks, 1)
                factor_values = factor_values.mean(dim=-1)  # 简化：取平均作为因子值
                factor_values = normalize_by_day(factor_values)
                
                # 计算IC
                ic = batch_pearsonr(factor_values, target_returns).mean()
                ic_losses.append(-ic)  # 负IC作为损失（最大化IC）
            
            if ic_losses:
                ic_loss = torch.stack(ic_losses).mean()
        
        # 组合损失
        total_loss = mse_loss + cfg.ic_loss_weight * ic_loss
        return total_loss, mse_loss, ic_loss
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # 训练循环
    best_valid_loss = float("inf")
    best_weights = None
    patience_counter = 0
    
    for epoch in range(cfg.num_epochs_p):
        net.train()
        total_train_loss = 0
        total_mse_loss = 0
        total_ic_loss = 0
        total_samples_train = 0
        
        for batch_idx, (batch_x, batch_y, batch_w) in enumerate(train_loader):
            batch_x = batch_x.to(cfg.device)
            batch_y = batch_y.to(cfg.device)
            batch_w = batch_w.to(cfg.device)
            
            optimizer.zero_grad()
            outputs, _, factor_reprs = net(batch_x, return_factor=True)
            
            loss, mse_loss, ic_loss = combined_loss(
                outputs, batch_y, batch_w, factor_reprs, 
                batch_idx * cfg.batch_size_p + torch.arange(batch_x.size(0)))
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * batch_y.size(0)
            total_mse_loss += mse_loss.item() * batch_y.size(0)
            total_ic_loss += ic_loss.item() * batch_y.size(0)
            total_samples_train += batch_y.size(0)
        
        # 验证
        net.eval()
        with torch.no_grad():
            total_valid_loss = 0
            total_samples_valid = 0
            
            for batch_x, batch_y, batch_w in valid_loader:
                batch_x = batch_x.to(cfg.device)
                batch_y = batch_y.to(cfg.device)
                batch_w = batch_w.to(cfg.device)
                
                outputs, _, factor_reprs = net(batch_x, return_factor=True)
                loss, _, _ = combined_loss(outputs, batch_y, batch_w, factor_reprs, None)
                
                total_valid_loss += loss.item() * batch_y.size(0)
                total_samples_valid += batch_y.size(0)
            
            average_valid_loss = total_valid_loss / total_samples_valid
            
        print(f"Epoch [{epoch+1}/{cfg.num_epochs_p}], "
              f"Train Loss: {total_train_loss/total_samples_train:.5f} "
              f"(MSE: {total_mse_loss/total_samples_train:.5f}, "
              f"IC: {total_ic_loss/total_samples_train:.5f}), "
              f"Valid Loss: {average_valid_loss:.5f}")
        
        # Early stopping
        if average_valid_loss < best_valid_loss - 1e-5:
            best_valid_loss = average_valid_loss
            patience_counter = 0
            best_weights = copy.deepcopy(net.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= cfg.es_p:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_weights is not None:
        net.load_state_dict(best_weights)

