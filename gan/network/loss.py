
import torch

def loss_simi(loss_inputs,cfg):
    onehot_tensor_1 = loss_inputs['onehot_tensor_1'] #（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
    onehot_tensor_2 = loss_inputs['onehot_tensor_2']

    simi = torch.sum(onehot_tensor_1*onehot_tensor_2,dim=-1).sum(dim = -1) # (batch_size,)
    simi = simi / onehot_tensor_1.shape[1]

    simi = simi - cfg.l_simi_thresh
    simi = torch.relu(simi)
    # simi = simi**2
    simi = simi.mean()
    return simi

def loss_pred(loss_inputs,cfg):
    pred_1 = loss_inputs['pred_1'][:,0] #（batch_size）

    return - pred_1.mean()

def loss_potential(loss_inputs,cfg):

    
    epsilong=cfg.l_potential_epsilon
    u1,u2=loss_inputs['latent_1'],loss_inputs['latent_2']#(batch_size*n_sample,latent_size_netP)
    u1=u1.clip(epsilong,1-epsilong)
    u2=u2.clip(epsilong,1-epsilong)
    similarity=(u1*u2).sum(axis=1)/ ( 
        ((u1**2).sum(axis=1))**0.5 * ((u2**2).sum(axis=1))**0.5
                            ) -cfg.l_potential_thresh

    similarity=similarity*(similarity>0)# 针对每个元素选择大于0的
    return similarity.mean()

def loss_entropy(loss_inputs,cfg):
    onehot_tensor_1 = loss_inputs['onehot_tensor_1'] #（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
    onehot_tensor_2 = loss_inputs['onehot_tensor_2']

    entropy_1 = -torch.sum(onehot_tensor_1*torch.log(onehot_tensor_1),dim=-1).sum(dim = -1) # (batch_size,)
    entropy_1 = entropy_1 / onehot_tensor_1.shape[1]

    entropy_2 = -torch.sum(onehot_tensor_2*torch.log(onehot_tensor_2),dim=-1).sum(dim = -1) # (batch_size,)
    entropy_2 = entropy_2 / onehot_tensor_2.shape[1]

    entropy = entropy_1 + entropy_2
    entropy = entropy.mean()
    return entropy

def get_losses(loss_inputs, cfg, current_epoch=0, total_epochs=100):
    # 动态权重调度
    progress = current_epoch / total_epochs
    
    # 早期更注重多样性，后期更注重质量
    diversity_weight = max(0.3, 1.0 - progress * 0.7)  # 从1.0降到0.3
    quality_weight = min(1.0, 0.3 + progress * 0.7)    # 从0.3升到1.0
    
    loss = 0
    loss_components = {}
    
    # 预测损失（质量）
    if cfg.l_pred != 0:
        pred_loss = loss_pred(loss_inputs, cfg)
        weighted_pred_loss = cfg.l_pred * quality_weight * pred_loss
        loss += weighted_pred_loss
        loss_components['pred'] = weighted_pred_loss.item()
    
    # 相似性损失（多样性）
    if cfg.l_simi != 0:
        simi_loss = loss_simi(loss_inputs, cfg)
        weighted_simi_loss = cfg.l_simi * diversity_weight * simi_loss
        loss += weighted_simi_loss
        loss_components['simi'] = weighted_simi_loss.item()
    
    # 潜在空间损失（多样性）
    if cfg.l_potential != 0:
        potential_loss = loss_potential(loss_inputs, cfg)
        weighted_potential_loss = cfg.l_potential * diversity_weight * potential_loss
        loss += weighted_potential_loss
        loss_components['potential'] = weighted_potential_loss.item()
    
    # 添加IC引导损失
    if hasattr(cfg, 'l_ic_guide') and cfg.l_ic_guide != 0:
        ic_guide_loss = loss_ic_guide(loss_inputs, cfg)
        weighted_ic_loss = cfg.l_ic_guide * quality_weight * ic_guide_loss
        loss += weighted_ic_loss
        loss_components['ic_guide'] = weighted_ic_loss.item()
    
    return loss, loss_components

def loss_ic_guide(loss_inputs, cfg):
    """基于预测器的因子表示计算IC引导损失"""
    latent_1 = loss_inputs['latent_1']
    latent_2 = loss_inputs['latent_2']
    
    # 计算因子表示的差异性，鼓励生成不同的因子
    factor_similarity = torch.cosine_similarity(latent_1, latent_2, dim=1).mean()
    
    # 惩罚过高的相似度
    diversity_loss = torch.relu(factor_similarity - 0.5)
    
    return diversity_loss