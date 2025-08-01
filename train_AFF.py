import torch
print(torch.version.cuda)
import os
from gan.dataset import Collector
from gan.network.masker import NetM
from gan.network.predictor import train_regression_model_with_weight
from alphagen.rl.env.wrapper import SIZE_ACTION
from gan.utils import Builders
from alphagen_generic.features import *
from alphagen.data.expression import *
from alphagen.utils.correlation import batch_ret,batch_pearsonr
import numpy as np
from alphagen.utils.random import reseed_everything
from gan.utils import filter_valid_blds,save_blds
from gan.network.generater import train_network_generator
import gc
from gan.utils.data import get_data_by_year
from alphagen_generic.features import target
import copy
from alphagen.utils.pytorch_utils import normalize_by_day
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import scipy.stats


# 检查PyTorch是否是CUDA版本
is_cuda_available = torch.cuda.is_available()
print(f"PyTorch CUDA support is available: {is_cuda_available}")

if is_cuda_available:
    # 查看PyTorch编译时使用的CUDA版本号
    cuda_version_pytorch = torch.version.cuda
    print(f"PyTorch was compiled with CUDA version: {cuda_version_pytorch}")

    # 查看当前使用的GPU设备
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")
    
def pre_process_y(y):
    # 方案1：使用排序
    ranks = scipy.stats.rankdata(y) / len(y)
    return ranks
    

def numpy2onehot(integer_matrix,max_num_categories=None,min_num_categories=None):
    if max_num_categories is None:
        max_num_categories = np.max(integer_matrix) + 1
    if min_num_categories is None:
        min_num_categories = np.min(integer_matrix)
    integer_matrix = integer_matrix - min_num_categories
    num_categories = max_num_categories - min_num_categories
    return np.eye(num_categories)[integer_matrix]

from typing import List

def blds_list_to_tensor(blds_list,weights_list:List[int]):
    assert len(blds_list) == len(weights_list)

    x_numpy_list = []
    y_numpy_list = []
    weights_numpy_list = []
    for blds,weight_int in zip(blds_list,weights_list):
        x_numpy = numpy2onehot(np.array(blds.builders_tokens),SIZE_ACTION,0).astype('float32')
        y_numpy = np.array(blds.scores).astype('float32')[:,None]
        weights_numpy = np.ones(x_numpy.shape[0]).astype('float32')[:,None] * weight_int
        x_numpy_list.append(x_numpy)
        y_numpy_list.append(y_numpy)
        weights_numpy_list.append(weights_numpy)
    x_numpy = np.concatenate(x_numpy_list,axis=0)
    y_numpy = np.concatenate(y_numpy_list,axis=0)
    weights_numpy = np.concatenate(weights_numpy_list,axis=0)
    x = torch.from_numpy(x_numpy)
    y = torch.from_numpy(y_numpy)
    weights = torch.from_numpy(weights_numpy)
    return x,y,weights

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# 修改：完全重写train_net_p_with_weight函数，加入IC损失
def train_net_p_with_weight(cfg, net, x, y, weights, data, target, lr=0.001):
    """训练预测器，同时优化分数预测和IC"""
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

    # 修改：新的组合损失函数
    def combined_loss(pred_scores, true_scores, weights, factor_reprs=None):
        # 原始的加权MSE损失
        mse_loss = ((pred_scores - true_scores) ** 2 * weights).mean()

        # IC损失：通过因子表示计算IC（简化版本）
        ic_loss = torch.tensor(0.0).to(cfg.device)
        if factor_reprs is not None and cfg.ic_loss_weight > 0:
            # 使用因子表示的统计特性作为IC的代理
            # 鼓励因子表示具有高方差（信息量）和低相关性（独特性）
            factor_std = factor_reprs.std(dim=0).mean()
            ic_proxy_loss = -factor_std  # 最大化标准差
            ic_loss = ic_proxy_loss

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

            # 修改：检查网络是否支持return_factor参数
            try:
                outputs, _, factor_reprs = net(batch_x, return_factor=True)
            except:
                outputs = net(batch_x)
                factor_reprs = None

            loss, mse_loss, ic_loss = combined_loss(
                outputs, batch_y, batch_w, factor_reprs)

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

                try:
                    outputs, _, factor_reprs = net(batch_x, return_factor=True)
                except:
                    outputs = net(batch_x)
                    factor_reprs = None

                loss, _, _ = combined_loss(outputs, batch_y, batch_w, factor_reprs)

                total_valid_loss += loss.item() * batch_y.size(0)
                total_samples_valid += batch_y.size(0)

            average_valid_loss = total_valid_loss / total_samples_valid

        # 修改：更详细的训练日志
        print(f"Epoch [{epoch + 1}/{cfg.num_epochs_p}], "
              f"Train Loss: {total_train_loss / total_samples_train:.5f} "
              f"(MSE: {total_mse_loss / total_samples_train:.5f}, "
              f"IC: {total_ic_loss / total_samples_train:.5f}), "
              f"Valid Loss: {average_valid_loss:.5f}")

        # Early stopping
        if average_valid_loss < best_valid_loss - 1e-5:
            best_valid_loss = average_valid_loss
            patience_counter = 0
            best_weights = copy.deepcopy(net.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= cfg.es_p:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_weights is not None:
        net.load_state_dict(best_weights)


# 修改：新增改进的生成器训练函数
def train_network_generator_improved(netG, netM, netP, cfg, data, target,
                                     current_round, random_method, metric, lr, n_actions,
                                     total_iterations=50):
    """改进的生成器训练，支持动态权重和更好的损失记录"""
    from gan.network.loss import get_losses

    opt = torch.optim.Adam(netG.parameters(), lr=lr)
    best_weights = None
    best_score = -float('inf')
    patience_counter = 0
    z1 = torch.zeros([cfg.batch_size, cfg.potential_size]).to(cfg.device)
    z2 = torch.zeros([cfg.batch_size, cfg.potential_size]).to(cfg.device)

    netM.eval()
    netP.eval()

    empty_blds = None
    best_str_to_print = ''

    for epoch in range(cfg.num_epochs_g):
        netG.train()
        opt.zero_grad()
        z1 = random_method(z1)
        z2 = random_method(z2)
        logit_raw_1 = netG(z1)
        logit_raw_2 = netG(z2)

        masked_x_1, masks_1, blds_1 = netM(logit_raw_1)
        masked_x_2, masks_2, blds_2 = netM(logit_raw_2)

        onehot_tensor_1 = torch.nn.functional.gumbel_softmax(masked_x_1, hard=True)
        pred_1, latent_1 = netP(onehot_tensor_1, latent=True)

        onehot_tensor_2 = torch.nn.functional.gumbel_softmax(masked_x_2, hard=True)
        pred_2, latent_2 = netP(onehot_tensor_2, latent=True)

        loss_inputs = {
            'logit_raw_1': logit_raw_1,
            'logit_raw_2': logit_raw_2,
            'masked_x_1': masked_x_1,
            'masked_x_2': masked_x_2,
            'masks_1': masks_1,
            'masks_2': masks_2,
            'blds_1': blds_1,
            'blds_2': blds_2,
            'z1': z1,
            'z2': z2,
            'onehot_tensor_1': onehot_tensor_1,
            'onehot_tensor_2': onehot_tensor_2,
            'pred_1': pred_1,
            'pred_2': pred_2,
            'latent_1': latent_1,
            'latent_2': latent_2,
        }

        # 修改：使用动态权重的损失计算
        if cfg.use_dynamic_weights:
            # 计算当前进度
            current_epoch = epoch + current_round * cfg.num_epochs_g
            total_epochs = total_iterations * cfg.num_epochs_g
            progress = min(current_epoch / total_epochs, 1.0)

            # 动态调整损失权重
            loss = get_dynamic_losses(loss_inputs, cfg, progress)
        else:
            loss = get_losses(loss_inputs, cfg)

        blds: Builders = blds_1 + blds_2
        idx = [i for i in range(blds.batch_size) if blds.builders[i].is_valid()]
        blds.drop_invalid()
        blds.evaluate(data, target, metric)

        str_to_print = f"##{epoch}/{cfg.num_epochs_g} R{current_round}: n_valid_train:{len(idx)}, n_valid:{len(blds.scores)}, loss:{loss:.4f}"
        mean_score = np.mean(blds.scores) if len(blds.scores) > 0 else 0
        max_score = np.max(blds.scores) if len(blds.scores) > 0 else 0
        std_score = np.std(blds.scores) if len(blds.scores) > 0 else 0
        str_to_print += f", max_score:{max_score:.4f}, mean_score:{mean_score:.4f}, std_score:{std_score:.4f}"
        blds.drop_duplicated()
        str_to_print += f", unique:{blds.batch_size}"
        print(str_to_print)

        if max_score > 0:
            exprs = blds.exprs_str[np.argmax(blds.scores)]
            print(f"Max score {max_score} expr: {exprs}")

        if empty_blds is None:
            empty_blds = blds
        else:
            empty_blds = empty_blds + blds

        if cfg.g_es_score == 'mean':
            es_score = mean_score
        elif cfg.g_es_score == 'max':
            es_score = max_score
        elif cfg.g_es_score == 'combined':
            es_score = max_score + 2. * std_score
        else:
            raise NotImplementedError

        if es_score > best_score:
            best_score = es_score
            best_weights = copy.deepcopy(netG.state_dict())
            best_str_to_print = str_to_print
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > cfg.g_es:
                print(f'Early stopping triggered at epoch {epoch}, best score: {best_score}!')
                break

        if epoch > 0:
            loss.backward()
            opt.step()

    if best_weights is not None:
        print('Loading best weights')
        netG.load_state_dict(best_weights)
        print(best_str_to_print)

    empty_blds.drop_duplicated()
    return empty_blds


# 修改：新增动态损失函数
def get_dynamic_losses(loss_inputs, cfg, progress):
    """根据训练进度动态调整损失权重"""
    # 早期更注重多样性，后期更注重质量
    diversity_weight = max(0.3, 1.0 - progress * 0.7)  # 从1.0降到0.3
    quality_weight = min(1.0, 0.3 + progress * 0.7)  # 从0.3升到1.0

    loss = 0

    # 导入原始损失函数
    from gan.network.loss import loss_simi, loss_pred, loss_potential, loss_entropy

    # 预测损失（质量）
    if cfg.l_pred != 0:
        pred_loss = loss_pred(loss_inputs, cfg)
        loss += cfg.l_pred * quality_weight * pred_loss

    # 相似性损失（多样性）
    if cfg.l_simi != 0:
        simi_loss = loss_simi(loss_inputs, cfg)
        loss += cfg.l_simi * diversity_weight * simi_loss

    # 潜在空间损失（多样性）
    if cfg.l_potential != 0:
        potential_loss = loss_potential(loss_inputs, cfg)
        loss += cfg.l_potential * diversity_weight * potential_loss

    # 熵损失
    if cfg.l_entropy != 0:
        entropy_loss = loss_entropy(loss_inputs, cfg)
        loss += cfg.l_entropy * entropy_loss

    return loss


def get_metric(zoo_blds,device,corr_thresh=0.5,metric_target='ic'):

    n_blds = len(zoo_blds)
    if n_blds >0:
        n_days = len(zoo_blds.ret_list[0])
        existed = zoo_blds.ret_list # [blds1,blds2,...] ,blds1: (n_days,)
        existed = np.vstack(existed) # (n_blds,n_days)
        existed = torch.from_numpy(existed).to(device)
        assert existed.shape == (n_blds,n_days)
        print(f"existed n_blds == {n_blds}")
    else:
        print("n_blds == 0")
    
    def get_score(fct,tgt):
        # 添加更健壮的NaN和数据检查
        if torch.isnan(fct).all() or torch.isnan(tgt).all():
            return {'score': 0., 'ret': np.zeros(tgt.shape[1] if len(tgt.shape) > 1 else tgt.shape[0]), 
                   'multi_score': {'ic': 0, 'icir': 0, 'ret': 0, 'sharpe': 0, 'retir': 0}}
        
        # 确保有足够的有效数据点
        valid_mask = ~(torch.isnan(fct) | torch.isnan(tgt))
        valid_ratio = valid_mask.float().mean()
        
        if valid_ratio < 0.1:  # 如果有效数据少于10%
            return {'score': 0., 'ret': np.zeros(tgt.shape[1] if len(tgt.shape) > 1 else tgt.shape[0]), 
                   'multi_score': {'ic': 0, 'icir': 0, 'ret': 0, 'sharpe': 0, 'retir': 0}}
        
        # 计算指标
        try:
            ret = batch_ret(fct,tgt)
            ic = batch_pearsonr(fct,tgt)
            
            # 添加更健壮的统计计算
            ic_finite = ic[torch.isfinite(ic)]
            ret_finite = ret[torch.isfinite(ret)]
            
            if len(ic_finite) == 0 or len(ret_finite) == 0:
                return {'score': 0., 'ret': np.zeros(tgt.shape[1] if len(tgt.shape) > 1 else tgt.shape[0]), 
                       'multi_score': {'ic': 0, 'icir': 0, 'ret': 0, 'sharpe': 0, 'retir': 0}}
            
            ic_mean = ic_finite.mean().abs().item()
            icir = (ic_mean / ic_finite.std()).item() if ic_finite.std() > 1e-8 else 0.
            ret_mean = ret_finite.mean().abs().item()
            ret_ir = (ret_mean / ret_finite.std()).item() if ret_finite.std() > 1e-8 else 0.
            sharpe = ((ret_mean - 0.03/252) / ret_finite.std() * np.sqrt(252)).item() if ret_finite.std() > 1e-8 else 0.
            
            def invalid_to_zero(x):
                if not np.isfinite(x):
                    return 0.
                else:
                    return max(x,0.)
                    
            multi_score = {'ic':ic_mean,'icir':icir,'ret':ret_mean,'sharpe':sharpe,'retir':ret_ir}
            multi_score = {k:invalid_to_zero(v) for k,v in multi_score.items()}
            score = multi_score[metric_target]
            
            #  too many nan
            if torch.isfinite(fct[0]).sum()/torch.isfinite(tgt[0]).sum() <0.8:
                score = 0.
            
            # unique ratio too small
            elif len(torch.unique(fct[0])) / len(torch.unique(tgt[0])) <0.01:
                score = 0.
            
            if n_blds > 0 and score > 0.:
                assert len(ret.shape) == 1 , f"{ret.shape},{n_days}"
                assert len(ret) == n_days , f"{ret.shape},{n_days}"

                all_matrix = torch.concatenate([existed,ret[None]],dim=0) # (n_blds+1,n_days)
                assert all_matrix.shape == (n_blds+1,n_days) , f"{all_matrix.shape}"

                corr_score = torch.corrcoef(all_matrix)[-1,:-1].abs().max().item()

                if corr_score > corr_thresh:
                    score = 0.

            return {'score':score,'ret':ret.detach().cpu().numpy(),'multi_score':multi_score}
            
        except Exception as e:
            print(f"Error in get_score: {e}")
            return {'score': 0., 'ret': np.zeros(tgt.shape[1] if len(tgt.shape) > 1 else tgt.shape[0]), 
                   'multi_score': {'ic': 0, 'icir': 0, 'ret': 0, 'sharpe': 0, 'retir': 0}}
    
    return get_score


def main(
        instruments: str = "csi500",
        train_end_year:int = 2020,
        freq:str = 'day',
        seeds:str = '[0]',
        cuda:int = 0,
        save_name:str = 'test',
        zoo_size:int = 100,
        corr_thresh:float = 0.7,  
        ic_thresh:float = 0.03,    
        icir_thresh:float = 0.1,  
):
    if isinstance(seeds,str):
        seeds = eval(seeds)
    assert isinstance(seeds,list)   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda)
    train_end = train_end_year
    returned = get_data_by_year(
        train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
        instruments=instruments, target=target,freq=freq,
    )
    data_all,data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned
    
    # 添加数据维度调试信息
    print(f"Instruments: {instruments}")
    print(f"实际股票列表长度: {len(data._stock_ids)}")
    print(f"股票列表前10个: {data._stock_ids[:10].tolist()}")
    print(f"数据形状详情: {data.data.shape}")
    print(f"数据维度: {data.data.shape}")
    print(f"时间范围: backtrack={data.max_backtrack_days}, future={data.max_future_days}")
    print(f"股票数量: {data.n_stocks}")
    print(f"时间长度: {data.n_days}")
    print(f"特征数量: {data.n_features}")

    
    for seed in seeds:
        reseed_everything(seed)

        # 修改：配置类增加新参数
        class cfg:
            name = f'{save_name}_{instruments}_{train_end}_{seed}'
            max_len = 20

            # 减小batch size以节省内存
            batch_size = 256
            potential_size = 100

            n_layers = 2
            d_model = 128
            dropout = 0.2
            num_factors = zoo_size

            # generator configuaration
            num_epochs_g = 200
            g_es_score = 'max'  # max mean std combined
            g_es = 10
            g_hidden = 128  # 从128减小
            g_lr = 1e-3

            # predictor configuration
            p_hidden = 128  # 从128减小
            p_lr = 1e-3
            es_p = 10
            batch_size_p = 64  # 从64减小
            num_epochs_p = 100
            data_keep_p = 20000

            f_corr_thresh = corr_thresh  # threshold to penalize the correlation
            f_add_thresh = corr_thresh  # threshold to add new exprs to the zoo
            f_score_thresh = ic_thresh  # threshold to filter exprs in the zoo
            f_multi_score_thresh = {'icir': icir_thresh}

            # loss configuaration
            l_pred = 1.
            l_simi = 10.
            l_simi_thresh = 0.4

            l_potential = 10.
            l_potential_thresh = 0.4
            l_potential_epsilon = 1e-7

            l_entropy = 0

            device = 'cuda:0'

            # 修改：新增的配置参数
            # IC损失权重
            ic_loss_weight = 0.1  # 初始权重较小

            # 动态权重调度
            use_dynamic_weights = True
            warmup_epochs = 20  # 预热期

            # 垃圾回收频率
            gc_frequency = 10  # 每10轮进行一次垃圾回收

            # 最大表达式生成数量
            max_expressions_per_batch = 1000





        print(f"seed:{seed},name:{cfg.name}")

        from gan.network.predictor import NetP
        from gan.network.generater import NetG_DCGAN
        NetG_CLS = NetG_DCGAN
        NetP_CLS = NetP

        def random_call(z):
            return z.normal_()

        netG = NetG_CLS(
            n_chars=SIZE_ACTION,
            latent_size=cfg.potential_size,
            seq_len=cfg.max_len,
            hidden = cfg.g_hidden,
            ).to(cfg.device)

        netM = NetM(max_len=cfg.max_len,size_action=SIZE_ACTION).to(cfg.device)

        netP = NetP_CLS(
            n_chars=SIZE_ACTION, seq_len=cfg.max_len,hidden = cfg.p_hidden,
            ).to(cfg.device)

        z = torch.zeros([cfg.batch_size,cfg.potential_size])
        z = z.to(cfg.device)
        random_call(z)

        # initialize the zoo
        zoo_blds = Builders(0,max_len=cfg.max_len,n_actions=SIZE_ACTION)
        metric = get_metric(zoo_blds,device=cfg.device,corr_thresh=cfg.f_corr_thresh)
        empty_metric = get_metric(
            Builders(0,max_len=cfg.max_len,n_actions=SIZE_ACTION),
            device=cfg.device,corr_thresh=cfg.f_corr_thresh
        )

        coll = Collector(seq_len=cfg.max_len,n_actions=SIZE_ACTION)
        coll.reset(data,target,metric)

        coll.collect_target_num(netG,netM,z,data,target,metric,
                                target_num=10000,reset_net=True,drop_invalid=False, 
                                randomly = False,
                                random_method = random_call,max_iter = 200)  

        # train and mine untill the zoo is full
        t = 0
        total_iterations = 50
        while len(zoo_blds)<cfg.num_factors:
            print(f"\n{'=' * 60}")
            print(f"=== Training iteration {t} | Zoo size: {len(zoo_blds)}/{cfg.num_factors} ===")
            print(f"{'=' * 60}")

            # 修改：动态调整IC损失权重
            if cfg.use_dynamic_weights:
                progress = min(t / cfg.warmup_epochs, 1.0)
                cfg.ic_loss_weight = 0.1 + 0.4 * progress  # 从0.1增加到0.5
                print(f"Dynamic IC loss weight: {cfg.ic_loss_weight:.3f}")
            
            if not zoo_blds.examined:
                print(' zoo_blds not examined')
                zoo_blds.evaluate(data,target,empty_metric,verbose=True)

            ### update the metric for the current zoo
            metric = get_metric(zoo_blds,device = cfg.device,corr_thresh=cfg.f_corr_thresh)

            ### Prepare data to train predictor
            coll.blds.evaluate(data,target,metric,verbose=True)
            if coll.blds_bak.batch_size>cfg.data_keep_p:
                # sample the training data of predictor to keep the size
                print(f'sample datas to keep {coll.blds_bak.batch_size} to {cfg.data_keep_p}')
                indices = np.random.choice(np.arange(coll.blds_bak.batch_size),cfg.data_keep_p,replace=False)
                coll.blds_bak = coll.blds_bak.filter_by_index(indices)
            coll.blds_bak.evaluate(data,target,metric,verbose=True)

            if coll.blds_bak.batch_size > 0:
                # give the current builders more weights in training p
                blds_list = [coll.blds_bak,coll.blds]
                weight_list = [1.,2.]
            else:
                blds_list = [coll.blds]
                weight_list = [1.]



            # 检查是否有有效的builders
            if len(blds_list) == 0 or all(len(blds.builders_tokens) == 0 for blds in blds_list):
                print("No valid builders found, skipping this iteration")
                t += 1
                continue

            x, y, weights = blds_list_to_tensor(blds_list,weight_list)
            y = pre_process_y(y)

            ### 修改：训练预测器时传入数据和目标
            print("\n=== Training Predictor ===")
            netP.initialize_parameters()
            train_net_p_with_weight(cfg, netP, x, y, weights, data, target, lr=cfg.p_lr)

            ### 修改：使用改进的生成器训练函数
            print("\n=== Training Generator ===")
            netG.initialize_parameters()
            blds_in_train = train_network_generator_improved(
                netG, netM, netP, cfg, data, target, t,
                random_method=random_call, metric=metric, lr=cfg.g_lr,
                n_actions=SIZE_ACTION, total_iterations=total_iterations
            )


            
            ### Generate new alpha factors from current Generator
            print("\n=== Generating New Factors ===")
            coll.reset(data,target,metric)
            # 减少生成的表达式数量
            coll.collect_target_num(netG,netM,z,data,target,metric,
                                    target_num=cfg.max_expressions_per_batch,
                                    reset_net=False,drop_invalid=False,
                                    randomly = False,
                                    random_method = random_call,max_iter = 100) 

            lengh_s = {"train":len(blds_in_train)}
            lengh_s['new']=len(coll.blds)
            coll.blds = coll.blds + blds_in_train
            coll.blds.drop_duplicated()
            lengh_s['all_new']=len(coll.blds)

            print(f"\nGeneration summary: {lengh_s['train']} (train) + {lengh_s['new']} (new)  =  {lengh_s['all_new']} (all_new)")
            
            ### get the valid alpha factors during the training process and the generating process
            print("\n=== Filtering Valid Factors ===")
            new_zoo = filter_valid_blds(
                coll.blds,
                corr_thresh=cfg.f_add_thresh,
                score_thresh=cfg.f_score_thresh,
                multi_score_thresh = cfg.f_multi_score_thresh,
                device = cfg.device,
                verbose= True,
                )
            lengh_s['zoo_prev'] = len(zoo_blds)
            zoo_blds = zoo_blds + new_zoo

            print(f" zoo_prev:{lengh_s['zoo_prev']}, new_valid:{len(new_zoo)}, current:{len(zoo_blds)}")
            zoo_blds.evaluate(data,target,empty_metric,verbose=True)

            # 修改：定期重新平衡zoo，去除相关性过高的因子
            if t % 5 == 2:
                print('\n' + '#' * 20 + " Zoo Rebalancing " + '#' * 20)
                prev_size = len(zoo_blds)
                zoo_blds = filter_valid_blds(
                    zoo_blds,
                    corr_thresh=cfg.f_add_thresh,
                    score_thresh=cfg.f_score_thresh,
                    multi_score_thresh=cfg.f_multi_score_thresh,
                    device=cfg.device,
                    verbose=False,
                )
                print(f"Zoo rebalanced: {prev_size} -> {len(zoo_blds)}")


            # save the zoo
            save_blds(zoo_blds,f"out/{cfg.name}",'zoo_final')

            # 修改：随机生成一些因子以促进探索
            print("\n=== Random Exploration ===")
            random_target = min(500, cfg.max_expressions_per_batch // 2)
            coll.collect_target_num(netG, netM, z, data, target, metric,
                                    target_num=random_target,
                                    reset_net=False, drop_invalid=False,
                                    randomly=True,
                                    random_method=random_call, max_iter=50)

            # 修改：定期清理内存
            if t % cfg.gc_frequency == 0:
                print("\nPerforming periodic memory cleanup...")
                del x, y, weights
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # 打印内存使用情况
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024 ** 3
                    reserved = torch.cuda.memory_reserved() / 1024 ** 3
                    print(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

            t += 1

            # 修改：添加最大迭代限制，防止无限循环
            if t > total_iterations:
                print(f"\nReached maximum iterations ({total_iterations}), stopping...")
                break

        # 修改：最终评估和保存
        print(f"\n{'=' * 60}")
        print(f"=== Training Complete! Final Zoo Size: {len(zoo_blds)} ===")
        print(f"{'=' * 60}")

        empty_blds = Builders(0,max_len=cfg.max_len,n_actions=SIZE_ACTION)
        metric = get_metric(empty_blds,device = cfg.device,corr_thresh=cfg.f_corr_thresh)
        zoo_blds.evaluate(data,target,metric,verbose=True)
        save_blds(zoo_blds,f"out/{cfg.name}",'zoo_final')

        # 打印最终统计信息
        if len(zoo_blds.scores) > 0:
            scores = np.array(zoo_blds.scores)
            print(f"\nFinal Zoo Statistics:")
            print(f"  - Number of factors: {len(zoo_blds)}")
            print(f"  - Average score: {np.mean(scores):.4f}")
            print(f"  - Max score: {np.max(scores):.4f}")
            print(f"  - Min score: {np.min(scores):.4f}")
            print(f"  - Std score: {np.std(scores):.4f}")

            # 打印top 10因子
            top_indices = np.argsort(scores)[-10:][::-1]
            print(f"\nTop 10 Factors:")
            for i, idx in enumerate(top_indices):
                print(f"  {i + 1}. Score: {scores[idx]:.4f}, Expr: {zoo_blds.exprs_str[idx]}")

        # 最终保存
        save_blds(zoo_blds, f"out/{cfg.name}", 'zoo_final')
        print(f"\nResults saved to: out/{cfg.name}/zoo_final")

if __name__ == '__main__':
    import fire
    fire.Fire(main)