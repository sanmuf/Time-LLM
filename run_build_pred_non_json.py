import argparse
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import pandas as pd

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_non_json, load_content

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# ------------------- 基础配置 -------------------
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear, TimeLLM]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# ------------------- 数据加载 -------------------
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M: multivariate predict multivariate, S: univariate predict univariate, MS: multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# ------------------- 预测任务 -------------------
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# ------------------- 模型 -------------------
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')
parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension')

# ------------------- 优化器 -------------------
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function (MSE or BCE)')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--use_oversample', action='store_true', help='Use oversampling for training')


args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deep_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deep_plugin)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        
        self.gamma_neg = gamma_neg # 负样本的衰减力度，建议设大一点 (2~4)
        self.gamma_pos = gamma_pos # 正样本的衰减力度，建议设小 (0~1)
        self.clip = clip           # 概率平移阈值，小于这个概率的负样本梯度归零
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        x: input logits (没有经过 sigmoid 的输出)
        y: targets (0 或 1)
        """
        # 计算 Sigmoid 概率
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid_pos = x_sigmoid
        x_sigmoid_neg = 1 - x_sigmoid

        # --- 处理正样本 (Positive Samples) ---
        # 计算 pt (probability of target)
        pt0 = x_sigmoid_pos * y
        pt1 = x_sigmoid_neg * (1 - y)  # pt = p if y=1 else 1-p
        pt = pt0 + pt1

        # --- 处理负样本的 Shift (Probability Shifting) ---
        # 核心逻辑：如果负样本预测得很好，直接忽略它
        x_sigmoid_neg_shifted = (x_sigmoid_neg + self.clip).clamp(max=1) 
        pt1_shifted = x_sigmoid_neg_shifted * (1 - y)
        pt_shifted = pt0 + pt1_shifted # 只有负样本做了 shift

        # --- 计算不对称的权重 ---
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt_shifted, one_sided_gamma)

        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)
        
        # --- 计算 Loss ---
        loss = -one_sided_w * torch.log(pt_shifted + self.eps)

        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)
            
        return loss.mean()

# ------------------- 二分类指标工具 -------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _bin_metrics_from_logits(logits: np.ndarray, labels: np.ndarray, thr: float = 0.5):
    probs = _sigmoid(logits)
    preds = (probs >= thr).astype(np.int64)
    y = labels.astype(np.int64)
    acc = float((preds == y).mean()) if y.size else 0.0
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    order = np.argsort(-probs)
    y_sorted = y[order]
    P = float((y_sorted == 1).sum())
    N = float((y_sorted == 0).sum())
    if P == 0 or N == 0:
        auc = float('nan')
    else:
        tps = np.cumsum(y_sorted == 1)
        fps = np.cumsum(y_sorted == 0)
        tpr = tps / P
        fpr = fps / N
        auc = float(np.trapz(tpr, fpr))
    return acc, f1, auc

def find_best_f1(logits: np.ndarray, labels: np.ndarray):
    best_f1, best_thr, best_acc, best_auc = 0.0, 0.5, 0.0, float('nan')
    for t in np.arange(0.1, 0.9 + 0.001, 0.01): 
        acc, f1, auc = _bin_metrics_from_logits(logits, labels, thr=t)
        if f1 > best_f1:
            best_f1, best_thr, best_acc, best_auc = f1, t, acc, auc
    return best_acc, best_f1, best_auc, best_thr

@torch.no_grad()
def eval_binary_cls(args, accelerator, model, data_loader):
    model.eval()
    all_logits, all_labels = [], []
    #for batch_x, batch_y, batch_x_mark, batch_y_mark,batch_token_ids,batch_attention_mask in data_loader:
    for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
        batch_x = batch_x.float().to(accelerator.device)
        batch_y = batch_y.float().to(accelerator.device)
        batch_x_mark = batch_x_mark.float().to(accelerator.device)
        batch_y_mark = batch_y_mark.float().to(accelerator.device)
    #    batch_token_ids=batch_token_ids.to(accelerator.device)
    #    batch_attention_mask=batch_attention_mask.to(accelerator.device)
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)
    #    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_token_ids,batch_attention_mask)
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if args.output_attention:
            outputs = outputs[0]
        f_dim = -1 
        logits = outputs[:, -args.pred_len:, f_dim:]
        labels = batch_y[:, -args.pred_len:, f_dim:]
        all_logits.append(logits.detach().reshape(-1).cpu())
        all_labels.append(labels.detach().reshape(-1).cpu())
    if not all_logits:
        return 0.0, 0.0, float('nan')
    logits_np = torch.cat(all_logits).float().cpu().numpy()
    labels_np = torch.cat(all_labels).float().cpu().numpy()
    acc, f1, auc, thr = find_best_f1(logits_np, labels_np)
    print(f"[Eval] Best Thr={thr:.2f}, Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    return acc, f1, auc

# ------------------- 主训练循环 -------------------
for ii in range(args.itr):
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name, args.model_id, args.model, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.d_layers,
        args.d_ff, args.factor, args.embed, args.des, ii)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    pos_count, neg_count = 0, 0
    for _, batch_y, _, _,_,_ in train_loader:
       batch_y = batch_y.float()
        # 只统计预测位置的标签
       batch_y_last = batch_y[:, -args.pred_len:, -1]  
       pos_count += batch_y_last.sum().item()
       neg_count += (batch_y_last.numel() - batch_y_last.sum().item())

    pos_weight = torch.tensor([neg_count / (pos_count + 1e-8)], dtype=torch.float32)

    args.content = load_content(args)

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    if args.loss.upper() in ['BCE', 'BCEWITHLOGITSLOSS']:
        # criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight.to(accelerator.device))
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(accelerator.device))
        # criterion = nn.BCEWithLogitsLoss()
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        is_binary_cls = True
    else:
        criterion = nn.MSELoss()
        is_binary_cls = False

    mae_metric = nn.L1Loss()
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # ---- 初始化表格存储指标 ----
    results = []

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        model.train()
        epoch_time = time.time()

       # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_token_ids,batch_attention_mask) in tqdm(enumerate(train_loader)):
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
        #    batch_token_ids=batch_token_ids.to(accelerator.device)
        #    batch_attention_mask=batch_attention_mask.to(accelerator.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    #outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_token_ids,batch_attention_mask)[0] if args.output_attention else \
                    #          model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_token_ids,batch_attention_mask)
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if args.output_attention else \
                              model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if args.features in ['MS','S'] else 0 
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y_slice = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = criterion(outputs, batch_y_slice)
                    train_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
               # outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_token_ids,batch_attention_mask)[0] if args.output_attention else \
               #           model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_token_ids,batch_attention_mask)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if args.output_attention else \
                          model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if args.features in ['MS','S'] else 0  
                outputs = outputs[:, -args.pred_len:, f_dim:]

                batch_y_slice = batch_y[:, -args.pred_len:, f_dim:]

                # loss = criterion(outputs, batch_y_slice)
                last_status = batch_x[:, -1, -1].to(accelerator.device)

                # 2. 获取“当前时刻”的真实状态 (Target)
                # batch_y_slice shape: [Batch, Pred_Len, 1] -> 变成 [Batch]
                current_status = batch_y_slice.squeeze(-1).squeeze(-1).to(accelerator.device)

                # 3. 找出“状态突变”的样本 (0->1 或 1->0)
                # 如果相等，diff 为 0；如果不等，diff 为 1
                is_change = (last_status != current_status).float()

                # 4. 设置权重
                # 没变的样本权重 = 1.0
                # 突变的样本权重 = 1.0 + 15.0 (或者更高，比如 20.0)
                # 这个系数 15.0 可以调节，越大说明你越想让模型关注突变
                sample_weights = 1.0 + 10.0 * is_change 

                # 5. 计算逐样本的 Loss (不求平均，reduction='none')
                # 注意：这里需要手动调用 functional 的 BCE Loss
                # per_sample_loss = F.binary_cross_entropy_with_logits(
                per_sample_loss = criterion(
                    outputs.squeeze(-1),       # [Batch, 1] -> [Batch]
                    batch_y_slice.squeeze(-1), # [Batch, 1] -> [Batch]
                    reduction='none'           # 关键：保留每个样本的 Loss
                )

                # 6. 加权并求平均
                loss = (per_sample_loss * sample_weights).mean()


                train_loss.append(loss.item())
                accelerator.backward(loss)
                model_optim.step()
            
          #  scheduler.step()

            if (i + 1) % 100 == 0:
                accelerator.print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                speed = (time.time() - epoch_time) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                iter_count = 0
                epoch_time = time.time()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()
    
        train_loss_avg = np.average(train_loss)
        vali_loss, vali_mae_loss = vali_non_json(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        test_loss, test_mae_loss = vali_non_json(args, accelerator, model, test_data, test_loader, criterion, mae_metric)

        if is_binary_cls:
            val_acc, val_f1, val_auc = eval_binary_cls(args, accelerator, model, vali_loader)
            test_acc, test_f1, test_auc = eval_binary_cls(args, accelerator, model, test_loader)
            accelerator.print(f"Epoch: {epoch+1} | Train Loss: {train_loss_avg:.7f} "
                              f"Val Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f} | "
                              f"VAL[ACC {val_acc:.4f} F1 {val_f1:.4f} AUC {val_auc:.4f}] "
                              f"TEST[ACC {test_acc:.4f} F1 {test_f1:.4f} AUC {test_auc:.4f}]")
            results.append([epoch+1, train_loss_avg, vali_loss, test_loss,
                            val_acc, val_f1, val_auc, test_acc, test_f1, test_auc])
        else:
            results.append([epoch+1, train_loss_avg, vali_loss, test_loss,
                            None, None, None, None, None, None])

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    # ---- 保存 CSV ----
    if accelerator.is_local_main_process:
        os.makedirs("./metrics", exist_ok=True)
        results_df = pd.DataFrame(results, columns=['epoch','train_loss','vali_loss','test_loss',
                                                    'val_acc','val_f1','val_auc',
                                                    'test_acc','test_f1','test_auc'])
        csv_path = os.path.join("./metrics", f"{args.model_id}_metrics.csv")
        results_df.to_csv(csv_path, index=False)
        accelerator.print(f"Metrics saved to {csv_path}")

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    del_files('./checkpoints')
    accelerator.print('success delete checkpoints')
