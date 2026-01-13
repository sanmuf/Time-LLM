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

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

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
parser.add_argument('--train_path', type=str, default='train_fail.csv', help='train data file')
parser.add_argument('--val_path', type=str, default='val_fail.csv', help='val data file')
parser.add_argument('--test_path', type=str, default='test_fail.csv', help='test data file')
parser.add_argument('--train_json_path', type=str, default='train_fail.json', help='train json file')
parser.add_argument('--val_json_path', type=str, default='val_fail.json', help='val json file')
parser.add_argument('--test_json_path', type=str, default='test_fail.json', help='test json file')
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
parser.add_argument('--loss', type=str, default='MSE', help='loss function (MSE/BCE/FOCAL)')
parser.add_argument('--focal_alpha', type=float, default=0.25, help='alpha for focal loss')
parser.add_argument('--focal_gamma', type=float, default=3.0, help='gamma for focal loss')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--use_oversample', action='store_true', help='Use oversampling for training')
parser.add_argument('--val_split', type=float, default=0.1, help='val split ratio when no val set provided')
parser.add_argument('--nan_debug', action='store_true', help='print NaN/Inf debug info and skip bad batches')


args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deep_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deep_plugin)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.40, gamma=4.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1-pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    
def hard_negative_mining(losses, labels, ratio=0.5):
    neg_indices = (labels == 0).nonzero(as_tuple=True)[0]
    neg_losses = losses[neg_indices]
    k = int(len(neg_losses) * ratio)
    if k == 0:
        return neg_indices
    topk = torch.topk(neg_losses, k=k).indices
    return neg_indices[topk]

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction: str = "mean"):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg  # 负样本的衰减力度，建议设大一点 (2~4)
        self.gamma_pos = gamma_pos  # 正样本的衰减力度，建议设小 (0~1)
        self.clip = clip            # 概率平移阈值，小于这个概率的负样本梯度归零
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """"
        x: input logits (没有经过 sigmoid 的输出)
        y: targets (0 或 1)
        """
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid_pos = x_sigmoid
        x_sigmoid_neg = 1 - x_sigmoid

        # 正样本/负样本的概率
        pt0 = x_sigmoid_pos * y
        pt1 = x_sigmoid_neg * (1 - y)  # pt = p if y=1 else 1-p

        # 负样本概率平移
        x_sigmoid_neg_shifted = (x_sigmoid_neg + self.clip).clamp(max=1)
        pt1_shifted = x_sigmoid_neg_shifted * (1 - y)
        pt_shifted = pt0 + pt1_shifted  # 只有负样本做了 shift

        # 不对称的权重
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt_shifted, one_sided_gamma)

        loss = -one_sided_w * torch.log(pt_shifted + self.eps)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none"

# ------------------- 二分类指标工具 -------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

def _bin_metrics_from_logits(logits: np.ndarray, labels: np.ndarray, thr: float = 0.5):
    valid_mask = np.isfinite(logits) & np.isfinite(labels)
    if not np.all(valid_mask):
        invalid_count = int((~valid_mask).sum())
        print(f"warning: filtered {invalid_count} invalid logits/labels")
    logits = logits[valid_mask]
    labels = labels[valid_mask]
    if logits.size == 0:
        print("probs mean: nan (no valid logits)")
        return 0.0, 0.0, float('nan'), 0.0, 0.0
    probs = _sigmoid(logits)
    print("probs mean:", float(np.mean(probs)))
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
    return acc, f1, auc,rec,prec

def _bin_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(probs[mask]))
        bin_acc = float(np.mean(labels[mask]))
        ece += (np.sum(mask) / len(probs)) * abs(bin_acc - bin_conf)
    return float(ece)

def find_best_threshold(logits: np.ndarray, labels: np.ndarray):
    best_min_pr = -1.0
    best_f1, best_thr, best_acc, best_auc, best_rec, best_prec = 0.0, 0.5, 0.0, float('nan'), 0.0, 0.0
    for t in np.arange(0.1, 0.9 + 0.001, 0.01):
        acc, f1, auc, rec, prec = _bin_metrics_from_logits(logits, labels, thr=t)
        min_pr = min(rec, prec)
        if (min_pr > best_min_pr) or (min_pr == best_min_pr and f1 > best_f1):
            best_min_pr = min_pr
            best_f1, best_thr, best_acc, best_auc, best_rec, best_prec = f1, t, acc, auc, rec, prec
    return best_acc, best_f1, best_auc, best_rec, best_prec, best_thr

@torch.no_grad()
def eval_binary_cls(args, accelerator, model, data_loader, search_thr: bool = True, fixed_thr: float = 0.5):
    model.eval()
    all_logits, all_labels = [], []
    for batch_x, batch_y, batch_x_mark, batch_y_mark,batch_token_ids,batch_feat,_ in data_loader:
    #for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
        batch_x = batch_x.float().to(accelerator.device)
        batch_y = batch_y.float().to(accelerator.device)
        batch_x_mark = batch_x_mark.float().to(accelerator.device)
        batch_y_mark = batch_y_mark.float().to(accelerator.device)
        batch_feat = batch_feat.float().to(accelerator.device)
        print("batch_feat shape:", batch_feat.shape)
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_token_ids,batch_feat) 
       # outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if args.output_attention:
            outputs = outputs[0]
        f_dim = -1 
        logits = outputs[:, -args.pred_len:, f_dim:]
        labels = batch_y[:, -args.pred_len:, f_dim:]
        all_logits.append(logits.detach().reshape(-1).cpu())
        all_labels.append(labels.detach().reshape(-1).cpu())
    if not all_logits:
        return 0.0, 0.0, float('nan'), 0.0, 0.0, float('nan'), float('nan')
    logits_np = torch.cat(all_logits).float().cpu().numpy()
    labels_np = torch.cat(all_labels).float().cpu().numpy()
    probs = _sigmoid(logits_np)
    if search_thr:
        acc, f1, auc, recall, prec, thr = find_best_threshold(logits_np, labels_np)
    else:
        acc, f1, auc, recall, prec = _bin_metrics_from_logits(logits_np, labels_np, thr=fixed_thr)
        thr = fixed_thr
    ece = _bin_ece(probs, labels_np, n_bins=15)
    print(f"[Eval] Best Thr={thr:.2f}, Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f},Recall={recall:.4f},Prec={prec:.4f},ECE={ece:.4f}")
    return acc, f1, auc, recall, prec, ece, thr

# ------------------- 主训练循环 -------------------
for ii in range(args.itr):
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name, args.model_id, args.model, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.d_layers,
        args.d_ff, args.factor, args.embed, args.des, ii)

    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    pos_count, neg_count = 0, 0
    for _, batch_y, _, _,_,_,_ in train_loader:
        batch_y = batch_y.float()
        batch_y_last = batch_y[:, -args.pred_len:, -1]  
        pos_count += batch_y_last.sum().item()
        neg_count += (batch_y_last.numel() - batch_y_last.sum().item())
        print(f"pos_count: {pos_count}, neg_count: {neg_count}, ratio: {pos_count/(pos_count+neg_count):.4f}")

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

    if args.loss.upper() in ['BCE', 'BCEWITHLOGITSLOSS']:
        # criterion = FocalLoss(alpha=0.25, gamma=3.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(accelerator.device))
        # criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        is_binary_cls = True
    elif args.loss.upper() in ['FOCAL', 'FOCALLOSS']:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        is_binary_cls = True
    else:
        criterion = nn.MSELoss()
        is_binary_cls = False

    if is_binary_cls:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            model_optim, mode='max', factor=0.5, patience=2, threshold=1e-4
        )
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            model_optim, mode='min', factor=0.5, patience=2, threshold=1e-4
        )

    mae_metric = nn.L1Loss()
    train_loader, val_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, val_loader, test_loader, model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # ---- 初始化表格存储指标 ----
    results = []

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        model.train()
        epoch_time = time.time()

        all_losses=torch.zeros(len(train_data))
        all_labels=torch.zeros(len(train_data),dtype=torch.long)

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_text,batch_feat,batch_idx) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_feat = batch_feat.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            if i == 0:
                print("batch_x.shape:", batch_x.shape)
                print("batch_y.shape:", batch_y.shape)
                print("batch_feat.shape:", batch_feat.shape)
                print("batch_x example:", batch_x[0, :5, :3])
                print("batch_y sample:", batch_y[0, :5, :3])
                print("batch_text example:", batch_text[0][:200])

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_text,batch_feat)[0] if args.output_attention else \
                              model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_text,batch_feat)
                    f_dim = -1 if args.features in ['MS','S'] else 0 
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y_slice = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = criterion(outputs, batch_y_slice)
                    train_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_text,batch_feat)[0] if args.output_attention else \
                          model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_text,batch_feat)
                f_dim = -1 if args.features in ['MS','S'] else 0 
                
                if i == 0 :
                    print("outputs shape:", outputs.shape) 
                    print("outputs example:", outputs[0, :5])
                outputs = outputs[:, -args.pred_len:, f_dim:]
                if i == 0 :
                    print("outputs slice shape:", outputs.shape)
                    print("outputs slice example:", outputs[0, :5])

                batch_y_slice = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y_slice)

                train_loss.append(loss.item())
                accelerator.backward(loss)
                model_optim.step()
               # all_losses[batch_idx]=loss.detach().cpu()
               # all_labels[batch_idx]=batch_y_slice.detach().cpu().long().view(-1)
            
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

       # hard_neg_idx=hard_negative_mining(all_losses,all_labels,ratio=0.25)

        #pos_idx = (all_labels == 1).nonzero(as_tuple=True)[0]
        #selected_idx = torch.cat([pos_idx, hard_neg_idx])
        #_,train_loader = data_provider(args,"train",indices=selected_idx)

        #print(f"Selected {len(hard_neg_idx)} hard negatives for next epoch.")

        train_loss_avg = np.average(train_loss)
        #vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        #test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        
        if is_binary_cls:
            val_acc, val_f1, val_auc, val_rec, val_prec, val_ece, val_thr = eval_binary_cls(
                args, accelerator, model, val_loader, search_thr=True
            )
            test_acc, test_f1, test_auc, test_rec, test_prec, test_ece, _ = eval_binary_cls(
                args, accelerator, model, test_loader, search_thr=False, fixed_thr=val_thr
            )
            accelerator.print(
                f"Train Loss: {train_loss_avg:.7f} "
                f"VAL[ACC {val_acc:.4f} F1 {val_f1:.4f} AUC {val_auc:.4f} Recall {val_rec:.4f} Prec {val_prec:.4f} ECE {val_ece:.4f}] "
                f"TEST[ACC {test_acc:.4f} F1 {test_f1:.4f} AUC {test_auc:.4f} Recall {test_rec:.4f} Prec {test_prec:.4f} ECE {test_ece:.4f}]"
            )
            results.append([
                train_loss_avg,
                val_acc, val_f1, val_auc, val_rec, val_prec, val_ece,
                test_acc, test_f1, test_auc, test_rec, test_prec, test_ece
            ])
        else:
            results.append([train_loss_avg] + [None] * 12)

        if is_binary_cls:
            early_stopping(val_f1, model, path)
        else:
            early_stopping(train_loss_avg, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if is_binary_cls:
            scheduler.step(val_f1)
        else:
            scheduler.step(train_loss_avg)
        accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))

    # ---- 保存 CSV ----

        
    if accelerator.is_local_main_process:
        os.makedirs("./metrics", exist_ok=True)
        results_df = pd.DataFrame(results, columns=[
            'train_loss',
            'val_acc', 'val_f1', 'val_auc', 'val_rec', 'val_prec', 'val_ece',
            'test_acc', 'test_f1', 'test_auc', 'test_rec', 'test_prec', 'test_ece'
        ])
        csv_path = os.path.join("./metrics", f"{args.model_id}_metrics.csv")
        results_df.to_csv(csv_path, index=False)
        accelerator.print(f"Metrics saved to {csv_path}")

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    del_files('./checkpoints')
    accelerator.print('success delete checkpoints')
