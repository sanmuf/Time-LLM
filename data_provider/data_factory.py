from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4,Dataset_Build_pred,Dataset_Data_pred
from torch.utils.data import DataLoader,WeightedRandomSampler,Subset
import numpy as np
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'build': Dataset_Build_pred,
    'build_data':Dataset_Data_pred,
    'm4': Dataset_M4,
}


def data_provider(args, flag,indices=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    elif args.data == 'build_data':
        data_set = Data(
            root_path=args.root_path,
            train_path=args.train_path,
            test_path=args.test_path,
            val_path=args.val_path,
            train_json_path=args.train_json_path,
            test_json_path=args.test_json_path,
            val_json_path=args.val_json_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns,
            val_split=getattr(args, "val_split", 0.1)
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )

    if indices is not None:
        data_set=Subset(data_set,indices)

    if flag == 'train' and getattr(args, "use_oversample", False):
        tot_windows = len(data_set) 

        if tot_windows <= 0:
            raise ValueError("Dataset length is zero — 检查 seq_len/pred_len 与数据长度的关系。")
        row_indices = np.arange(tot_windows) + data_set.seq_len + data_set.pred_len - 1
        max_row = data_set.data_y.shape[0] - 1
        if row_indices[-1] > max_row or row_indices[0] < 0:
            raise IndexError(f"Computed row_indices out of range: 0..{max_row}, got {row_indices[0]}..{row_indices[-1]}")
        col_idx = -1
        window_labels = data_set.data_y[row_indices, col_idx].astype(np.float32)
        window_labels = (window_labels != 0).astype(np.int64)
        class_weights = np.array([0.8, 0.2])
        sample_weights = class_weights[window_labels] 

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

    return data_set, data_loader
