import numpy as np
import pandas as pd
import os
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


def FFT_for_Period(x, k=5, min_period=10, zero_pad_factor=8, use_window=True, max_retry=10,
                   fixed_topk=500, adaptive_threshold=True):
    """
    Args:
        adaptive_threshold: whether to enable adaptive threshold filtering (based on 1/k)
    """
    B, T, C = x.shape
    n_fft = T * zero_pad_factor

    # Apply smoothing window to reduce spectral leakage
    if use_window:
        window = torch.hann_window(T, device=x.device)
        x = x * window[None, :, None]

    xf = torch.fft.rfft(x, n=n_fft, dim=1)
    amplitude = torch.abs(xf).mean(dim=0).mean(dim=-1)
    amplitude[:max(int(T * 0.001), 2)] = 0  # Remove DC component

    # Initial parameters
    max_period_ratio = 0.03
    attempt = 0

    while attempt < max_retry:
        # Take more frequency components
        actual_k = min(fixed_topk, len(amplitude) - 1)
        topk_values, topk_indices = torch.topk(amplitude, actual_k)
        periods = (n_fft / topk_indices.float()).cpu().numpy()
        weights = topk_values.detach().cpu()

        # Filter too small or too large periods
        mask = (periods >= min_period) & (periods <= T * max_period_ratio)
        periods, weights = periods[mask], weights[mask]

        # Deduplicate and merge close periods
        periods, weights = deduplicate_periods(periods, weights)

        # Step 1: pick top-k
        if len(periods) >= k:
            periods_topk = periods[:k]
            weights_topk = weights[:k]

            # Step 2: within top-k, filter based on normalized weights
            if adaptive_threshold:
                # Normalize weights
                weights_norm = weights_topk / (weights_topk.sum() + 1e-8)

                # Threshold = 1/k
                threshold = 1.0 / k

                # Select periods with weight above threshold
                mask_threshold = weights_norm >= threshold
                n_selected = mask_threshold.sum().item()

                # Boundary constraint: at least 1, at most k/2
                n_selected = max(1, min(n_selected, k//2))

                # Final selection
                periods_final = periods_topk[:n_selected]
                weights_final = weights_topk[:n_selected]

                # Re-normalize
                weights_final = weights_final / (weights_final.sum() + 1e-8)

                return np.round(periods_final).astype(int), weights_final

            else:
                # Without adaptive filtering, directly return top-k
                weights_topk = weights_topk / (weights_topk.sum() + 1e-8)
                return np.round(periods_topk).astype(int), weights_topk

        # Relax conditions
        fixed_topk = fixed_topk * 2
        max_period_ratio = min(max_period_ratio * 1.5, 0.1)
        attempt += 1

    # If still insufficient k periods, return whatever found
    if len(periods) == 0:
        print(f"[Warning] No valid periods detected in {T} timesteps, using default period")
        return np.array([min(T, 24 if T > 100 else T // 4)]), torch.tensor([1.0])

    # Normalize weights
    weights = weights / (weights.sum() + 1e-8)
    return np.round(periods).astype(int), weights


def deduplicate_periods(periods, weights, tolerance=0.05):
    """
    Deduplication: merge close periods (relative difference < tolerance)
    """
    if len(periods) == 0:
        return np.array([]), torch.tensor([])

    final_periods = []
    final_weights = []

    # Sort by weight descending
    for p, w in sorted(zip(periods, weights), key=lambda x: x[1], reverse=True):
        # Check whether similar to existing periods
        is_duplicate = False
        for fp in final_periods:
            if abs(p - fp) / max(fp, 1e-6) < tolerance:
                is_duplicate = True
                break

        if not is_duplicate:
            final_periods.append(p)
            final_weights.append(w)

    return np.array(final_periods), torch.tensor(final_weights)





class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None, top_k=1, device='cpu', percent=100):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        self.top_k = top_k
        self.device = device
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # add cycle
        if self.cycle == 0:
            border_a = border1s[0]
            border_b = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len
            x_tensor = torch.tensor(data[border_a:border_b], dtype=torch.float32).unsqueeze(0).to(self.device)
            period, weight = FFT_for_Period(x_tensor, self.top_k)
            # print(f"[Info] train len: {-border_a+border_b+1}")
            # print(f"[Info] Auto detected cycle: {period}")
            # print(f"[Info] weight: {weight}")

            self.cycle_index_list = [
                (np.arange(len(data)) % c)[border1:border2]
                for c in period
            ]
            self.cycle_inf = [period]
            self.cycle_inf.append(weight)

        else:
            self.cycle_index_list = [(np.arange(len(data)) % self.cycle)[border1:border2]]
            self.cycle_inf = [[self.cycle]]
            self.cycle_inf.append(torch.tensor(1.0))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor([ci[s_end] for ci in self.cycle_index_list], dtype=torch.long)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def get_cycle(self):
        return self.cycle_inf

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', cycle=None, top_k=1, device='cpu',percent =100):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        self.top_k = top_k
        self.device = device
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # add cycle
        if self.cycle == 0:
            border_a = border1s[0]
            border_b = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len
            x_tensor = torch.tensor(data[border_a:border_b], dtype=torch.float32).unsqueeze(0).to(self.device)
            period, weight = FFT_for_Period(x_tensor, self.top_k)
            # print(f"[Info] train len: {-border_a+border_b+1}")
            # print(f"[Info] Auto detected cycle: {period}")
            # print(f"[Info] weight: {weight}")

            self.cycle_index_list = [
                (np.arange(len(data)) % c)[border1:border2]
                for c in period
            ]
            self.cycle_inf = [period]
            self.cycle_inf.append(weight)

        else:
            self.cycle_index_list = [(np.arange(len(data)) % self.cycle)[border1:border2]]
            self.cycle_inf = [[self.cycle]]
            self.cycle_inf.append(torch.tensor(1.0))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor([ci[s_end] for ci in self.cycle_index_list], dtype=torch.long)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def get_cycle(self):
        return self.cycle_inf

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None, top_k=1, device='cpu',percent=100):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        self.top_k = top_k
        self.device = device
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # add cycle
        if self.cycle == 0:
            border_a = border1s[0]
            border_b = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len
            x_tensor = torch.tensor(data[border_a:border_b], dtype=torch.float32).unsqueeze(0).to(self.device)
            period, weight = FFT_for_Period(x_tensor, self.top_k)
            # print(f"[Info] train len: {-border_a+border_b+1}")
            # print(f"[Info] Auto detected cycle: {period}")
            # print(f"[Info] weight: {weight}")

            self.cycle_index_list = [
                (np.arange(len(data)) % c)[border1:border2]
                for c in period
            ]
            self.cycle_inf = [period]
            self.cycle_inf.append(weight)

        else:
            self.cycle_index_list = [(np.arange(len(data)) % self.cycle)[border1:border2]]
            self.cycle_inf = [[self.cycle]]
            self.cycle_inf.append(torch.tensor(1.0))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor([ci[s_end] for ci in self.cycle_index_list], dtype=torch.long)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def get_cycle(self):
        return self.cycle_inf

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, cycle=None, top_k=1, device='cpu',percent=100):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        self.top_k = top_k
        self.device = device
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]



        # add cycle
        if self.cycle == 0:
            border_a = border1s[0]
            border_b = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len
            x_tensor = torch.tensor(data[border_a:border_b], dtype=torch.float32).unsqueeze(0).to(self.device)
            period, weight = FFT_for_Period(x_tensor, self.top_k)
            # print(f"[Info] train len: {-border_a+border_b+1}")
            # print(f"[Info] Auto detected cycle: {period}")
            # print(f"[Info] weight: {weight}")

            self.cycle_index_list = [
                (np.arange(len(data)) % c)[border1:border2]
                for c in period
            ]
            self.cycle_inf = [period]
            self.cycle_inf.append(weight)

        else:
            self.cycle_index_list = [(np.arange(len(data)) % self.cycle)[border1:border2]]
            self.cycle_inf = [[self.cycle]]
            self.cycle_inf.append(torch.tensor(1.0))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        cycle_index = torch.tensor([ci[s_end] for ci in self.cycle_index_list], dtype=torch.long)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def get_cycle(self):
        return self.cycle_inf

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None, top_k=1, device='cpu',percent=100):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        self.top_k = top_k
        self.device = device
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        num_train = int(len(data) * 0.6)
        num_test = int(len(data) * 0.2)
        num_valid = int(len(data) * 0.2)
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # add cycle
        if self.cycle == 0:
            border_a = border1s[0]
            border_b = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len
            x_tensor = torch.tensor(data[border_a:border_b], dtype=torch.float32).unsqueeze(0).to(self.device)
            period, weight = FFT_for_Period(x_tensor, self.top_k)
            # print(f"[Info] train len: {-border_a+border_b+1}")
            # print(f"[Info] Auto detected cycle: {period}")
            # print(f"[Info] weight: {weight}")

            self.cycle_index_list = [
                (np.arange(len(data)) % c)[border1:border2]
                for c in period
            ]
            self.cycle_inf = [period]
            self.cycle_inf.append(weight)
        else:
            self.cycle_index_list = [(np.arange(len(data)) % self.cycle)[border1:border2]]
            self.cycle_inf = [[self.cycle]]
            self.cycle_inf.append(torch.tensor(1.0))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        cycle_index = torch.tensor([ci[s_end] for ci in self.cycle_index_list], dtype=torch.long)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def get_cycle(self):
        return self.cycle_inf

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


