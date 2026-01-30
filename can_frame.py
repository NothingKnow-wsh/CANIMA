import numpy as np
import torch
import torch.utils.data as data
from pandas import read_csv, read_table
from input_generator import frame_generator,one_hot_encode


class CANIMG(data.Dataset):
    def __init__(self, csv_file, sig_ref, attack_label, frame_sqlen, frame_size, is_data, is_onehot, step, is_drop=False, sr=None):
        """
        :param csv_file: 文件路径
        :param t: 图像序列长度
        :param attack_label: 攻击类型标签
        :param is_drop: 用于丢弃数据集中的正常图像，非必要不使用
        :param sr: 可用于跳过数据集的指定部分(前)
        """
        self.frame_sqlen = frame_sqlen
        self.frame_size = frame_size
        self.ak_label = attack_label
        self.is_drop = is_drop
        self.step = step
        self.is_data = is_data
        self.is_onehot = is_onehot
        series = read_csv(csv_file, skiprows=sr)
        sig_series = read_csv(sig_ref, skiprows=None)

        self.ID_list = series.Arbitration_ID.values
        self.data_field = series.Data.values
        self.flag_list = series.Class.values

        self.id_list = list(sig_series.ID.values)
        self.phy_signal = sig_series.physical
        self.cyc_signal = sig_series.cyclic
        self.le_flag = sig_series.LE
        self.dlc = list(sig_series.DLC)
        self.total_frame = series.shape[0]

    def __getitem__(self, idx):
        start_idx = idx*self.step

        frame_data, end_idx = frame_generator(s_idx=start_idx, ID_list=self.ID_list, data_field=self.data_field, id_list=self.id_list, phy_signal=self.phy_signal,
                                     cyc_signal=self.cyc_signal, le_flag=self.le_flag, dlc=self.dlc, frame_size=self.frame_size, sq=self.frame_sqlen, is_onehot=self.is_onehot, is_data=self.is_data)
        sub_flag_list = self.flag_list[start_idx: end_idx]
        frame_data = torch.reshape(frame_data, [-1, 1, self.frame_size, self.frame_size])
        # state_num = len(np.unique(sub_flag_list))
        anomaly_num = np.sum(sub_flag_list == 'Attack')
        # if state_num != 1 and "Normal" in sub_flag_list:   # 如果sub_flag_list中有Normal且不止Normal，则有攻击
        if anomaly_num / (end_idx-start_idx+1) > 0.1:
            label = self.ak_label
        else:
            label = 0  # 反之，无攻击
        if self.is_drop is True and label == 0:
            return [None, None]
        else:
            out = [frame_data, label]
            return out

    def __len__(self):
        frame = []
        for i in range(self.total_frame - 1, 0, -1):
            id_part = self.ID_list[i]
            data_part = self.data_field[i]

            if self.is_onehot is True:
                bin_id = one_hot_encode(id_part)
            else:
                binstr = bin(int(id_part, 16))
                binstr = binstr[2:].zfill(11)
                bin_id = [int(i) for i in binstr]
            if self.is_data is True:
                # if id_part in self.id_list:
                #     id_index = self.id_list.index(id_part)
                    data_content = data_part.replace(' ', '')
                    data_str = bin(int(data_content, 16))
                    # data_str = data_str[2:].zfill(self.dlc[id_index] * 8)
                    data_str = data_str[2:].zfill(64)
                    bin_data = [int(j) for j in data_str]
                #     if self.le_flag[id_index] == 1:
                #         temp = bin_data
                #         for j in range(0, len(bin_data), 8):
                #             temp[j:j + 8] = bin_data[self.dlc[id_index] * 8 - j - 8:self.dlc[id_index] * 8 - j]
                #         bin_data = temp
                #     phy_ref = []
                #     cyc_ref = []
                #     if str(self.phy_signal[id_index]) != 'nan':
                #         phy_ref = [int(sig.strip()) for sig in self.phy_signal[id_index].split(' ')]
                #     if str(self.cyc_signal[id_index]) != 'nan':
                #         cyc_ref = [int(sig.strip()) for sig in self.cyc_signal[id_index].split(' ')]
                #     phy_part = []
                #     cyc_part = []
                #     for j in range(0, len(phy_ref), 2):
                #         phy_part = phy_part + bin_data[phy_ref[j] - 1:phy_ref[j + 1]]
                #     for j in range(0, len(cyc_ref), 2):
                #         cyc_part = cyc_part + bin_data[cyc_ref[j] - 1:cyc_ref[j + 1]]
                #     bin_sig = phy_part + cyc_part
                # else:
                #     bin_sig = []
                    bin_sig = bin_data
            else:
                bin_sig = []
            current_frame = bin_id + bin_sig
            frame = frame + current_frame
            if len(frame) > self.frame_size * self.frame_size * self.frame_sqlen:
                dataset_size = i + 1
                break

        return int(dataset_size/self.step)-1


