import numpy as np
import torch
import torch.utils.data as data
from pandas import read_csv
import math
from input_generator import one_hot_encode
# training phase
pure_normal_file = r"Car_Hacking_Challenge_Dataset_rev20Mar2021\0_Preliminary\0_Training\Pre_train_D_0.csv"

ref = r"signal.csv"


frame_size = 28
sq = 3

series = read_csv(pure_normal_file, skiprows=None)
ID_list = series.Arbitration_ID.values
data_field = series.Data.values

sig_series = read_csv(ref, skiprows=None)
id_list = list(sig_series.ID.values)
phy_signal = sig_series.physical
cyc_signal = sig_series.cyclic
le_flag = sig_series.LE
dlc = list(sig_series.DLC)

white_list = list(set(ID_list))
total_frame = series.shape[0]
is_onehot = False

frame = []
for i in range(total_frame):
    id_part = ID_list[i]
    data_part = data_field[i]

    if is_onehot is True:
        bin_id = one_hot_encode(id_part)
    else:
        binstr = bin(int(id_part, 16))
        binstr = binstr[2:].zfill(12)
        bin_id = [int(i) for i in binstr]


    if id_part in id_list:
        id_index = id_list.index(id_part)
        data_content = data_part.replace(' ', '')
        data_str = bin(int(data_content, 16))
        data_str = data_str[2:].zfill(dlc[id_index]*8)
        bin_data = [int(j) for j in data_str]
        if le_flag[id_index] == 1:
            temp = bin_data
            for j in range(0, len(bin_data), 8):
                temp[j:j+8] = bin_data[dlc[id_index]*8-j-8:dlc[id_index]*8-j]
            bin_data = temp
        phy_ref = []
        cyc_ref = []
        if str(phy_signal[id_index]) != 'nan':
            phy_ref = [int(sig.strip()) for sig in phy_signal[id_index].split(' ')]
        if str(cyc_signal[id_index]) != 'nan':
            cyc_ref = [int(sig.strip()) for sig in cyc_signal[id_index].split(' ')]
        phy_part = []
        cyc_part = []
        for j in range(0, len(phy_ref), 2):
            phy_part = phy_part + bin_data[phy_ref[j]-1:phy_ref[j+1]]
        for j in range(0, len(cyc_ref), 2):
            cyc_part = cyc_part + bin_data[cyc_ref[j]-1:cyc_ref[j+1]]
        bin_sig = phy_part + cyc_part
    current_frame = bin_id + bin_sig
    temp_frame = frame
    frame = frame + current_frame
    if len(frame) > frame_size * frame_size * sq:
        frame = temp_frame + [0] * (frame_size * frame_size * sq - len(temp_frame))
        frame = np.array(frame)
        frame = frame.reshape(sq, frame_size, frame_size)
        break

frame = []
for i in range(total_frame - 1, 0, -1):
    id_part = ID_list[i]
    data_part = data_field[i]

    if is_onehot is True:
        bin_id = one_hot_encode(id_part)
    else:
        binstr = bin(int(id_part, 16))
        binstr = binstr[2:].zfill(12)
        bin_id = [int(i) for i in binstr]

    if id_part in id_list:
        id_index = id_list.index(id_part)
        data_content = data_part.replace(' ', '')
        data_str = bin(int(data_content, 16))
        data_str = data_str[2:].zfill(dlc[id_index] * 8)
        bin_data = [int(j) for j in data_str]
        if le_flag[id_index] == 1:
            temp = bin_data
            for j in range(0, len(bin_data), 8):
                temp[j:j + 8] = bin_data[dlc[id_index] * 8 - j - 8:dlc[id_index] * 8 - j]
            bin_data = temp
        phy_ref = []
        cyc_ref = []
        if str(phy_signal[id_index]) != 'nan':
            phy_ref = [int(sig.strip()) for sig in phy_signal[id_index].split(' ')]
        if str(cyc_signal[id_index]) != 'nan':
            cyc_ref = [int(sig.strip()) for sig in cyc_signal[id_index].split(' ')]
        phy_part = []
        cyc_part = []
        for j in range(0, len(phy_ref), 2):
            phy_part = phy_part + bin_data[phy_ref[j] - 1:phy_ref[j + 1]]
        for j in range(0, len(cyc_ref), 2):
            cyc_part = cyc_part + bin_data[cyc_ref[j] - 1:cyc_ref[j + 1]]
        bin_sig = phy_part + cyc_part
    current_frame = bin_id + bin_sig
    temp_frame = frame
    frame = frame + current_frame
    if len(frame) > frame_size * frame_size * sq:
        frame = temp_frame + [0] * (frame_size * frame_size * sq - len(temp_frame))
        frame = np.array(frame)
        frame = frame.reshape(sq, frame_size, frame_size)
        dataset_size = i+1
        break
print(dataset_size)


print()