import numpy as np
import torch


def one_hot_encode(hex_str):
    """
    输入十六进制字符串，输出对应的 ONEHOT 编码向量
    :param hex_str:
    :return: ONEHOT 编码向量 one_hot_code
    """
    # 判断输入是 0xxx 还是 xxx，前者舍弃第一位
    if len(hex_str) == 4:
        hex_str = hex_str[1:]
    # hex_str 的每一位都转换为一串16位二进制向量，在对应数值的index处为1，其余为0
    one_hot_code = [0] * 16 * len(hex_str)
    for i in range(len(hex_str)):
        one_hot_code[int(hex_str[i], 16) + 16 * i] = 1
    return one_hot_code


def trans_hexframe2binframe(frame_content, is_data, is_onehot):
    frame_content = frame_content.replace(' ', '')
    frame_content = frame_content[:3] + frame_content[3:].zfill(16)
    if is_data is False:
        info = frame_content[:3]
        if is_onehot is True:
            binframe = one_hot_encode(info)
        else:
            binstr = bin(int(info, 16))
            binstr = binstr[2:].zfill(11)
            binframe = [int(i) for i in binstr]
    else:
        info = frame_content
        binstr = bin(int(info, 16))
        binstr = binstr[2:].zfill(75)
        binframe = [int(i) for i in binstr]
    return binframe

def frame_generator(s_idx, ID_list, data_field, id_list, phy_signal, cyc_signal, le_flag, dlc, frame_size, sq, is_onehot, is_data):  #输出单张frame的tensor类型
    frame = []
    for i in range(s_idx, len(ID_list)):
        id_part = ID_list[i]
        data_part = data_field[i]
        if is_onehot is True:
            bin_id = one_hot_encode(id_part)
        else:
            binstr = bin(int(id_part, 16))
            binstr = binstr[2:].zfill(11)
            bin_id = [int(i) for i in binstr]
        if is_data is True:
            # if id_part in id_list:
            #     id_index = id_list.index(id_part)
                data_content = data_part.replace(' ', '')
                data_str = bin(int(data_content, 16))
                # data_str = data_str[2:].zfill(dlc[id_index] * 8)
                data_str = data_str[2:].zfill(64)
                bin_data = [int(j) for j in data_str]
            #     if le_flag[id_index] == 1:
            #         temp = bin_data
            #         for j in range(0, len(bin_data), 8):
            #             temp[j:j + 8] = bin_data[dlc[id_index] * 8 - j - 8:dlc[id_index] * 8 - j]
            #         bin_data = temp
            #     phy_ref = []
            #     cyc_ref = []
            #     if str(phy_signal[id_index]) != 'nan':
            #         phy_ref = [int(sig.strip()) for sig in phy_signal[id_index].split(' ')]
            #     if str(cyc_signal[id_index]) != 'nan':
            #         cyc_ref = [int(sig.strip()) for sig in cyc_signal[id_index].split(' ')]
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
        temp_frame = frame
        frame = frame + current_frame
        if len(frame) > frame_size * frame_size * sq:
            frame = temp_frame + [0] * (frame_size * frame_size * sq - len(temp_frame))
            frame = np.array(frame)
            frame = frame.reshape(sq, frame_size, frame_size)
            end_idx = i
            break
    frame_data = torch.from_numpy(frame)  # 转tensor
    return frame_data, end_idx


