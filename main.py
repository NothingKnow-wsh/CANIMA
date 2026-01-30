from pandas import read_csv
from can_frame import CANIMG
import torch
from tqdm import tqdm
from network import can_convlstm
from metrix import cal, paint_ROC

sq_len = 4       # 设置frame序列长度
layers_num = 5   # 设置ConvLSTM网络层数
layers_channel = [64, 32, 16, 8, 2]  # 设置ConvLSTM网络每层的通道数，从左到右对应自下而上[32,32,16,8,2]
batch_size = 64  # 设置训练batch-size
frame_size = 28
is_data = True
is_onehot = False
ref = r"signal.csv"
save_path = r"check_point"   # 实验结果保存路径
pure_normal_file = r"Car_Hacking_Challenge_Dataset_rev20Mar2021\0_Preliminary\0_Training\Pre_train_D_0.csv"

attack_train_file1 = r"Car_Hacking_Challenge_Dataset_rev20Mar2021\0_Preliminary\0_Training\Pre_train_D_1.csv"

attack_train_file2 = r"Car_Hacking_Challenge_Dataset_rev20Mar2021\0_Preliminary\0_Training\Pre_train_D_2.csv"

attack_train_file3 = r"Car_Hacking_Challenge_Dataset_rev20Mar2021\0_Preliminary\1_Submission\Pre_submit_D.csv"

def main():
    trainset = CANIMG(csv_file=attack_train_file1, sig_ref=ref, attack_label=1, frame_sqlen=sq_len, frame_size=frame_size, is_data=is_data, is_onehot=is_onehot, step=10)
    testset = CANIMG(csv_file=attack_train_file2, sig_ref=ref, attack_label=1, frame_sqlen=sq_len, frame_size=frame_size, is_data=is_data, is_onehot=is_onehot, step=10)

    # small_train_set, _ = torch.utils.data.random_split(trainset, [int(len(trainset) * 0.7), len(trainset) - int(len(trainset) * 0.7)])
    small_test_set, _ = torch.utils.data.random_split(testset, [int(len(testset)*0.2), len(testset)-int(len(testset)*0.2)])

    trainLoader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=False,
                                              num_workers = 6)
    testLoader = torch.utils.data.DataLoader(small_test_set,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=False,
                                         num_workers = 6)

    can_network = can_convlstm(input_dim=1, hidden_dim=layers_channel, kernel_size=(5, 5), num_layers=layers_num, frame_size=frame_size, batch_first=True)
    # GPU训练
    if torch.cuda.is_available():
        can_network = can_network.cuda()
    # 设置损失函数，学习率，优化器，epoch
    loss_func = torch.nn.CrossEntropyLoss()
    loss_func = loss_func.cuda()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(can_network.parameters(), lr=learning_rate)
    epoch = 50

    normal_case = 0
    attack_case = 0
    for e in range(epoch):
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        total_ac = 0
        can_network.train()
        for i, (imgs, labels) in enumerate(t):
            count = torch.unique(labels, return_counts=True)
            normal_case = normal_case + count[1][0]
            attack_case = attack_case + count[1][1]
            imgs = imgs.cuda()
            labels = labels.cuda()
            L_out = can_network(imgs)
            accuracy = (L_out.argmax(1) == labels).sum()
            total_ac += accuracy
            loss = loss_func(L_out, labels)
            loss_aver = loss.item() / batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(e)
            })
        torch.save(can_network, '{}/{}_layers_{}_input_length_epoch_{}.pth'.format(save_path, layers_num, sq_len, e))
        can_network.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            y_score = []
            sum_time = 0
            t = tqdm(testLoader, leave=False, total=len(testLoader))
            for i, (imgs, labels) in enumerate(t):
                if i == len(testLoader)-1:
                    continue
                imgs = imgs.cuda()
                labels = labels.cuda()
                L_out = can_network(imgs)
                y_score.append(L_out)
                y_true.append(labels)
                y_pred.append(L_out.argmax(1))
            y_true = torch.reshape(torch.stack(y_true), [-1, ])
            y_pred = torch.reshape(torch.stack(y_pred), [-1, ])
            y_score = torch.reshape(torch.stack(y_score), [-1, 2])
            y_true = y_true.cpu()
            y_pred = y_pred.cpu()
            y_score = y_score.cpu()
            print("-------------------------epoch {}".format(e))
            f, p, r = cal(y_true, y_pred, e)
            paint_ROC(y_true, y_score, e)
            print("---------------------------------------------------")
    print(normal_case)
    print(attack_case)
                    # drop_last = False


if __name__ == '__main__':
    main()
