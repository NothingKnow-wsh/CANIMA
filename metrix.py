import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# 这是一个多分类问题，y_true是target，y_pred是模型预测结果，数据格式为numpy

def cal(y_true, y_pred, epoch):
    # confusion matrix row means GT, column means predication
    name = 'Confusion matrix'
    '''画混淆矩阵'''
    mat = confusion_matrix(y_true, y_pred)
    da = pd.DataFrame(mat, index=['0', '1'])
    sns.heatmap(da, annot=True, cbar=None, cmap='Blues', fmt='.20g')
    plt.title(name)
    # plt.tight_layout()yt
    plt.ylabel('True Label')
    plt.xlabel('Predict Label')
    plt.savefig('{}/{}_{}.png'.format('result2', name, epoch))  # 将混淆矩阵图片保存下来
    plt.show()
    plt.close()

    '''计算指标'''
    tp = np.diagonal(mat)  # 每类的tp
    gt_num = np.sum(mat, axis=1)  # axis = 1 指每行 ，每类的总数
    pre_num = np.sum(mat, axis=0)  # axis = 0 指每列 ，预测结果中每类的总数
    fp = pre_num - tp   # 我预测n个0类，只预测对了m个是真0类，那么剩下n-m个就是假0类
    fn = gt_num - tp    # 一共n个0类，只预测对了m个是真0类，那么剩下n-m个就是真0类被预测成其他的数量
    num = np.sum(gt_num)   # 所有类别的数量总和
    num = np.repeat(num, gt_num.shape[0])
    gt_num0 = num - gt_num   # 除去各个类别，剩下的总数量，如去除0类，剩下1+2类的总数量为gt_num0[0]
    tn = gt_num0 - fp    # 除去本身类别剩下的总数量，再减去将本身类别错误预测成其他类别的数量，就是正确的阴性数量

    recall = tp.astype(np.float32) / gt_num
    specificity = tn.astype(np.float32) / gt_num0
    precision = tp.astype(np.float32) / pre_num
    F1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp + tn).astype(np.float32) / num

    print('recall:', recall, '\nmean recall:{:.4f}'.format(np.mean(recall)))
    print('specificity:', specificity, '\nmean specificity:{:.4f}'.format(np.mean(specificity)))
    print('precision:', precision, '\nmean precision:{:.4f}'.format(np.mean(precision)))
    print('F1:', F1, '\nmean F1:{:.4f}'.format(np.mean(F1)))
    print('acc:', acc, '\nmean acc:{:.4f}'.format(np.mean(acc)))
    return np.mean(F1), np.mean(precision), np.mean(recall)

def paint_ROC(y_test, y_score, epoch):

    '''画ROC曲线'''
    plt.figure()
    # 修改颜色
    colors = ['','darkred', 'darkorange', 'cornflowerblue', 'forestgreen']

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # print('label',y_test)
    # print('score', y_score)

    label = np.zeros((len(y_test), 2),  dtype="uint8")
    # y_score = np.zeros((len(y_pred), 3),  dtype="uint8")
    for i in range(len(y_test)):
        label[i][int(y_test[i])] = 1
        # y_score[i][int(y_pred[i])] = 1
    # print('label',label)

    for i in range(1,3):
        fpr[i], tpr[i], _ = metrics.roc_curve(label[:, i-1], y_score[:, i-1])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr["mean"], tpr["mean"], _ = metrics.roc_curve(label.ravel(), y_score.ravel())
    roc_auc["mean"] = metrics.auc(fpr["mean"], tpr["mean"])

    lw = 2
    plt.plot(fpr["mean"], tpr["mean"],
         label='average, ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["mean"]),
         color='k', linewidth=lw)

    for i in range(1,3):
        auc = roc_auc[i]
        # 输出不同类别的FPR\TPR\AUC
        print('label: {}, auc: {}'.format(i,  auc))
        plt.plot(fpr[i], tpr[i], color=colors[i],linestyle=':',lw = lw, label='Label = {0}, ROC curve (area = {1:0.4f})'.format(i-1, auc))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.grid(linestyle='-.')
    plt.grid(True)
    plt.legend(loc="lower right")
    # 保存绘制好的ROC曲线
    plt.savefig('{}/{}_{}.png'.format('result2', 'ROC', epoch))
    plt.show()
    plt.close()