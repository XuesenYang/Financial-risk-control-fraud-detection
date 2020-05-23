"""
tradaboost 是一个框架，理论上可以把任意分类器嵌入到该框架中，例如本案例把决策树作为分类器
"""
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt

def tradaboost(trans_S, trans_A, label_S, label_A, test, test_y, N):
    trans_data = np.concatenate((trans_A, trans_S), axis=0)
    trans_label = np.concatenate((label_A, label_S), axis=0)

    row_A = trans_A.shape[0]
    row_S = trans_S.shape[0]
    row_T = test.shape[0]

    test_data = np.concatenate((trans_data, test), axis=0)

    # 初始化权重
    weights_A = np.ones([row_A, 1]) / row_A
    weights_S = np.ones([row_S, 1]) / row_S
    weights = np.concatenate((weights_A, weights_S), axis=0)

    bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

    # 存储每次迭代的标签和bata值？
    bata_T = np.zeros([1, N])
    result_label = np.ones([row_A + row_S + row_T, N])

    predict = np.zeros([row_T])
    predict_pro = np.zeros([row_T])

    print('params initial finished.')
    trans_data = np.asarray(trans_data, order='C')
    trans_label = np.asarray(trans_label, order='C')
    test_data = np.asarray(test_data, order='C')


    for i in range(N):
        P = calculate_P(weights, trans_label)

        result_label[:, i] = train_classify(trans_data, trans_label, test_data, P)
        # print('result,', result_label[:, i], row_A, row_S, i, result_label.shape)

        error_rate = calculate_error_rate(label_S, result_label[row_A:row_A + row_S, i],
                                          weights[row_A:row_A + row_S, :])
        print('Error rate:', error_rate)

        if error_rate > 0.5:
            error_rate = 0.5
        if error_rate == 0:
            N = i
            break  # 防止过拟合
            # error_rate = 0.001

        bata_T[0, i] = error_rate / (1 - error_rate)

        # 调整源域样本权重
        for j in range(row_S):
            weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i],
                                                               (-np.abs(result_label[row_A + j, i] - label_S[j])))

        # 调整辅域样本权重
        for j in range(row_A):
            weights[j] = weights[j] * np.power(bata, np.abs(result_label[j, i] - label_A[j]))
    # print bata_T
    for i in range(row_T):
        # 跳过训练数据的标签
        left = np.sum(
            result_label[row_A + row_S + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
        right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))

        predict_pro[i] = left/(left+right)
        if left >= right:
            predict[i] = 1
        else:
            predict[i] = 0
            # print left, right, predict[i]

    return predict, predict_pro


def calculate_P(weights, label):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


def train_classify(trans_data, trans_label, test_data, P):
    clf = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random")
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    return clf.predict(test_data)


def calculate_error_rate(label_R, label_H, weight):
    total = np.sum(weight)

    # print(weight[:, 0] / total)
    # print(np.abs(label_R - label_H))
    return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))


if __name__ == '__main__':
    data = pd.read_excel('tra_sample.xlsx')
    data.head()
    feature_lst = ['zx_score', 'msg_cnt', 'phone_num_cnt', 'register_days']
    train = data[data.type == 'target'].reset_index().copy()
    diff = data[data.type == 'origin'].reset_index().copy()
    val = data[data.type == 'offtime'].reset_index().copy()

    TrainS = train[feature_lst].copy()
    LabelS = train['bad_ind'].copy()

    TrainA = diff[feature_lst].copy()
    LabelA = diff['bad_ind'].copy()

    Test = val[feature_lst].copy()
    Test_y = val['bad_ind'].copy()

    # H 测试样本分类结果
    # TrainS 原训练样本 np数组
    # TrainA 辅助训练样本
    # LabelS 原训练样本标签
    # LabelA 辅助训练样本标签
    # Test  测试样本
    # N 迭代次数

    predict_label, predict_probability = tradaboost(TrainS, TrainA, LabelS, LabelA, Test, Test_y, 20)
    auc_0 = roc_auc_score(Test_y, predict_probability)
    print('迁移学习后的auc值:', auc_0)
    input_x = trans_data = np.concatenate((TrainA, TrainS), axis=0)
    input_y = np.concatenate((LabelA, LabelS), axis=0)
    clf = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random")
    clf.fit(input_x, input_y)
    auc_1 = roc_auc_score(Test_y, clf.predict_proba(Test)[:,1])
    print('单纯决策树建模的auc值', auc_1)


    fpr_0, tpr_0, _ = roc_curve(Test_y,predict_probability)
    fpr_1, tpr_1, _ = roc_curve(Test_y, clf.predict_proba(Test)[:,1])

    plt.plot(fpr_0, tpr_0, label='DT+tradaboost AUC={0:0.2f}'.format(auc_0))
    plt.plot(fpr_1, tpr_1, label='simpleDT AUC={0:0.2f}'.format(auc_1))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

