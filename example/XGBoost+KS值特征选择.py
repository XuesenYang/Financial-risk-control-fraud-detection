import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve


# 计算KS值，kS值可以描述模型对于正负例的区分能力
def sloveKS(model, X, Y):
    """
    :param model: 预测模型
    :param X: 特征
    :param Y: 真实标签
    :return: KS值
    """
    Y_predict = [s[1] for s in model.predict_proba(X)]
    fpr, tpr, thresholds = roc_curve(Y, Y_predict)
    ks = max(tpr - fpr)
    return ks

# 计算PSI值， 稳定度指标(population stability index ,PSI)可衡量测试样本及模型开发样本评分的的分布差异
def slovePSI(model, train_x, val_x):
    train_predict_y = [s[1] for s in model.predict_proba(train_x)]  
    train_nrows = train_x.shape[0]  
    train_predict_y.sort()  
    #等频分箱成10份  
    cutpoint = [-100] + [train_predict_y[int(train_nrows/10*i)] 
                         for i in range(1, 10)] + [100]  
    cutpoint = list(set(cutpoint))  
    cutpoint.sort()
    val_predict_y = [s[1] for s in list(model.predict_proba(val_x))]  
    val_nrows = val_x.shape[0]  
    PSI = 0  
    #每一箱之间分别计算PSI  
    for i in range(len(cutpoint)-1):  
        start_point, end_point = cutpoint[i], cutpoint[i+1]
        train_cnt = [p for p in train_predict_y 
                                 if start_point <= p < end_point]  
        train_ratio = len(train_cnt) / train_nrows + 1e-10  
        val_cnt = [p for p in val_predict_y 
                                 if start_point <= p < end_point]  
        val_ratio = len(val_cnt) / val_nrows + 1e-10  
        psi = (train_ratio - val_ratio) * math.log(train_ratio/val_ratio)
        PSI += psi  
    return PSI
  
class feature_selection_model(object):
    def __init__(self, datasets, uid, dep,
                 var_names,  max_del_var_nums=0):
        self.datasets = datasets  
        #样本唯一标识，不参与建模  
        self.uid = uid       
        #二分类标签  
        self.dep = dep
        #特征列表  
        self.var_names = var_names
        #单次迭代最多删除特征的个数  
        self.max_del_var_nums = max_del_var_nums    
        self.row_num = 0  
        self.col_num = 0  
  
    def training(self, min_score=0.0001):
        lis = self.var_names[:]
        orthers, test_data = train_test_split(self.datasets, test_size=0.2)  # 测试集
        train_data, val_data = train_test_split(orthers, test_size=0.3)  # 训练集，验证集

        model = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)
        while len(lis) > 0:   
            #模型训练
            model.fit(X=train_data[self.var_names], y=train_data[self.dep])  
            #得到特征重要性  
            scores = model.feature_importances_
            #清空字典  
            lis.clear()      
            ''' 
            当特征重要性小于预设值时， 
            将特征放入待删除列表。 
            当列表长度超过预设最大值时，跳出循环。 
            即一次只删除限定个数的特征。 
            '''  
            for (idx, var_name) in enumerate(self.var_names):  
                #小于特征重要性预设值则放入列表  
                if scores[idx] < min_score:    
                    lis.append(var_name)  
                #达到预设单次最大特征删除个数则停止本次循环  
                if len(lis) >= self.max_del_var_nums:     
                    break  
            #训练集KS  
            trainks = sloveKS(model, train_data[self.var_names],
                              train_data[self.dep])
            #初始化ks值和PSI  
            valks, testks, valpsi, testpsi = 0.0, 0.0, 0.0, 0.0 
            #测试集KS和PSI  
            if not isinstance(val_data, str):  
                valks = sloveKS(model, val_data[self.var_names],
                                     val_data[self.dep])
                valpsi = slovePSI(model, train_data[self.var_names], val_data[self.var_names])
            #跨时间验证集KS和PSI  
            if not isinstance(test_data, str):  
                testks = sloveKS(model, test_data[self.var_names],test_data[self.dep])
                testpsi = slovePSI(model, train_data[self.var_names], test_data[self.var_names])
            #将三个数据集的KS和PSI放入字典  
            dic = {"train_ks": float(trainks), "val_ks": float(valks), "test_ks": testks,
                   "val_psi": float(valpsi), "test_psi": testpsi}
            print("del var: ", len(self.var_names), "-->", len(self.var_names) - len(lis),
                  "ks: ", dic, ",".join(lis))
            # 去掉重要性较小的特征
            self.var_names = [var_name for var_name in self.var_names if var_name not in lis]

        return self.var_names

if __name__ == '__main__':
    data = pd.read_csv('训练数据集_tag.csv')
    for j in data.columns.values.tolist():
        data[j].loc[data[j] == '\\N'] = 0
        data[j].loc[data[j] == '~'] = 0
    data.fillna(0, inplace=True)

    features_names = data.columns.values.tolist()
    fs_model =  feature_selection_model(datasets = data, uid = 'id', dep = 'flag',
                 var_names = data.columns.values.tolist()[8:], max_del_var_nums=5)
    feature_subset = fs_model.training( min_score=0.001)
    print(feature_subset)
