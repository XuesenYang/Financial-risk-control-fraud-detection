import pandas as pd  
import numpy as np  
import os
from sklearn import tree
import pydotplus
from IPython.display import Image
from io import StringIO
import os


# Windows安装Graphviz, 版本为2.38，默认安装路径'C:/Program Files (x86)/Graphviz2.38/bin/'   
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  
#读取数据  
data = pd.read_excel('data_for_tree.xlsx')
data.head()  

# org_list 为用户信息特征； agg_list 为连续性信贷特征，dstc_list为离散型信贷特征
org_lst = ['uid','create_dt','oil_actv_dt','class_new','bad_ind']
agg_lst = ['oil_amount','discount_amount','sale_amount','amount','pay_amount','coupon_amount','payment_coupon_amount']
dstc_lst = ['channel_code','oil_code','scene','source_app','call_source']

# df为抽取出来准备进行特征提取的数据
df = data[org_lst].copy()
df[agg_lst] = data[agg_lst].copy()
df[dstc_lst] = data[dstc_lst].copy()

base = df[org_lst].copy()
base = base.drop_duplicates(['uid'],keep = 'first')

# 特征提取过程，对于每个用户id groupby 然后多维度抽取特征
gn = pd.DataFrame()  
for i in agg_lst:  
    #计算个数  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                       lambda df:len(df[i])).reset_index())  
    tp.columns = ['uid',i + '_cnt']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #求历史特征值大于零的个数  
    tp = pd.DataFrame(df.groupby('uid').apply(
                          lambda df:np.where(df[i]>0,1,0).sum()).reset_index())  
    tp.columns = ['uid',i + '_num']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求和  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                  lambda df:np.nansum(df[i])).reset_index())  
    tp.columns = ['uid',i + '_tot']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求均值  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                    lambda df:np.nanmean(df[i])).reset_index())  
    tp.columns = ['uid',i + '_avg']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求最大值  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                     lambda df:np.nanmax(df[i])).reset_index())  
    tp.columns = ['uid',i + '_max']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求最小值  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                    lambda df:np.nanmin(df[i])).reset_index())  
    tp.columns = ['uid',i + '_min']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求方差  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                     lambda df:np.nanvar(df[i])).reset_index())  
    tp.columns = ['uid',i + '_var']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求极差  
    tp = pd.DataFrame(df.groupby('uid').apply(
                lambda df:np.nanmax(df[i])-np.nanmin(df[i]) ).reset_index())  
    tp.columns = ['uid',i + '_ran']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')  
    #对历史数据求变异系数,为防止除数为0，利用0.01进行平滑  
    tp = pd.DataFrame(df.groupby('uid').apply(lambda df:np.nanmean(df[i])/(np.nanvar(df[i])+0.01))).reset_index()  
    tp.columns = ['uid',i + '_cva']  
    if gn.empty == True:  
        gn = tp  
    else:  
        gn = pd.merge(gn,tp,on = 'uid',how = 'left') 

gc = pd.DataFrame()  
for i in dstc_lst:  
    tp = pd.DataFrame(df.groupby('uid').apply(
                                   lambda df: len(set(df[i]))).reset_index())  
    tp.columns = ['uid',i + '_dstc']  
    if gc.empty == True:  
        gc = tp  
    else:  
        gc = pd.merge(gc,tp,on = 'uid',how = 'left')

fn =  base.merge(gn,on='uid').merge(gc,on='uid')  
fn = pd.merge(fn,gc,on= 'uid')

x = fn.drop(['uid','oil_actv_dt','create_dt','bad_ind','class_new'],axis = 1)
y = fn.bad_ind.copy()
# 训练决策树模型
dtree = tree.DecisionTreeRegressor(max_depth = 2,min_samples_leaf = 500,min_samples_split = 5000)
dtree = dtree.fit(x,y)

#打印决策树
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
dot_data = StringIO()
tree.export_graphviz(dtree, out_file=dot_data,
                         feature_names=x.columns,
                         class_names=['bad_ind'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('example.png')
Image(graph.create_png())
