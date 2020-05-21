import pandas as pd
import numpy as np
# 训练数据集_trd.csv是顾客消费的数据，每位顾客可能有多条消费记录，按标准时间格式读取数据文件
data = pd.read_csv('训练数据集_trd.csv',parse_dates=['trx_tm'])
# 查看顾客的消费均为2019年5月份和6月份，同理可查看消费季节dt.quarter等
print(set(data['trx_tm'].dt.year))
print(set(data['trx_tm'].dt.month))

# 提取顾客消费月份，以及消费金额
data['month'] = data['trx_tm'].dt.month
data = data[['id', 'cny_trx_amt', 'month']]

# 处理数据,提取每个客户id分别在5、6月份的消费金额
df = pd.DataFrame(data.groupby('id').apply(lambda data: [np.nansum(data[data['month']==5]['cny_trx_amt']), np.nansum(data[data['month']==6]['cny_trx_amt'])]), columns=['cny_trx_amt']).reset_index(drop=False)

# 特征提取
def Extract_features(df):
    trans_features = df['cny_trx_amt']
    # 计算月消费金额大于0的月份
    pos_nag_features = [len([xi for xi in x if xi>0]) for x in trans_features]
    df['positive_Transaction'] = pos_nag_features
    # 计算两个月消费的均值
    Aver_features = [np.mean(x) for x in trans_features]
    df['Average_Transaction'] = Aver_features
    # 计算两个月的差值
    Difference_features = [x[1]-x[0] for x in trans_features]
    df['Difference_Transaction'] = Difference_features

    return df

if __name__ == '__main__':
    new_features = Extract_features(df)
