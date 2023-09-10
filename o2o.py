import numpy as np
import pandas as pd
import xgboost as xgb


def train_prepare(train):
    data = train.copy()
    data=data_prepare(data)
    data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    data['Date'].fillna(0,inplace=True)
    return data

def data_prepare(test):
    data = test.copy()
    # 处理空缺
    data['Distance'].fillna(-1, inplace=True)
    # 处理折扣率
    data['discount_rate'] = data['Discount_rate'].map(lambda x:float(x) if ":" not in str(x) else (float(str(x.split(':')[0])) - float(str(x.split(':')[1]))) / float(str(x.split(":")[0])))
    data['discount_rate'].fillna(0,inplace=True)
    # 优惠卷是否为满减
    data['is_manjian'] = data['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    # 最低满减消费
    data['min_cost_of_manjian'] = data['Discount_rate'].map(lambda x: 0 if ':' not in str(x) else int(str(x).split(':')[0]))
    # 大额优惠卷标记
    data['high_discount_rate'] = data['discount_rate'].map(lambda x: 1 if x <= 0.7 else 0)
    # 添加回头客机制
    data['is_bake'] = data['Distance'].map(lambda x: 1 if x >= 5 else 0)
    # 转时间类型
    data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    # 添加领券时间为周几
    data['weekday'] = data['date_received'].apply(lambda x: x.isoweekday())
    # 是否为周末
    data['is_weekend'] = data['weekday'].map(lambda x: 1 if (6<=x<=7) else 0)
    #领卷在月中时间
    data['is_start_month'] = data['date_received'].map( lambda x: 1 if 0<x.day <= 10 else 0)
    data['is_middle_month'] = data['date_received'].map( lambda x: 1 if 10<x.day <= 20 else 0)
    data['is_end_month'] = data['date_received'].map( lambda x: 1 if 20<x.day<= 30 else 0)

    return data


#提取特征前的准备
def feature_prepare(data):
    data['User_id']=data['User_id'].map(int)
    data['Merchant_id']=data['Merchant_id'].map(int)
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['Distance']=data['Distance'].map(int)
    if 'Date' in data.columns.tolist():
        data['Date'] = data['Date'].map(int)
    #方便提取特征
    data['cnt'] = 1
    return data


def get_user_feature(label_field):
    data=label_field.copy()
    feature=feature_prepare(data).copy()
    print('用户特征提取中')
    key = ['User_id']
    # 用户领券数
    receive_cnt = pd.pivot_table(data, index=key, values='cnt', aggfunc=sum)
    receive_cnt = pd.DataFrame(receive_cnt).rename(columns={'cnt': 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, receive_cnt, on=key, how='left')
    # 领券并消费数
    received_and_consume_cnt = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != 'NaN')], index=key, values='cnt', aggfunc=len)
    received_and_consume_cnt = pd.DataFrame(received_and_consume_cnt).rename(columns={'cnt': 'received_and_consume_cnt'}).reset_index()
    feature = pd.merge(feature, received_and_consume_cnt, on=key, how='left')
    # 领券未消费数
    received_no_consume_cnt = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) == 'NaN')], index=key, values='cnt', aggfunc=len)
    received_no_consume_cnt = pd.DataFrame(received_no_consume_cnt).rename(columns={'cnt': 'received_no_consume_cnt'}).reset_index()
    feature = pd.merge(feature, received_no_consume_cnt, on=key, how='left')
    # 用户核销率
    feature['receive_and_consume_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,feature['received_and_consume_cnt'],feature['receive_cnt']))
    # 用户核销优惠券数
    user_consume_cnt = data[data['Date_received'] != 'NaN'].groupby('User_id')['Coupon_id'].count().reset_index()
    user_consume_cnt.columns = ['User_id', 'user_consume_cnt']
    # 所有核销数
    all_consume = data[data['Date_received'] != 'NaN'].shape[0]
    # 用户核销占总核销的比例
    user_consume_cnt['user_consume_rate'] = user_consume_cnt['user_consume_cnt'] / all_consume
    feature = pd.merge(feature, user_consume_cnt[['User_id', 'user_consume_rate']], on='User_id', how='left')

    # 多少不同商家领取优惠券
    receive_differ_merchant_cnt = pd.pivot_table(data, index=key, values='Merchant_id', aggfunc=len)
    receive_differ_merchant_cnt = pd.DataFrame(receive_differ_merchant_cnt).rename(columns={'Merchant_id': 'receive_differ_merchant_cnt'}).reset_index()
    feature = pd.merge(feature, receive_differ_merchant_cnt, on=key, how='left')

    # 在多少不同商家领取并消费优惠券
    pivot = pd.pivot_table(data[data['Date_received'].map(lambda x: str(x) != 'NaN')], index=key,values='Merchant_id',aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': 'receive_consume_differ_merchant_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=key, how='left')

    #不同商家优惠券核销率
    feature['received_differ_Merchant_rate'] = list(map(lambda x, y: x / y if y != 0 else 0,feature['receive_consume_differ_merchant_cnt'],feature['receive_differ_merchant_cnt']))
    feature.fillna(0, downcast='infer')
    return feature


def get_user_merchant_feature(label_field):
    data=label_field.copy()
    feature=feature_prepare(data).copy()
    print('用户商家特征提取中')

    keys=['User_id', 'Merchant_id']
    # 用户-商家历史交易次数
    user_merchant_hist_cnt = data.groupby(keys)['cnt'].count().reset_index()
    user_merchant_hist_cnt.columns = [keys[0], keys[1], 'user_merchant_hist_cnt']
    feature = pd.merge(feature, user_merchant_hist_cnt, on=['User_id', 'Merchant_id'], how='left')

    # 用户-商家平均距离
    user_merchant_distance_avg = data.groupby(['User_id', 'Merchant_id'])['Distance'].mean().reset_index()
    user_merchant_distance_avg.columns = ['User_id', 'Merchant_id', 'user_merchant_avg_distance']
    feature = pd.merge(feature, user_merchant_distance_avg, on=['User_id', 'Merchant_id'], how='left')

    # 用户-商家联系天数
    user_merchant_conn_days = data.groupby(['User_id', 'Merchant_id'])['Date_received'].apply(lambda x: x.max() - x.min()).reset_index()
    user_merchant_conn_days.columns = ['User_id', 'Merchant_id', 'user_merchant_conn_days']
    feature = pd.merge(feature, user_merchant_conn_days, on=['User_id', 'Merchant_id'], how='left')

    return feature

def get_merchant_feature(label_field):
    data=label_field.copy()
    feature=feature_prepare(data).copy()
    keys = ['Merchant_id']

    # 商家优惠券被领取的次数
    pivot = pd.pivot_table(data, values='cnt', index=keys, aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt':'merchant_receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature.fillna(0, downcast='infer')

    # 商家被核销数
    comsume_cnt = data[data['Date_received'] != 'NaN'].groupby('Merchant_id')['Coupon_id'].count().reset_index()
    comsume_cnt.columns = ['Merchant_id', 'merchant_consume_cnt']
    feature = pd.merge(feature, comsume_cnt, on='Merchant_id', how='left')

    # 商家核销率
    pivot['merchant_consume_rate'] = comsume_cnt['merchant_consume_cnt'] / pivot['merchant_receive_cnt']
    feature = pd.merge(feature, pivot[['Merchant_id', 'merchant_consume_rate']], on='Merchant_id', how='left')
    # 所有优惠券次数
    all_coupon_cnt = data['Coupon_id'].nunique()
    # 商家被领取占所有优惠券的比例
    pivot['merchant_receive_rate'] = pivot['merchant_receive_cnt'] / all_coupon_cnt
    feature = pd.merge(feature, pivot[['Merchant_id', 'merchant_receive_rate']], on='Merchant_id', how='left')

    # 商家被不同用户领取的次数
    receive_differ_user_cnt = pd.pivot_table(data, values='User_id', index=keys, aggfunc=len)
    receive_differ_user_cnt = pd.DataFrame(receive_differ_user_cnt).rename(columns={'User_id': 'receive_differ_user_cnt'}).reset_index()
    feature = pd.merge(feature, receive_differ_user_cnt, on=keys, how='left')
    feature.fillna(0, downcast='infer')

    # 商家有多少种不同的优惠券
    differ_coupon_cnt = pd.pivot_table(data, values='Coupon_id', index=keys, aggfunc=len)
    differ_coupon_cnt = pd.DataFrame(differ_coupon_cnt).rename(columns={'Coupon_id': 'differ_coupon_cnt'}).reset_index()
    feature = pd.merge(feature, differ_coupon_cnt, on=keys, how='left')
    feature.fillna(0, downcast='infer')

    return feature

def get_coupon_feature(label_field):
    data=label_field.copy()
    feature=feature_prepare(data).copy()

    keys = ['Coupon_id']
    # 优惠券被领取的次数
    coupon_receive_cnt = pd.pivot_table(data, values='cnt', index=keys, aggfunc=len)
    coupon_receive_cnt = pd.DataFrame(coupon_receive_cnt).rename(columns={'cnt': 'coupon_receive_cnt'}).reset_index()
    feature = pd.merge(feature, coupon_receive_cnt, on=keys, how='left')
    feature.fillna(0, downcast='infer')

    # 优惠券满减类型的中位最低消费值
    pivot = pd.pivot_table(data[data['is_manjian'] == 1], values='min_cost_of_manjian', index=keys, aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(columns={'min_cost_of_manjian': 'manjian_median_min_price'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature.fillna(0, downcast='infer')

    # 用户-优惠券历史交易次数
    keys=['User_id', 'Coupon_id']
    pivot = data.groupby(keys)['cnt'].count().reset_index()
    pivot.columns = [keys[0], keys[1], 'user_coupon_hist_cnt']
    feature = pd.merge(feature, pivot, on=keys, how='left')

    return feature

def get_label(dataset):
    data = dataset.copy()
    data['label'] = list(map(lambda x, y: 1 if (x-y).total_seconds()/(60*60*24) <= 15 else 0, data['date'],data['date_received']))
    return data

# 构造数据集
def get_dataset(dataset):
    #提取特征
    data_feature = get_user_feature(dataset)
    data_feature = get_user_merchant_feature(data_feature)
    data_feature = get_merchant_feature(data_feature)
    data=get_coupon_feature(data_feature)

    if 'Date' in data.columns.tolist():
        data.drop(['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date'], axis=1, inplace=True)
        dataset = data['label'].tolist()
        data.drop(['label'], axis=1, inplace=True)
        data['label'] = dataset
    else:
        data.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)

    # 修正数据类型
    if 'label' in data.columns.tolist():
        data['label'] = data['label'].map(lambda x: int(x) if x == x else 0)

    data.drop_duplicates(keep='first')
    data.index = range(len(dataset))

    return data

def xgboost_model(train, test):

    data_train = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    data_test = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))

    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1,
              'gamma': 0.3,
              'lambda': 1,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.9,
              'scale_pos_weight': 1}

    #训练
    train_list = [(data_train, 'train')]
    model = xgb.train(params, data_train, num_boost_round=2000, evals=train_list)
    # 预测
    predict = model.predict(data_test)
    # 结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    # 特征重要性
    importance = pd.DataFrame(columns=['Users_feature', 'importance'])
    importance['Users_feature'] = model.get_score().keys()
    importance['importance'] = model.get_score().values()
    importance.sort_values(['importance'], ascending=False, inplace=True)
    return result, importance

if __name__ == '__main__':
    #读入数据
    print('读入数据中')
    train_data = pd.read_csv(r'./dataset/ccf_offline_stage1_train.csv')
    test_data = pd.read_csv(r'./dataset/ccf_offline_stage1_test_revised.csv')
    # 预处理
    print('预处理中')
    train_data =train_prepare(train_data)
    test_data =data_prepare(test_data)
    # 打标
    print('打标')
    train_data=get_label(train_data)
    #划分数据集区间
    print('划分数据区间中')
    train_field = train_data[train_data['date_received'].isin(pd.date_range('2016/2/20', periods=60))]
    verification_field = train_data[train_data['date_received'].isin(pd.date_range('2016/4/20', periods=60))]
    test_field = test_data.copy()
    # 构造训练集，测试集，验证集
    print('构造训练集中')
    train_data = get_dataset(train_field)
    print('构造验证集中')
    verification_data = get_dataset(verification_field)
    print('构造测试集中')
    test_data = get_dataset(test_field)
    #训练集
    train = pd.concat([train_data, verification_data], axis=0)
    result, importance = xgboost_model(train_data, test_data)
    result.to_csv(r"./data.csv", index=False, header=None)
    print(importance)

