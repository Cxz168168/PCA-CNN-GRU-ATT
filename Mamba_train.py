import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler

def split_data(data, timestep, input_size):
    dataX = []
    dataY = []

    for index in range(len(data) - timestep):
        window = data.iloc[index: index + timestep].values.flatten()
        dataX.append(window)
        dataY.append(data['cost'].iloc[index + timestep])

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    train_size = int(np.round(0.8 * dataX.shape[0]))
    x_train = dataX[:train_size]
    y_train = dataY[:train_size]
    x_test = dataX[train_size:]
    y_test = dataY[train_size:]

    return [x_train, y_train, x_test, y_test, train_size]

def if_PCA(PCA_path):
    '''
    读取数据并提取日期、特征和目标。
    '''
    data = pd.read_excel(PCA_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)

    dates = data['date'].values
    features = data[['cost', 'y1', 'y2']]
    target = data['cost']
    input_size = features.shape[1]  # 3
    return dates, features, target, input_size

if __name__ == '__main__':

    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    PCA_path = r'data\PCA.xlsx'
    timestep = 5


    dates, features, target, input_size = if_PCA(PCA_path)


    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

    assert len(dates) == scaled_features_df.shape[0], "dates 数组长度与特征数据行数不一致！"


    x_train, y_train, x_test, y_test, train_size = split_data(scaled_features_df, timestep, input_size * timestep)
    print(f"训练数据形状: {x_train.shape}, 训练标签形状: {y_train.shape}")
    print(f"测试数据形状: {x_test.shape}, 测试标签形状: {y_test.shape}")


    model = mamba.Model([
        {'type': 'Dense', 'units': 64, 'activation': 'relu'},
        {'type': 'Dense', 'units': 1}
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=400, batch_size=6)

    model.eval()
    predictions = model.predict(x_test)
