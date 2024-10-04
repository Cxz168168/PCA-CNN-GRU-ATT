import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import os
import inspect
from model.Transformer import Model as model1
from model.PCA_LSTM import Model as model2
from model.PCA_CNN_LSTM_ATT import Model as model3
from model.noPCA_LSTM import Model as model4
from model.PCA_CNN_GRU_ATT import Model as model5
def split_data(data, timestep, input_size):
    dataX = []
    dataY = []
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep].values)
        dataY.append(data['cost'][index + timestep])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    train_size = int(np.round(0.8 * dataX.shape[0]))
    x_train = dataX[: train_size, :].reshape(-1, timestep, input_size)
    y_train = dataY[: train_size].reshape(-1, 1)
    x_test = dataX[train_size:, :].reshape(-1, timestep, input_size)
    y_test = dataY[train_size:].reshape(-1, 1)
    return [x_train, y_train, x_test, y_test, train_size]
def if_PCA(no_PCA_path, PCA_path, model):
    '''
    根据模型选择读取的数据类型
        no_PCA_path: no_PCA数据集地址
        PCA_path: PCA数据集地址
        model: 模型
        return: 数据，特征，标签
    '''
    model_filename = inspect.getfile(model)
    file_name = os.path.basename(model_filename)
    print(f'导入模型文件为: {file_name}')
    if file_name[:2] == 'no':
        data = pd.read_excel(no_PCA_path)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')

        dates = data['date'].values
        features = data[['cost', 'y1', 'y2', 'y3', 'y4', 'y5']]
        target = data['cost']
        input_size = 6
    else:
        data = pd.read_excel(PCA_path)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        dates = data['date'].values
        features = data[['cost', 'y1', 'y2']]
        target = data['cost']
        input_size = 3
    return file_name[:-3], dates, features, target, input_size

def main(model):

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    no_PCA_path = r'data\no_PCA.xlsx'
    PCA_path = r'data\PCA.xlsx'

    timestep = 5

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)



    model_name, dates, features, target, input_size = if_PCA(no_PCA_path, PCA_path, model)
    print(model_name)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    x_train, y_train, x_test, y_test, train_size = split_data(features, timestep, input_size)
    print(x_train.shape)

    x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
    y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
    x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
    y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    test_data = TensorDataset(x_test_tensor, y_test_tensor)
    batch_size = 24

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size,
                                               False)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size,
                                              False)

    model = model()
    model_path = (f'result/{model_name}_prediction_model.pth')
    load_model = torch.load(model_path)
    model.load_state_dict(load_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())

    predictions = np.concatenate(predictions)
    y_test_raw = np.concatenate(actuals)

    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions))
    mape = np.mean(np.abs((y_test_raw - predictions) / y_test_raw))
    r2 = r2_score(y_test_raw, predictions)
    return [predictions, y_test_raw, dates, train_size,[rmse,mape,r2]]


def draw_all(predictions, y_test_raw, dates, train_size):
    timestep = 5
    plt.rcParams['font.family'] = ['Times New Roman', 'STSong']
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'STSong']
    plt.rcParams['font.weight'] = 'bold'


    plt.figure(figsize=(9, 7), dpi=300)
    plt.plot(dates[train_size + timestep:], y_test_raw, label='Actual', linestyle='--', color='black', linewidth=2)
    plt.plot(dates[train_size + timestep:], predictions[0], label='Transformer', color='tab:pink', linestyle='-',
             linewidth=1)
    plt.plot(dates[train_size + timestep:], predictions[1], label='PCA-LSTM', color='tab:olive', linestyle='-',
             linewidth=1)
    plt.plot(dates[train_size + timestep:], predictions[2], label='PCA-CNN-LSTM-ATT', color='tab:green',
             linestyle='-', linewidth=1)
    plt.plot(dates[train_size + timestep:], predictions[3], label='LSTM', color='tab:cyan', linestyle='-',
             linewidth=1)
    plt.plot(dates[train_size + timestep:], predictions[4], label='PCA-CNN-GRU-ATT', color='purple', linestyle='-',
             linewidth=1)
    plt.xlabel('Date', fontsize=24, fontweight='bold')
    plt.ylabel('Bell pepper price (yuan)', fontsize=24, fontweight='bold')
    plt.legend(fontsize=16, frameon=True, framealpha=1, bbox_to_anchor=(0.65, 0.6))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=-45, fontsize=20, fontweight='bold')
    plt.tick_params(labelsize=20, width=2)
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gcf().autofmt_xdate()
    plt.tick_params(axis='both', direction='in')
    plt.show()

if __name__ == '__main__':
    predictions_list = []
    model_list = [model1, model2, model3, model4, model5]
    indicators_list = []
    for model in model_list:
        predictions, y_test_raw, dates, train_size,indicators = main(model)
        predictions_list.append(predictions)
        indicators_list.append(indicators)
    print(indicators_list)
    draw_all(predictions_list, y_test_raw, dates, train_size)