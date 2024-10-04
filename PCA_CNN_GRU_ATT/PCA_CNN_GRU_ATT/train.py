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

from model.PCA_CNN_GRU_ATT import Model as model
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
    print(f'the file of the input model is: {file_name}')
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


if __name__ == '__main__':


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

    print(y_test)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.3)
    num_epochs = 400
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    plt.rcParams['font.family'] = ['Times New Roman', 'STSong']
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'STSong']


    plt.figure(figsize=(8, 8))
    plt.plot(test_losses, color='black', linewidth=1)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Test set loss', fontsize=20)
    plt.grid(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()


    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            # print('input:',inputs,'\nlabls', labels)
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())

    predictions = np.concatenate(predictions)
    y_test_raw = np.concatenate(actuals)

    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions))
    mape = np.mean(np.abs((y_test_raw - predictions) / y_test_raw)) * 100
    r2 = r2_score(y_test_raw, predictions)

    print(f'RMSE: {rmse:.2f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'R^2: {r2:.4f}')

    plt.rcParams['font.family'] = ['Times New Roman', 'STSong']
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'STSong']


    plt.figure(figsize=(10, 12))
    plt.plot(dates[train_size + timestep:], y_test_raw, label='Actual', color='black', linewidth=2)
    plt.plot(dates[train_size + timestep:], predictions, label='Predicted', color='deeppink', linestyle='--', linewidth=2)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Bell pepper price (yuan)', fontsize=20)
    plt.legend(fontsize=20, frameon=False)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=-45)
    plt.tick_params(labelsize=20)
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()


