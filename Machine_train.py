import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import os
import inspect
import joblib


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


seed = 42
np.random.seed(seed)
random.seed(seed)

PCA_path = r'data\PCA.xlsx'
timestep = 5


dates, features, target, input_size = if_PCA(PCA_path)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)


assert len(dates) == scaled_features_df.shape[0], "dates Array length does not match the number of feature data rows！"


x_train, y_train, x_test, y_test, train_size = split_data(scaled_features_df, timestep, input_size * timestep)
print(f"Training data shape: {x_train.shape}, Training label shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Test label shape: {y_test.shape}")


models = {
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'SVR': SVR(kernel='rbf'),
    'RF': RandomForestRegressor(n_estimators=100, random_state=seed),
    'XGB': xgb.XGBRegressor(n_estimators=100, random_state=seed, objective='reg:squarederror'),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=seed)
}


predictions_dict = {}
metrics_dict = {}

os.makedirs('result', exist_ok=True)


for model_name, model in models.items():
    print(f"\nThe model is being trained.: {model_name}")

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)


    predictions_for_inverse = np.zeros((predictions.shape[0], input_size))
    predictions_for_inverse[:, 0] = predictions

    predictions_original = scaler.inverse_transform(predictions_for_inverse)[:, 0]

    predictions_dict[model_name] = predictions_original

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    try:
        mape = mean_absolute_percentage_error(y_test, predictions) * 100
    except:
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    r2 = r2_score(y_test, predictions)

    metrics_dict[model_name] = {'RMSE': rmse, 'MAPE': mape, 'R2': r2}

    print(f"{model_name} redult:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")

    joblib.dump(model, f'result/{model_name}_model.joblib')
    print(f"{model_name} The model has been saved to result/{model_name}_model.joblib")


plt.rcParams['font.family'] = ['Times New Roman', 'STSong']
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'STSong']

plt.figure(figsize=(12, 6),dpi=300)


test_dates = dates[train_size + timestep:]
y_test = features['cost'][train_size + timestep:]

plt.plot(test_dates, y_test, label='Actual', color='black',  linestyle='--',linewidth=2)


colors = {
    'KNN': 'deeppink',
    'SVR': 'blue',
    'RandomForest': 'green',
    'XGBoost': 'orange',
    'LightGBM': 'purple'
}


if __name__ == "__main__":
    for model_name, predictions in predictions_dict.items():
        plt.plot(test_dates, predictions, label=f'Predicted ({model_name})', color=colors.get(model_name, 'gray'), linestyle='--', linewidth=2)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Bell pepper price (yuan)', fontsize=14)
    plt.title('Result', fontsize=16)
    plt.legend(fontsize=12)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


    print("\nall models evaluation metrics :")
    for model_name, metrics in metrics_dict.items():
        print(f"{model_name}: RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%, R²={metrics['R2']:.4f}")
