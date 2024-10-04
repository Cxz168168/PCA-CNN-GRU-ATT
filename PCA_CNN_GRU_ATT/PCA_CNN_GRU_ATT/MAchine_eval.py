from Machine_train import predictions_dict, test_dates, y_test
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from eval import main
from model.PCA_CNN_GRU_ATT import Model as model

colors = {
    'KNN': 'tab:cyan',
    'SVR': 'blue',
    'RandomForest': 'tab:green',
    'XGBoost': 'tab:olive',
    'LightGBM': 'purple',
    "PCA-CNN-GRU-ATT":'tab:pink'
}
predictions_dict["PCA-CNN-GRU-ATT"] = main(model)[0]


plt.rcParams['font.family'] = ['Times New Roman', 'STSong']
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'STSong']
plt.rcParams['font.weight'] = 'bold'

for model_name, predictions in predictions_dict.items():
    print(predictions.shape)
    plt.plot(test_dates, predictions, label=f'{model_name}', color=colors.get(model_name, 'gray'), linestyle='-', linewidth=1.5)

plt.xlabel('Date', fontsize=24, fontweight='bold')
plt.ylabel('Bell pepper price (yuan)', fontsize=24, fontweight='bold')
plt.legend(fontsize=15, frameon=True, framealpha=1, bbox_to_anchor=(0.75, 0.59))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=-45, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.tick_params(labelsize=20, width=2)
plt.grid(False)  # 关闭网格线
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gcf().autofmt_xdate()
plt.tick_params(axis='both', direction='in')
plt.show()