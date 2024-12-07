from Machine_train import predictions_dict, test_dates, y_test
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from eval import main
from model.PCA_CNN_GRU_ATT import Model as model

# 定义颜色字典
colors = {
    'KNN': 'tab:cyan',
    'SVR': 'blue',
    'RF': 'tab:green',
    'XGB': 'tab:olive',
    'LightGBM': 'purple',
    "PCA-CNN-GRU-ATT": 'red'
}
predictions_dict["PCA-CNN-GRU-ATT"] = main(model)[0]

# 设置字体样式
plt.rcParams['font.family'] = ['Times New Roman', 'STSong']
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'STSong']
plt.rcParams['font.weight'] = 'bold'

# 创建主图
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
plt.plot(test_dates, y_test, label='Actual', linestyle='--', color='black', linewidth=2)

# 绘制预测结果曲线
for model_name, predictions in predictions_dict.items():
    plt.plot(test_dates, predictions, label=f'{model_name}', color=colors.get(model_name, 'gray'), linestyle='-', linewidth=1.5)

# 设置坐标轴标签
plt.xlabel('Date', fontsize=24, fontweight='bold')
plt.ylabel('Bell pepper price (yuan)', fontsize=24, fontweight='bold')

# 设置图例
plt.legend(fontsize=15, frameon=True, framealpha=1, bbox_to_anchor=(0.76, 0.62))

# 格式化日期
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 设置x轴和y轴的刻度和样式
plt.xticks(rotation=-45, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.tick_params(labelsize=20, direction='in')  # 刻度线向内，移除 width 参数
plt.grid(False)  # 关闭网格线
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 自动调整日期格式
plt.gcf().autofmt_xdate()

# 创建局部放大图
ax_inset = inset_axes(ax, width=2.5, height=1.8, loc='center left',
                      bbox_to_anchor=(0.38, 0.95), bbox_transform=ax.transAxes, borderpad=0)

# 放大图的区间
zoom_start = 100  # 放大起始索引
zoom_end = 115    # 放大结束索引
plt.plot(test_dates, y_test, label='Actual', linestyle='--', color='black', linewidth=2)

# 在局部放大图中绘制预测结果
for model_name, predictions in predictions_dict.items():
    ax_inset.plot(test_dates, predictions, color=colors.get(model_name, 'gray'), linestyle='-', linewidth=1)

# 设置局部放大图的显示范围
ax_inset.set_xlim(test_dates[zoom_start], test_dates[zoom_end])
ax_inset.set_ylim(min(predictions_dict['PCA-CNN-GRU-ATT'][zoom_start:zoom_end]),
                  max(predictions_dict['PCA-CNN-GRU-ATT'][zoom_start:zoom_end]))

# 格式化局部放大图的日期，精确到日
ax_inset.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# 设置局部放大图的 x 轴刻度倾斜，但 y 轴不倾斜
ax_inset.tick_params(axis='x', labelrotation=45, labelsize=10, direction='in')  # x 轴倾斜，刻度线向内
ax_inset.tick_params(axis='y', labelrotation=0, labelsize=10, direction='in')  # y 轴不倾斜

# 设置局部放大图的网格线
ax_inset.grid(True)

# 添加主图和局部放大图之间的连接线
mark_inset(ax, ax_inset, loc1=3, loc2=4, fc="none", ec="gray")

# 显示图像
plt.show()
