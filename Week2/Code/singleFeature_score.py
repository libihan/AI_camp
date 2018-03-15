import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 比较单个特征和score的相关性

def autolabel(rects):
    """
    显示数值
    :param rects:
    :return:
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()-rect.get_width(), 1.03*height, '%s' % round(height,2))

if __name__ == '__main__':
    # 读取数据
    train_data = pd.read_csv(r'C:\Users\XPS13\Desktop\0206\AI_Camp\Feature\train_data.csv')

    # 载入train和test数据集
    dataSet = train_data.iloc[:, :-3]
    labelSet = train_data.iloc[:,-3:]

    y_label = []
    for i in range(dataSet.shape[1]):
        y_label.append(pearsonr(list(labelSet.iloc[:, -1]), list(dataSet.iloc[:, i]))[0])

    plt.figure(figsize=(70, 20))
    # plt.xticks(range(len(y_label)),dataSet.columns,size='small',rotation=90)
    rect = plt.bar(range(len(y_label)),y_label,width = 0.35)
    plt.ylim(-0.2, 0.9)

    autolabel(rect)
    plt.show()