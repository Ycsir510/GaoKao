import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 标准化和归一化
from sklearn.model_selection import GridSearchCV

my_font = font_manager.FontProperties(fname="C:/WINDOWS/Fonts/msyh.ttc")
plt.rcParams['font.sans-serif'] = ['SimHei']

# 该函数将把近六年都招生的专业和相应分数线与录取位次筛选出来，共21个专业
class analyse:
    def __init__(self, conf: dict):
        pd.set_option('display.max_rows', None)  # 调整 Pandas 在数据显示和处理过程中的行为; 这一句话确保所有行数据都能被显示出来
        self.conf = conf
        self.data = None
        self.data = pd.read_csv('高考志愿1.csv', encoding='UTF-8')
        self.subj_name = []
        self.axis_scores = []  # 每年的分数线
        self.axis_ranks = []  # 每年的最低录取位次
        self.axis_year = np.arange(2017, 2023)  # np.arange(2017, 2023) 是 NumPy 中的函数，用于创建一个一维数组，包含从 2017 (包含) 开始到 2023 (不包含) 结束的整数序列。

        # 文理全部的索引和简称
        self.subj_index1 = []
        self.subj_alias1 = []

    def process(self):
        # step1:删除无用信息
        self.data = self.data.drop('省份', axis=1)
        self.data = self.data.drop('学校名称', axis=1)
        self.data = self.data.drop('选课要求', axis=1)
        self.data['专业'] = self.data['专业'].apply(lambda x: re.sub(r'\（.*?\）', '', x))
        # print(self.data)

        # step2:获得所有招生的专业
        grouped = self.data.groupby('年份')  # 根据年份分组
        yr_of_subj = grouped.get_group(2017)['专业']             # 从分组中取出 2017 年的 ‘专业’ 数据作为初始的 yr_of_subj 数据。
        for y in range(2018, 2023):
            tmp = grouped.get_group(y)['专业']                   # 从分组中取出当前年份 y 的 ‘专业’ 数据
            tmp.reset_index(drop=True, inplace=True)
            yr_of_subj = pd.concat([yr_of_subj, tmp], axis=1)   # 将当前年份 y 的 ‘专业’ 数据与 yr_of_subj 进行列方向的拼接。
        yr_of_subj.index = range(1, self.conf['max_subj'] + 1)  # 修改行名为 1 到最大专业数（max_subj）。
        yr_of_subj.columns = range(2017, 2023)                  # 修改列名为 2017 到 2022
        print('每年招生的专业：')
        print(yr_of_subj)

        # step3:提取出每年都招收的专业
        self.subj_name = list(set(yr_of_subj[yr_of_subj.columns[0]]).intersection(
            *[set(yr_of_subj[col]) for col in yr_of_subj.columns]))
        self.subj_name.sort()
        print('筛选得到专业：', self.subj_name)

        score = {}   # 初始化空字典
        rank = {}
        for (_, subj) in enumerate(self.subj_name):   # 遍历 subj_name 中的所有元素即各个专业
            score[subj] = [self.data.iloc[index, 2] for index in range(self.conf['total_subj'])    # iloc 是一种用于按位置（integer location）选择数据的方法。iloc 可以根据行和列的位置来访问数据，而不是根据标签或索引。访问 DataFrame 中第 index 行、第 2 列的数据
                           if self.data.iloc[index, 1] == subj]
            rank[subj] = [self.data.iloc[index, 3] for index in range(self.conf['total_subj'])
                          if self.data.iloc[index, 1] == subj]
        print('专业分数线：', score)
        print('专业录取位次', rank)
        self.axis_scores = np.array(list(score.values()), dtype=object)  # 转化为 numpy 数组
        self.axis_ranks = np.array(list(rank.values()), dtype=object)

        self.subj_index1 = np.array([0, 1])
        self.subj_alias1 = np.array([ '中国语言文学类', '心理学类'])


    def predict(self):  # 采用支持向量回归对2023年这十个专业的录取位次进行预测
        years = np.array([2017, 2018, 2019, 2020, 2021, 2022], dtype=np.float32)

        fig = plt.figure()                # 创建一个新的图形（Figure）对象。
        ax = fig.add_subplot(1, 1, 1)     # 在创建的图形中添加一个子图（subplot），这里是添加一个1x1的子图，并选择第1个位置。
        x = np.append(years, 2023)        # 将之前创建的years数组中的数据和数值2023合并为新的数组x
        dic_ranks = {}                    # 创建一个空字典dic_ranks，以用于存储后续的数据
        for i in range(2): # 连续5年招生的专业数
            ranks = np.array(self.axis_ranks[self.subj_index1[i]], dtype=np.float32)

            # 创建输入数据和标签
            input_data = years.reshape(-1, 1)  # 作用是将years数组变形为一个包含一列的二维数组，其中的-1表示由NumPy自动计算该维度的大小，而1表示该维度只包含一个元素
                                               # input_data可能会被用作支持向量回归（SVR）模型的输入，其中模型可能期望输入是一个包含单个特征的二维数组。
            labels = ranks

            # 创建支持向量回归模型实例
            svr_model = SVR(kernel='rbf')

            # 设置超参数
            C = [0.1, 0.2, 0.5, 0.8, 0.9, 1, 2, 5, 10]
            kernel = 'rbf'
            gamma = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8]
            epsilon = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
            # 参数字典
            params_dict = {
                'C': C,
                'gamma': gamma,
                'epsilon': epsilon
            }
            grid_search = GridSearchCV(svr_model, params_dict, cv=3)  # k折交叉验证
            grid_search.fit(input_data, labels)
            print("最佳度量值:", grid_search.best_score_)
            print("最佳参数:", grid_search.best_params_)
            print("最佳模型:", grid_search.best_estimator_)
            best_params = grid_search.best_params_
            svr_model = SVR(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], epsilon=best_params['epsilon'])


            # 训练模型
            svr_model.fit(input_data, labels)

            test_data = np.array([[2023]], dtype=np.float32)
            predicted_ranks = svr_model.predict(test_data)
            dic_ranks[self.subj_alias1[i]] = round(predicted_ranks[0])  # 四舍五入录取位次

            # # 可视化结果
            # y = np.append(ranks, predicted_ranks)
            # ax.plot(x, y, linewidth=1.0, label=self.subj_alias1[i])
            # ax.invert_yaxis()
            # ax.legend(loc='best', ncol=3)
            # plt.xlabel('年份')
            # plt.ylabel('录取位次')

        for i in range(2):
            print(self.subj_alias1[i] + ': ' + str(dic_ranks[self.subj_alias1[i]]))
        plt.show()

if __name__ == '__main__':
    conf = {'max_subj': 26, 'total_subj': 76}      # 参数看爬下来的数据中 csv文件中总行数（total_subj）以及总共的专业数（max_subj）
    my_analyse = analyse(conf)
    my_analyse.process()
    my_analyse.predict()



