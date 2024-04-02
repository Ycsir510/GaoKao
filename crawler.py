import csv
import requests
import os


class crawler:
    def __init__(self, appKey):
        self.appKey = appKey

    def save_data(self, data):
        with open('./高考志愿1.csv', encoding='UTF-8', mode='a+', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(data)
        f.close()


    def get_data(self):
        # 添加表头

        head = ['省份', '年份', '学校名称', '专业', '最低录取分', '最低录取名次', '选课要求']
        # 清除已存在的同名文件
        v_file = '高考志愿1.csv'
        if os.path.exists(v_file):
            os.remove(v_file)
            print('高考志愿存在，已清除:{}'.format(v_file))

        with open('./高考志愿1.csv', encoding='utf-8-sig', mode='w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(head)
            f.close()

        url = 'https://www.ayshuju.com/data/edu/specialline'
        headers = {"content-type": "application/json;charset=UTF-8"}

        param = {"appKey": self.appKey, "schoolName": "华中师范大学", "provinceName": "浙江", "pageNo": 1, "pageSize": 50}

        for m in range(2017, 2024):
            param['year'] = str(m)
            try:
                response = requests.post(url, json=param, headers=headers)
                response.raise_for_status()     # 调用 response.raise_for_status() 方法来确保请求的成功完成，以及在请求失败时及时处理异常情况，例如记录日志、重新尝试请求或采取其他必要的措施。
                data = response.json()['result']['records']
                print(data)
                for item in data:
                    # record = (item['provinceName'], item['year'], item['schoolName'],
                    #           item['min'], item['batchName'], item['zslxName'],
                    #           item['minSection'], item['sgName'], item['sgInfo'],
                    #          item['average'], item['spName'])
                    record = (item['provinceName'], item['year'], item['schoolName'],
                              item['spName'], item['min'], item['minSection'])

                    self.save_data(record)
            except Exception as e:
                print(f'An error occurred: {e}')


if __name__ == '__main__':
    appKey = 'U8L47j0v'
    my_crawler = crawler(appKey)
    my_crawler.get_data()
