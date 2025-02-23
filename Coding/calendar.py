"""
date,isTrading
2011-01-01,0
2011-01-02,0
2011-01-03,0
2011-01-04,1
2011-01-05,1
2011-01-06,1
2011-01-07,1
2011-01-08,0
....

given a ./data/calendar.csv which data like above, 'isTrading' column is 1 means
this date is trading date, while 0 means this date is not trading date, implement
the Calendar class please

"""
import pandas as pd

class Calendar:
    def __init__(self, csvFile="./data/calendar.csv"):
        """
        初始化 Calendar 类，加载交易日数据
        :param csvFile: CSV 文件路径，包含 date, isTrading 列
        """
        self.calendar_df = pd.read_csv(csvFile, parse_dates=["date"])
        self.calendar_df.sort_values("date", inplace=True)  # 确保日期有序

        # 仅存储交易日的数据，方便查询
        self.trading_dates = list(self.calendar_df[self.calendar_df["isTrading"] == 1]["date"].dt.strftime('%Y-%m-%d'))

    def isTrading(self, calendar_date):
        """
        判断指定日期是否为交易日
        :param calendar_date: 字符串格式 YYYY-MM-DD
        :return: True (交易日) / False (非交易日)
        """
        return calendar_date in self.trading_dates

    def getPrevTradingDate(self, calendar_date):
        """
        获取指定日期之前的最近一个交易日
        :param calendar_date: 字符串格式 YYYY-MM-DD
        :return: 前一个交易日 (YYYY-MM-DD) 或 None (无前一个交易日)
        """
        for trading_date in reversed(self.trading_dates):
            if trading_date < calendar_date:
                return trading_date
        return None

    def getNextTradingDate(self, calendar_date):
        """
        获取指定日期之后的最近一个交易日
        :param calendar_date: 字符串格式 YYYY-MM-DD
        :return: 下一个交易日 (YYYY-MM-DD) 或 None (无下一个交易日)
        """
        for trading_date in self.trading_dates:
            if trading_date > calendar_date:
                return trading_date
        return None

# 执行主程序
if __name__ == "__main__":
    calendar = Calendar("./data/calendar.csv")

    print("Is 2011-01-03 a trading day?", calendar.isTrading("2011-01-03"))  # False
    print("Previous trading day before 2011-01-08:", calendar.getPrevTradingDate("2011-01-08"))  # 2011-01-07
    print("Next trading day after 2011-01-04:", calendar.getNextTradingDate("2011-01-04"))  # 2011-01-05
