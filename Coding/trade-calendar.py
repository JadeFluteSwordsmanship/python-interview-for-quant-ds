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
        self.calendar_df = pd.read_csv(csvFile, parse_dates=["date"])
        self.calendar_df.sort_values("date", inplace=True)

        self.trading_dates = list(self.calendar_df[self.calendar_df["isTrading"] == 1]["date"].dt.strftime('%Y-%m-%d'))

    def isTrading(self, calendar_date):
        """
        isTrading return if calendar date is is trading date or not
        :param calendar_date: YYYY-MM-DD format date e.g. "2011-01-03"
        :return: true or false
        """
        return calendar_date in self.trading_dates

    def getPrevTradingDate(self, calendar_date):
        """
        getPrevTradingDate return the previous trading date before calendar_date
        :param calendar_date: YYYY-MM-DD format date e.g. "2011-01-03"
        :return: previous trading date,YYYY-MM-DD format
        """
        for trading_date in reversed(self.trading_dates):
            if trading_date < calendar_date:
                return trading_date
        return None

    def getNextTradingDate(self, calendar_date):
        """
        getNextTradingDate return the next trading date after calendar_date
        :param calendar_date: YYYY-MM-DD format date e.g. "2011-01-03"
        :return: next trading date,YYYY-MM-DD format
        """
        for trading_date in self.trading_dates:
            if trading_date > calendar_date:
                return trading_date
        return None

if __name__ == "__main__":
    calendar = Calendar("./data/calendar.csv")
    print(calendar.isTrading("2011-01-03"))
    print(calendar.getPrevTradingDate("2011-01-08"))
    print(calendar.getNextTradingDate("2011-01-04"))
    # print(calendar.getNextTradingDate("2011-01-07"))

