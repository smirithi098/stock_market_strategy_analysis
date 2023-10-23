import requests
import pandas as pd
from datetime import datetime, date, timedelta
from io import BytesIO


class NSE:
    def __init__(self, timeout=10):
        self.base_url = 'https://www.nseindia.com'
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.55",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "accept-language": "en-US,en;q=0.9"
        }
        self.timeout = timeout
        self.cookies = []

    def __get_cookies(self, renew=False):
        if len(self.cookies) > 0 and renew is False:
            return self.cookies

        r = requests.get(self.base_url, timeout=self.timeout, headers=self.headers)
        self.cookies = dict(r.cookies)
        return self.__get_cookies()

    def get_historical_data(self, symbol, series, start_year, current_date):
        try:
            list_df = []

            for yr in range(start_year, current_date.year+1):
                from_date = date(yr, 1, 1)

                if from_date.year == current_date.year:
                    to_date = current_date - timedelta(days=1)
                else:
                    to_date = date(yr, 12, 31)

                url = "/api/historical/cm/equity?symbol={0}&series=[%22{1}%22]&from={2}&to={3}&csv=true".format(
                    symbol.replace('&', '%26'), series, from_date.strftime('%d-%m-%Y'), to_date.strftime('%d-%m-%Y'))
                r = requests.get(self.base_url + url, headers=self.headers, timeout=self.timeout,
                                 cookies=self.__get_cookies())
                if r.status_code != 200:
                    r = requests.get(self.base_url + url, headers=self.headers, timeout=self.timeout,
                                     cookies=self.__get_cookies(True))

                list_df.append(pd.read_csv(BytesIO(r.content), sep=',', thousands=',', index_col=0, parse_dates=[0]))

            final_df = pd.concat(list_df)
            final_df = final_df.sort_index()

            final_df.to_csv(f"S:/Dissertation 2023/Stock market analysis/data_files/{symbol.lower()}.csv",
                            sep=',', header=True, index=True)

        except:
            return None

    def get_current_data(self, symbol, series, current_date, file_path):
        try:
            url = "/api/historical/cm/equity?symbol={0}&series=[%22{1}%22]&from={2}&csv=true".format(
                symbol.replace('&', '%26'), series, current_date.strftime('%d-%m-%Y'))

            r = requests.get(self.base_url + url, headers=self.headers, timeout=self.timeout,
                             cookies=self.__get_cookies())

            if r.status_code != 200:
                r = requests.get(self.base_url + url, headers=self.headers, timeout=self.timeout,
                                 cookies=self.__get_cookies(True))

            current_data = pd.read_csv(BytesIO(r.content), sep=',', thousands=',', index_col=0, parse_dates=[0])

            current_data.to_csv(file_path, mode="a", sep=',', index=True, header=False)

        except:
            return None


if __name__ == '__main__':
    from datetime import date
    from nse import NSE

    stock_list = ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK']

    stock_series = "EQ"
    year = 2006
    today_date = date.today()

    nse = NSE()
    for stock_symbol in stock_list:
        nse.get_historical_data(symbol=stock_symbol, series=stock_series, start_year=year, current_date=today_date)
