# import libraries

from datetime import date
from nse import NSE
import glob

# initialise class object
cmp = NSE()

# define static variables
stock_series = 'EQ'
today_date = date.today()
path = "S:/Dissertation 2023/Stock market analysis/data_files/*.csv"

# call the method to get current market price
for fname in glob.glob(path):
    stock_symbol = fname.split('\\')[1].split('.')[0].upper()
    f_name = f"S:/Dissertation 2023/Stock market analysis/data_files/{stock_symbol.lower()}.csv"

    cmp.get_current_data(symbol=stock_symbol, series=stock_series, current_date=today_date, file_path=f_name)
