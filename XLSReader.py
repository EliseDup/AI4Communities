# This is a sample Python script.
import statistics
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import statsmodels.api as sm


# Read xls file with consumption profile
# Header : Date;Heure;Conso(Wh);Index(Wh);Tension(mV);Courant(mA)
# Row example : 2020-02-06;10:36;;13;228833;252;CPU 60.4 C;

def read_donneeconso(name):
    headers = ["day_str", "hour_str", "ind"]
    dtypes = {'day_str': 'str', 'hour_str': 'str', 'ind': 'float'}
    # parse_dates = ['day']
    data = pd.read_csv(name, sep=";", dtype=dtypes, header=None, names=headers, usecols=[0, 1, 3],
                       skiprows=[0, 1, 2, 3, 4])  # , parse_dates=parse_dates) #, nrows=20)

    # Combine day and hour in a single date time
    date_format_str = "%Y-%m-%d,%H:%M"
    # drow row where date day is nan
    data.dropna(subset=['day_str', 'hour_str'], inplace=True)
    data['date'] = [datetime.strptime(data.day_str[i] + "," + data.hour_str[i], date_format_str) for i in data.index]
    # Consumption t = index(t+1) - index(t)
    # conso = [data.ind[i + 1] - data.ind[i] for i in range(0, n - 1)]
    # data['conso'] = conso + [conso[n - 2]]
    # data['power'] = data['conso'] / 0.25  # kW instead of kWh
    # data['month_int'] = [data.date[i].month for i in range(0, n)]
    # data['day_int'] = [data.date[i].day for i in range(0, n)]
    # data['hour_int'] = [data.date[i].hour for i in range(0, n)]
    return data


# Transform data set for quarter-hourly to hourly
def aggr_mean_profile(data, freq):
    aggr_data = data.groupby(pd.Grouper(key="date", freq=freq)).mean()
    return aggr_data


def aggr_sum_profile(data, freq):
    aggr_data = data.groupby(pd.Grouper(key="date", freq=freq)).sum()
    return aggr_data


def aggr_last_profile(data, freq):
    aggr_data = data.groupby(pd.Grouper(key="date", freq=freq)).last()
    return aggr_data


# Add consumption to a dataframe with index (i.e. meter reading)
def add_conso(data):
    n = data.index.size
    conso = [data.ind[i + 1] - data.ind[i] for i in range(0, n - 1)]
    data['conso'] = conso + [conso[n - 2]]
    return data


# Date	UT time	Temperature	Relative Humidity	Pressure	Wind speed	Wind direction	Rainfall	Snowfall	Snow depth	Short-wave irradiation,
# 1/06/18	00:15	290.45	86.03	1010.08	0.76	201.63	0.007092	0.000000	0.000000	0.0000,
# Problem : midnigh is expressed as 24:00 from previous day !
def read_meteo(name):
    headers = ["day_str", "hour_str", "temp_K", "wind_speed", "humidity", "irradiation_str"]
    dtypes = {'day_str': 'str', 'hour_str': 'str', 'temp': 'float', 'wind_speed': 'float', 'humidity': 'float',
              'irradiation_str': 'str'}
    data = pd.read_csv(name, sep=";", dtype=dtypes, header=None, names=headers, usecols=[0, 1, 2, 3, 5, 10],
                       skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                 24, 25])  # , nrows=100)  # , parse_dates=parse_dates) #, nrows=20)
    n: int = data.day_str.size
    date_format_str = "%Y-%m-%d,%H:%M"

    data['date_str'] = [data.day_str[i] + "," + data.hour_str[i] for i in range(0, n)]
    # Problem : day midnight is previous day + 24:00
    data['date_str'] = data['date_str'].str.replace('24:00', '00:00')
    # Combine day and hour in a single date time
    data['date'] = [datetime.strptime(data.date_str[i], date_format_str) for i in range(0, n)]
    # Add 1 day to date with hour = midnight !
    data['date'] = [midnight_correction(data.date[i]) for i in range(0, n)]
    data['irradiation'] = data['irradiation_str'].str.replace(',', '').astype(float)
    data['temp'] = data['temp_K'] - 273.15
    return data


def midnight_correction(day):
    if day.hour == 0 and day.minute == 0:
        return day + timedelta(days=1)
    else:
        return day


def add_meteo(data, meteo):
    return pd.merge(data, meteo, on="date")


def data_analysis(name, freq):
    meteo = read_meteo("data/production/SoDa_MERRA2_lat50.606_lon3.388_2018-06-01_2021-06-30_1590683819.csv")
    data = read_donneeconso(name)
    meteo = aggr_mean_profile(meteo, freq)
    data = aggr_last_profile(data, freq)
    data = add_meteo(data, meteo)
    data = add_conso(data)
    # Add hour day month index
    n = data.index.size

    data['month_int'] = [data.index[i].month for i in range(0, n)]
    data['day_int'] = [data.index[i].weekday() for i in range(0, n)]
    data['hour_int'] = [data.index[i].hour for i in range(0, n)]
    data['prev_conso'] = [data.conso[max(i - 1, 0)] for i in range(0, n)]

    max_h = [max(data[data['hour_int'] == i].conso) for i in range(0, 24)]
    max_d = [max(data[data['day_int'] == i].conso) for i in range(0, 7)]
    min_h = [min(data[data['hour_int'] == i].conso) for i in range(0, 24)]
    min_d = [min(data[data['day_int'] == i].conso) for i in range(0, 7)]
    median_h = [statistics.median(data[data['hour_int'] == i].conso) for i in range(0, 24)]
    median_d = [statistics.median(data[data['day_int'] == i].conso) for i in range(0, 7)]

    data['max_hour'] = [max_h[data.index[i].hour] for i in range(0, n)]
    data['min_hour'] = [min_h[data.index[i].hour] for i in range(0, n)]
    data['max_day'] = [max_d[data.index[i].weekday()] for i in range(0, n)]
    data['min_day'] = [min_d[data.index[i].weekday()] for i in range(0, n)]
    data['median_hour'] = [median_h[data.index[i].hour] for i in range(0, n)]
    data['median_day'] = [median_d[data.index[i].weekday()] for i in range(0, n)]
    print("Day")
    for i in range(0, 6):
        print(i, "\t", max_d[i], "\t", min_d[i], "\t", median_d[i])
    print("Hour")
    for i in range(0, 24):
        print(i, "\t", max_h[i], "\t", min_h[i], "\t", median_h[i])

    method = "kendall"
    # print(data.corr())
    print(name, "\t", data['prev_conso'].corr(data['conso'], method=method), "\t",
          data['month_int'].corr(data['conso'], method=method), "\t",
          data['day_int'].corr(data['conso'], method=method), "\t",
          data['hour_int'].corr(data['conso'], method=method), "\t", data['temp'].corr(data['conso'], method=method),
          "\t",
          data['irradiation'].corr(data['conso'], method=method), "\t",
          data['wind_speed'].corr(data['conso'], method=method), "\t",
          data['humidity'].corr(data['conso'], method=method), "\t",
          data['min_day'].corr(data['conso'], method=method), "\t",
          data['max_day'].corr(data['conso'], method=method), "\t",
          data['median_day'].corr(data['conso'], method=method), "\t",
          data['min_hour'].corr(data['conso'], method=method), "\t",
          data['max_hour'].corr(data['conso'], method=method), "\t",
          data['median_hour'].corr(data['conso'], method=method))

    return data


def box_plot_day(data):
    # Add min, max and range for each hour and day
    day_data = [data[data['day_int'] == i].conso for i in range(0, 7)]
    plt.boxplot(day_data)
    plt.show()


def box_plot_hour(data):
    hour_data = [data[data['hour_int'] == i].conso for i in range(0, 24)]
    plt.boxplot(hour_data)
    plt.show()


def prediction_model(data):
    X = data[['prev_conso', 'temp', 'irradiation']]  # data[['prev_conso', 'temp', 'irradiation']]
    y = data.conso
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    predictions = model.predict(X)
    plt.plot(data.index, data.conso)
    plt.plot(data.index, predictions)
    plt.show()
    print(model.summary())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    for i in range(1, 10):
        name = "data/consumptions/donneeconso0" + str(i) + ".csv"
        data = data_analysis(name, "1H")
        # box_plot_hour(data)
        # prediction_model(data)
    # for i in range(10, 17):
    #    name = "data/consumptions/donneeconso" + str(i) + ".csv"
    #    data_analysis(name, "1H")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
