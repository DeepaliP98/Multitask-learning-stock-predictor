import datetime
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import sklearn.preprocessing


def validate(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        return 0
    return 1


def predictStock(company, date):
    if(company == 'Toyata'):
        df = pd.read_csv("/home/saksham/Documents/microblog/app/dataset/toyota_ui.csv")
    else:
        df = pd.read_csv("/home/saksham/Documents/microblog/app/dataset/gm_ui.csv")

    if df['Date'].str.contains(date).any():
        pass
    else:
        return -1

    result = list()
    result.append((df[df['Date'] == date]['Predicted'].iloc[0])[1:-2])
    result.append((df[df['Date'] == date]['Actual'].iloc[0])[1:-2])
    return result


def checkRange(date):
    check = date.split("-")

    if (int(check[0]) >= 2013) and (int(check[0]) <= 2019) and (int(check[1]) >= 2):
        return 1
    else:
        return 0


def graphPlot(companies, destination):
    df = pd.read_csv("/home/saksham/Documents/microblog/app/dataset/combined.csv")
    df_stock_data = df[['TSLA', 'Apple', 'GM', 'Toyata', 'Google']].copy()

    colors = ['red', 'yellow', 'green', 'blue', 'orange']

    netChange = list()

    for company in companies:
        netChange.append(df_stock_data[company].values.tolist())

    timeline = range(len(netChange[0]))

    plt.figure(figsize=(8, 4))
    for i in range(len(netChange)):
        plt.plot(timeline, netChange[i], color=colors[i], label=companies[i])
    plt.legend()
    plt.savefig(destination)
    return 1


def DTWDistance(company1,company2):
    df = pd.read_csv("/home/saksham/Documents/microblog/app/dataset/combined.csv")
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    company1_data = min_max_scaler.fit_transform(df[company1].values.reshape(-1,1))
    company2_data = min_max_scaler.fit_transform(df[company2].values.reshape(-1,1))
    distance,path = fastdtw(company1_data,company2_data)
    return distance

