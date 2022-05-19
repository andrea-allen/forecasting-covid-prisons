import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import statsmodels.tsa.stattools
import matplotlib
import wrangle_data
from statsmodels.tsa.stattools import adfuller
font = {'family' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)


def stationarity_check(df, column_name):
    print("Observations of Dickey-fuller test")
    dftest = adfuller(df[column_name].diff(1)[1:],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
    for key,value in dftest[4].items():
        dfoutput['critical value (%s)'%key]= value
    print(dfoutput)
    plt.figure(100)
    rmean1=df[column_name].diff(1).rolling(window=12).mean()
    rstd1=df[column_name].diff(1).rolling(window=12).std()
    print(rmean1,rstd1)
    orig=plt.plot(df[column_name].diff(1) , color='black',label='Original')
    mean= plt.plot(rmean1 , color='red',label='Rolling Mean')
    std=plt.plot(rstd1,color='blue',label = 'Rolling Standard Deviation')
    plt.legend(loc='best')
    plt.title("Rolling mean and standard deviation")
    # plt.show(block=False)

def ad_fuller_results(df, column_name, title=None, lag=False):
    print(f"Observations of Dickey-fuller test for {title}")
    if lag:
        dftest = adfuller(df[column_name].diff(1)[1:],autolag='AIC')
    else:
        dftest = adfuller(df[column_name], autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','#lags used','number of observations used'])
    dfoutput["Data"] = title
    for key,value in dftest[4].items():
        dfoutput['critical value (%s)'%key]= value
    print(dfoutput)
    return dfoutput

def acf_plot(df):
    fig, ax = plt.subplots(1,2)
    import statsmodels.api as sm
    sm.graphics.tsa.plot_acf(df["Residents.Confirmed"].diff(1)[1:].values.squeeze(), lags=40, ax=ax[0], label='Residents.Confirmed')
    sm.graphics.tsa.plot_acf(df["Staff.Confirmed"].diff(1)[1:].values.squeeze(), lags=40, ax=ax[1], label='Staff.Confirmed')
    ax[0].legend()
    ax[1].legend()

def rolling_corr_plots(df, ax):
    ax.plot(np.arange(len(df)), df['Residents.Confirmed'].diff(1).rolling(2).corr(df['Staff.Confirmed'].diff(1)), label=f'window=2', alpha=0.6)
    ax.plot(np.arange(len(df)), df['Residents.Confirmed'].diff(1).rolling(4).corr(df['Staff.Confirmed'].diff(1)), label=f'window=4', alpha=0.8)
    ax.plot(np.arange(len(df)), df['Residents.Confirmed'].diff(1).rolling(6).corr(df['Staff.Confirmed'].diff(1)), label=f'window=6', alpha=0.8)
    ax.plot(np.arange(len(df)), df['Residents.Confirmed'].diff(1).rolling(10).corr(df['Staff.Confirmed'].diff(1)), label=f'window=10', alpha=0.9)
    ax.plot(np.arange(len(df)), df['Residents.Confirmed'].diff(1).rolling(12).corr(df['Staff.Confirmed'].diff(1)), label=f'window=12', alpha=0.9)
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Starting week")
    ax.legend(frameon=True, loc="lower right")


def evaluate_stationarity(df):
    """
    Reports on the stationairty of raw and 1st-differenced time series for residents and staff, full timeseries
    :param df:
    :return:
    """
    ca_res_adf = ad_fuller_results(wrangle_data.get_california(df), "Residents.Confirmed",
                                              "CA Residents", lag=True)
    ca_staff_adf = ad_fuller_results(wrangle_data.get_california(df), "Staff.Confirmed", "CA Staff",
                                                lag=True)

    wa_res_adf = ad_fuller_results(wrangle_data.get_washington(df), "Residents.Confirmed",
                                              "WA Residents", lag=True)
    wa_staff_adf = ad_fuller_results(wrangle_data.get_washington(df), "Staff.Confirmed", "WA Staff",
                                                lag=True)

    adf_results = pd.DataFrame([ca_res_adf, ca_staff_adf, wa_res_adf, wa_staff_adf])
    # adf_results.to_csv("./results/adf_lagged.csv", index=False)
    adf_lagged_tex = adf_results.to_latex()
    print(adf_lagged_tex)

    ca_res_adf = ad_fuller_results(wrangle_data.get_california(df), "Residents.Confirmed",
                                              "CA Residents", lag=False)
    ca_staff_adf = ad_fuller_results(wrangle_data.get_california(df), "Staff.Confirmed", "CA Staff",
                                                lag=False)

    wa_res_adf = ad_fuller_results(wrangle_data.get_washington(df), "Residents.Confirmed",
                                              "WA Residents", lag=False)
    wa_staff_adf = ad_fuller_results(wrangle_data.get_washington(df), "Staff.Confirmed", "WA Staff",
                                                lag=False)

    adf_results = pd.DataFrame([ca_res_adf, ca_staff_adf, wa_res_adf, wa_staff_adf])
    # adf_results.to_csv("./results/adf_not_lagged.csv", index=False)
    adf_not_lagged_tex = adf_results.to_latex()
    print(adf_not_lagged_tex)