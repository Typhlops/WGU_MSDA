import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.signal import periodogram
from pandas.plotting import autocorrelation_plot
from pmdarima.arima import auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


df_med = pd.read_csv('data/D213-medical-files/medical_time_series.csv')
pd.set_option('display.max_columns', None)

# When True, displays extensive summary statistics and plots. Output streamlined when False.
run_verbose = False

# auto_arima takes too long and too much memory to be practical to run for very long,
# so set to False unless desired otherwise
use_auto_arima = False

# Setting directory to save images. The function plots() will have errors if its parameter save=True and
# this isn't adjusted to fit the local environment.
img_dir = 'C:/Users/Joe/Desktop/Files/Programming/WGU MS/D213/Screenshots of plots and output/Task 1/'


print(f"\nChecking for columns with null values: {list(df_med.columns[df_med.isna().sum() > 0])}\n")
print(f"Number of duplicated values in column 'Day': {df_med.duplicated(subset='Day').sum()}\n")
print(f"Verifying 'Day' column is sequential from 1 to 731 with no gaps:\n{df_med.Day.describe()}\n")
print(f"Differences between rows are all 1.0:\n{df_med['Day'].diff().dropna().describe()}\n")


# Adapted from Dr. Festus Elleh's “D213 Webinar Task 1 Data Preprocessing Python” lecture video, accessed 2024.
# https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=8d32afc8-d66f-42d5-9117-b100016cf9ac

# Converts numbered day_coln to datetimes starting at January 1 of start_year.
def day_to_datetime(start_year=2020, day_coln='Day', df=df_med):
    start_month = 1
    start_day = 1
    df['Date'] = pd.date_range(start=datetime(start_year, start_month, start_day), periods=df.shape[0], freq='24h')
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')
    df = df.drop(columns=[day_coln])
    return df


# Adjusting dataframe from integer numbered days to dates starting at January 1, 2020 through December 31, 2021.
df_med = day_to_datetime(2020, 'Day', df_med)
df_stationary = df_med.diff().dropna()
print(df_med)
print(df_stationary)


# Prints results of augmented Dickey-Fuller test. Null hypothesis H_0 is the data is non-stationary.
# It's rejected (i.e. the data is stationary) for p-values below alpha = 0.05.
def aug_dfuller_testing(data):
    results_adful = adfuller(data)
    print(f"\nTest statistic: {results_adful[0]}\n")
    print(f"p-value: {results_adful[1]}\n")
    print(f"Number of lags used: {results_adful[2]}\n")
    print(f"Number of observations: {results_adful[3]}\n")
    print(f"Critical values: {results_adful[4]}\n")
    return results_adful


# Differences of one (day) make the data stationary
r_adf_trend = aug_dfuller_testing(df_med['Revenue'])
r_adf_statn = aug_dfuller_testing(df_stationary['Revenue'])


# Splitting data into train and test sets, ending at August 7 for the training and starting at August 8
# for the test using test_size=0.2. X is the original trending data and X_statn is the stationary difference data.
X_train, X_test = train_test_split(df_med, test_size=0.2, shuffle=False)
X_statn_train, X_statn_test = train_test_split(df_stationary, test_size=0.2, shuffle=False)

# Saving processed data to csv files
X_train.to_csv('X_train.csv')
X_test.to_csv('X_test.csv')
X_statn_train.to_csv('X_statn_train.csv')
X_statn_test.to_csv('X_statn_test.csv')


# Produces various plots of the trending data (df_mov) and stationary data (df_stat) such as
# psd, seasonal decomposition, ACF, PACF, and others. Saves image files when save=True using img_dir set towards the
# beginning of this file. Be sure to adjust it to fit the local environment to avoid errors when save=True.
# Parameter only_acf_pacf=True only shows the ACF and PACF plots for df_stat. Optional parameter label can be appended
# to names of saved images, e.g. label='_train'.
def plots(df_mov=df_med, df_stat=df_stationary, save=False, only_acf_pacf=False, label=''):
    revenue_column = 'Revenue'
    dim_figures = (12, 6)
    prd = 90

    if not only_acf_pacf:
        plt.figure(figsize=dim_figures)
        plt.plot(df_mov[revenue_column])
        x0 = matplotlib.dates.date2num(df_mov.index)
        L0 = np.poly1d(np.polyfit(x0, df_mov[revenue_column], 1))
        plt.plot(x0, L0(x0), 'm--')
        plt.title('Revenue (cumulative) for 2020 and 2021')
        plt.xlabel('Date')
        plt.ylabel('Cumulative (current) revenue (in millions USD)')
        plt.grid(True)
        if save:
            plt.savefig(img_dir+'revenue_mov'+label+'.png')
        plt.show()

        plt.figure(figsize=dim_figures)
        plt.plot(df_stat)
        plt.axhline(y=df_stat[revenue_column].mean(), color='red', label=f'mean ({round(df_stat[revenue_column].mean(), 2)})')
        plt.xlabel('Date')
        plt.ylabel('Daily revenue (in millions USD)')
        plt.title('Daily revenue for 2020 and 2021')
        plt.legend()
        if save:
            plt.savefig(img_dir+'revenue_stat'+label+'.png')
        plt.show()
    
        plt.figure(figsize=dim_figures)
        plt.psd(df_mov[revenue_column])
        plt.title('Spectral density for trending data')
        plt.xlabel('Frequency')
        plt.ylabel('Spectral density')
        plt.grid(True)
        if save:
            plt.savefig(img_dir+'spectral_mov'+label+'.png')
        plt.show()
    
        f, sp_den = periodogram(df_stat[revenue_column])
        plt.figure(figsize=dim_figures)
        plt.semilogy(f, sp_den)
        plt.axhline(y=sp_den.mean(), color='red', label=f'mean ({round(sp_den.mean(), 2)})')
        tmp_arr = sorted(sp_den)
        sp_den_median = tmp_arr[int(len(sp_den)/2)]
        plt.axhline(y=sp_den_median, color='black', label=f'median ({round(sp_den_median, 2)})')
        plt.ylim([1e-4, 1e1])
        plt.title('Spectral density for stationary data')
        plt.xlabel('Frequency')
        plt.ylabel('Spectral density')
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(img_dir+'spectral_stat'+label+'.png')
        plt.show()
    
        decomp_mov = seasonal_decompose(df_mov[revenue_column], period=prd)
        decomp_mov.plot()
        plt.show()

        plt.figure(figsize=dim_figures)
        plt.plot(decomp_mov.observed)
        plt.xlabel('Date')
        plt.ylabel('Cumulative (current) revenue (in millions USD)')
        plt.title('Revenue (cumulative)')
        if save:
            plt.savefig(img_dir+'decomp_revenue_mov'+label+'.png')
        plt.show()

        plt.figure(figsize=dim_figures)
        plt.plot(decomp_mov.trend)
        plt.xlabel('Date')
        plt.ylabel('Cumulative (current) revenue (in millions USD)')
        plt.title('Trend for revenue (cumulative)')
        if save:
            plt.savefig(img_dir+'decomp_trend_mov'+label+'.png')
        plt.show()

        plt.figure(figsize=dim_figures)
        plt.plot(decomp_mov.seasonal)
        plt.axhline(y=decomp_mov.seasonal.mean(), color='red', label=f'mean ({round(decomp_mov.seasonal.mean(), 2)})')
        plt.xlabel('Date')
        plt.ylabel('USD (in millions)')
        plt.title('Seasonal decomposition of revenue with trend')
        plt.legend()
        if save:
            plt.savefig(img_dir+'decomp_seasonal_mov'+label+'.png')
        plt.show()

        plt.figure(figsize=dim_figures)
        plt.plot(decomp_mov.resid)
        plt.axhline(y=decomp_mov.resid.mean(), color='red', label=f'mean ({round(decomp_mov.resid.mean(), 2)})')
        plt.xlabel('Date')
        plt.ylabel('USD (in millions)')
        plt.title('Residuals from seasonal decomposition of revenue with trend')
        plt.legend()
        if save:
            plt.savefig(img_dir+'decomp_residuals_mov'+label+'.png')
        plt.show()
    
        decomp_stat = seasonal_decompose(df_stat[revenue_column], period=prd)
        decomp_stat.plot()
        plt.show()

        plt.figure(figsize=dim_figures)
        plt.plot(decomp_stat.observed)
        plt.xlabel('Date')
        plt.ylabel('Daily revenue (in millions USD)')
        plt.title('Daily revenue')
        if save:
            plt.savefig(img_dir+'decomp_revenue_stat'+label+'.png')
        plt.show()

        plt.figure(figsize=dim_figures)
        plt.plot(decomp_stat.trend)
        plt.xlabel('Date')
        plt.ylabel('Daily revenue (in millions USD)')
        plt.title('Trend for revenue (daily)')
        if save:
            plt.savefig(img_dir+'decomp_trend_stat'+label+'.png')
        plt.show()

        plt.figure(figsize=dim_figures)
        plt.plot(decomp_stat.seasonal)
        plt.axhline(y=decomp_stat.seasonal.mean(), color='red', label=f'mean ({round(decomp_stat.seasonal.mean(), 2)})')
        plt.xlabel('Date')
        plt.ylabel('USD (in millions)')
        plt.title('Seasonal decomposition of stationary revenue')
        plt.legend()
        if save:
            plt.savefig(img_dir+'decomp_seasonal_stat'+label+'.png')
        plt.show()

        plt.figure(figsize=dim_figures)
        plt.plot(decomp_stat.resid)
        plt.axhline(y=decomp_stat.resid.mean(), color='red', label=f'mean ({round(decomp_stat.resid.mean(), 2)})')
        plt.xlabel('Date')
        plt.ylabel('USD (in millions)')
        plt.title('Residuals from seasonal decomposition of stationary revenue')
        plt.legend()
        if save:
            plt.savefig(img_dir+'decomp_residuals_stat'+label+'.png')
        plt.show()

        plt.figure(figsize=dim_figures)
        autocorrelation_plot(df_mov[revenue_column].tolist())
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation vs Lag')
        if save:
            plt.savefig(img_dir+'autocorr_v_lag_plot_mov'+label+'.png')
        plt.show()

        fig_0 = plt.figure(figsize=dim_figures)
        ax_0 = fig_0.add_subplot(111)
        fig_0 = plot_acf(df_mov[revenue_column], lags=30, ax=ax_0)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation on trending revenue data vs lag')
        if save:
            plt.savefig(img_dir+'acf_mov'+label+'.png')
        plt.show()

        fig_1 = plt.figure(figsize=dim_figures)
        ax_1 = fig_1.add_subplot(111)
        fig_1 = plot_pacf(df_mov[revenue_column], lags=30, ax=ax_1)
        plt.xlabel('Lag')
        plt.ylabel('Partial autocorrelation')
        plt.title('Partial autocorrelation on trending revenue data vs lag')
        if save:
            plt.savefig(img_dir+'pacf_mov'+label+'.png')
        plt.show()

    fig_2 = plt.figure(figsize=dim_figures)
    ax_2 = fig_2.add_subplot(111)
    fig_2 = plot_acf(df_stat[revenue_column], ax=ax_2)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation on stationary revenue data vs lag')
    if save:
        plt.savefig(img_dir+'acf_stat'+label+'.png')
    plt.show()

    fig_3 = plt.figure(figsize=dim_figures)
    ax_3 = fig_3.add_subplot(111)
    fig_3 = plot_pacf(df_stat[revenue_column], ax=ax_3)
    plt.xlabel('Lag')
    plt.ylabel('Partial autocorrelation')
    plt.title('Partial autocorrelation on stationary revenue data vs lag')
    if save:
        plt.savefig(img_dir+'pacf_stat'+label+'.png')
    plt.show()

    return


# Adapted from Dr. Festus Elleh's “Task 1 Building Arima Model in Python video” lecture video, accessed 2024.
# https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=1aaf2389-6483-4000-b498-b14f00441d57

# Selects ARIMA model with the lowest AIC and Ljung-Box statistic greater than lb_thresh = 0.05 and model
# coefficients with p-values below z_thresh = 0.10. Default value of d is 1 while p and q are looped from 0 to pq_rng-1.
# The SARIMAX trend parameter is looped over the array [None, 'c']. The models' AIC, smallest Ljung-Box value,
# largest p-value on model coefficients, and RSS values are placed in a dictionary with key (p, d, q, f_trend). The
# function returns the best model (lowest AIC and Ljung-Box > lb_thresh and coefficient p-values below z_thresh) and
# a dataframe for the models' AIC, smallest Ljung-Box value, the largest p-value for its coefficients, and RSS values
# (indexed by (p, d, q, f_trend) tuples). Methods other than 'lbfgs' were tried when fitting, but they all periodically
# had convergence issues.
def aic_arima_search(pq_rng, d=1, data=X_train, season_param=None):
    # Could include 't', 'ct', or other options, but anything other than None or 'c' consistently created poor models
    trend_arr = [None, 'c']
    lb_thresh = 0.05
    z_thresh = 0.10
    mthd = 'lbfgs'

    aic_dict = {}
    best_aic = np.inf
    best_order = None
    best_model = None
    best_lb = None
    best_z_max = None

    for p in range(pq_rng):
        for q in range(pq_rng):
            for f_trend in trend_arr:
                try:
                    tmp_model = SARIMAX(data, order=(p, d, q), seasonal_order=season_param, trend=f_trend, freq='D')
                    tmp_results = tmp_model.fit(method=mthd, maxiter=100)
                    tmp_aic = tmp_results.aic
                    tmp_rss_mov = sum((tmp_results.fittedvalues - data['Revenue'])**2)
                    # When d != 1 this needs to be rewritten as nested diff()^d - this is currently a placeholder functional when d = 1
                    tmp_rss_statn = sum((tmp_results.fittedvalues.diff(d).dropna() - data['Revenue'].diff(d).dropna())**2)
                    tmp_ljung_box_results = acorr_ljungbox(tmp_results.filter_results.standardized_forecasts_error[0], lags=10, return_df=True)
                    tmp_lb_min = tmp_ljung_box_results['lb_pvalue'].min()
                    tmp_z_score_arr = []
                    for i in range(1, len(tmp_results.summary().tables[1])):
                        tmp_z_score_arr.append(float(str(tmp_results.summary().tables[1][i][4])))
                    tmp_z_max = max(tmp_z_score_arr)
                    aic_dict[(p, d, q, f_trend)] = (tmp_results.aic, tmp_lb_min, tmp_z_max, tmp_rss_statn, tmp_rss_mov)
                    print("p, d, q, f_trend: ", p, d, q, f_trend)
                    print("AIC, BIC, min Ljung-Box, largest coef p-value, RSS (stationary): ", tmp_results.aic, tmp_results.bic, tmp_lb_min, tmp_z_max, tmp_rss_statn)
                    if tmp_aic < best_aic and tmp_lb_min > lb_thresh and tmp_z_max <= z_thresh:
                        best_aic = tmp_aic
                        best_order = (p, d, q, f_trend)
                        best_model = tmp_model
                        best_lb = tmp_lb_min
                        best_z_max = tmp_z_max
                except:
                    print(f"Encountered an error with the above SARIMAX model on (p, d, q) = {(p, d, q)} and trend = '{f_trend}'.\n")
                    print(p, d, q, None, None)

    print(f"\nBest AIC: {best_aic} | Order: {best_order} | Ljung-Box: {best_lb} | Largest coefficient p-value: {best_z_max} | Seasonal order fixed at {season_param}")
    aic_dict = {k: v for k, v in sorted(aic_dict.items(), key=lambda s: s[1][0])}
    df_aic = pd.DataFrame.from_dict(aic_dict, orient='index', columns=['AIC', 'Ljung-Box min', 'Largest coef p-value', 'RSS (stationary)', 'RSS (with trend)'])
    with pd.option_context('display.max_rows', None):
        print(f"\nModels with Ljung-Box p-values above {lb_thresh}:\n")
        print(df_aic[df_aic['Ljung-Box min'] > lb_thresh])
        print(f"\nModels with Ljung-Box p-values above {lb_thresh} and coefficient p-values below {z_thresh}:\n")
        print(df_aic[(df_aic['Ljung-Box min'] > lb_thresh) & (df_aic['Largest coef p-value'] <= z_thresh)])
    print("Best model:\n", best_model.fit(method=mthd, maxiter=100).summary())
    return best_model, df_aic


# Uses a Fast Fourier Transform on stationary data to look for prominent seasonal signals. Provides a plot and
# dictionary of the top 'max_peaks' frequencies, along with their associated periods and powers.
def seasonal_fft(max_peaks=5, df=df_stationary):
    itvl = 1.0
    fig_dims = (12, 6)

    fft_frq = np.fft.fftfreq(len(df))
    spec_pwr = np.abs(np.fft.fft(df['Revenue'])) ** 2

    plt.figure(figsize=fig_dims)
    plt.plot(fft_frq, spec_pwr)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.xlim(0.0, 0.5)
    plt.title('FFT power spectrum of detrended revenue')
    plt.show()

    dom_idx = np.argsort(spec_pwr[:len(fft_frq) // 2])[::-1][:max_peaks]
    dom_pwr = spec_pwr[np.argsort(spec_pwr[:len(fft_frq) // 2])[::-1][:max_peaks]]
    dom_frqs = fft_frq[dom_idx]
    dom_periods = itvl / np.array(dom_frqs)
    dom_dict = {k: v for k, v in zip(np.round(dom_periods, 3), np.round(dom_pwr, 3))}

    print(f'Top {max_peaks} dominant frequencies: {np.round(dom_frqs, 5)}')
    print(f'Corresponding dominant periods and powers: {dom_dict}')
    return dom_dict


# Produces several plots (such as ACF, PACF, and rolling differences) to investigate seasonality. The seasonal period
# is set by 'per', 'max_lags' defines the range of most plots, and save=False by default as the global 'img_dir' at
# the top of the file is set to the local environment.
def inspect_seasonal(per=100, max_lags=600, df=df_med, save=False):
    fig_dims = (12, 6)
    # roll_window_arr = [14, 32, 104]
    roll_window_arr = [15, 30, 90]
    color_arr = [['green', 'red', 'xkcd:dark rose'], ['orange', 'blue', 'xkcd:hot pink'], ['purple', 'xkcd:neon green', 'cyan']]
    df_season = df.diff().diff(per).dropna()

    fig_0 = plt.figure(figsize=fig_dims)
    ax_0 = fig_0.add_subplot(111)
    fig_0 = plot_acf(df_stationary['Revenue'], lags=max_lags, ax=ax_0)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation on stationary revenue data vs lag')
    if save:
        plt.savefig(img_dir+'seasonal_acf_statn.png')
    plt.show()

    fig_1 = plt.figure(figsize=(24, 12))
    for i, w in enumerate(roll_window_arr):
        fig_1.add_subplot(3, 1, i+1)
        mean_roll = df.rolling(w).mean()
        std_roll = df.rolling(w).std()
        df_roll_diff = (df - mean_roll).dropna()
        plt.plot(df['Revenue'], color='black', label='Original', linestyle='dotted')
        plt.plot(df_roll_diff['Revenue'], color=color_arr[i][0], label=f'Rolling difference ({w}d)', linestyle='dashed')
        plt.plot(mean_roll, color=color_arr[i][1], label=f'Rolling mean ({w}d)')
        plt.plot(std_roll, color=color_arr[i][2], label=f'Rolling std ({w}d)', linestyle='dashdot')
        plt.xlabel('Date')
        plt.ylabel('Revenue (in millions USD)')
        plt.title(f'Rolling difference ({w}d), mean, and standard deviation against original')
        plt.legend(loc='best')
    plt.tight_layout()
    if save:
        plt.savefig(img_dir+'seasonal_rolling_diffs.png')
    plt.show()

    fig_2 = plt.figure(figsize=fig_dims)
    ax_2 = fig_2.add_subplot(111)
    fig_2 = plot_acf(df_roll_diff, lags=max_lags, ax=ax_2)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation on difference of trending data with {roll_window_arr[-1]} day rolling average')
    plt.annotate(text='61', xy=(58, -0.193))
    plt.annotate(text='111', xy=(101, 0.249))
    plt.annotate(text='163', xy=(160, -0.235))
    plt.annotate(text='216', xy=(206, 0.229))
    plt.annotate(text='278', xy=(275, -0.316))
    plt.annotate(text='334', xy=(324, 0.062))
    plt.grid(True)
    if save:
        plt.savefig(img_dir+f'seasonal_rolling_diff_{roll_window_arr[-1]}_acf.png')
    plt.show()

    plt.figure(figsize=fig_dims)
    plt.plot(df_season['Revenue'])
    plt.xlabel('Date')
    plt.ylabel('Revenue (in millions USD)')
    plt.title(f'Time series after first order difference then seasonal difference of {per}')
    if save:
        plt.savefig(img_dir+f'seasonal_revenue_statn_{per}.png')
    plt.show()

    fig_3 = plt.figure(figsize=fig_dims)
    ax_3 = fig_3.add_subplot(111)
    fig_3 = plot_acf(df_season, lags=max_lags, ax=ax_3)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation on seasonally differenced data .diff().diff({per})')
    if save:
        plt.savefig(img_dir+f'seasonal_differenced_acf_{per}.png')
    plt.show()

    fig_4 = plt.figure(figsize=fig_dims)
    ax_4 = fig_4.add_subplot(111)
    fig_4 = plot_acf(df_season['Revenue'], lags=range(per, max_lags+1, per), ax=ax_4)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation on seasonally differenced data .diff().diff({per}) by periodic increments')
    if save:
        plt.savefig(img_dir+f'seasonal_differenced_acf_periodic_{per}.png')
    plt.show()

    fig_5 = plt.figure(figsize=fig_dims)
    ax_5 = fig_5.add_subplot(111)
    fig_5 = plot_pacf(df_season['Revenue'], lags=range(per, int(max_lags/2+1), per), ax=ax_5)
    plt.xlabel('Lag')
    plt.ylabel('Partial autocorrelation')
    plt.title(f'Partial autocorrelation on seasonally differenced data .diff().diff({per}) by periodic increments')
    if save:
        plt.savefig(img_dir+f'seasonal_differenced_pacf_periodic_{per}.png')
    plt.show()

    return


# Creates 16 subplots of seasonal decomposition seasonal and residual components on trending and stationary data,
# along with their corresponding autocorrelation functions.
def seasonal_plots(perd, save=False, df_mov=df_med, df_statn=df_stationary):
    max_lags = perd * 3
    decomp_mov = seasonal_decompose(df_mov['Revenue'], period=perd)
    fig1 = plt.figure(figsize=(24, 12))
    fig1.add_subplot(4, 4, 1)
    plt.plot(decomp_mov.resid)
    plt.xlabel('Date')
    plt.ylabel('USD (in millions)')
    plt.title(f'Residuals of seasonal decomposition of trending revenue (period {perd})')

    fig1.add_subplot(4, 4, 2)
    plt.plot(decomp_mov.seasonal)
    plt.xlabel('Date')
    plt.ylabel('USD (in millions)')
    plt.title(f'Seasonal decomposition of trending revenue (period {perd})')

    plot_acf(decomp_mov.resid.dropna(), lags=max_lags, ax=fig1.add_subplot(4, 4, 3))
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation on non-stationary residuals (period {perd})')

    plot_acf(decomp_mov.seasonal, lags=max_lags, ax=fig1.add_subplot(4, 4, 4))
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation on non-stationary seasonal component (period {perd})')

    decomp_mov = seasonal_decompose(df_mov['Revenue'])
    fig1.add_subplot(4, 4, 5)
    plt.plot(decomp_mov.resid)
    plt.xlabel('Date')
    plt.ylabel('USD (in millions)')
    plt.title(f'Residuals of seasonal decomposition of trending revenue (no period)')

    fig1.add_subplot(4, 4, 6)
    plt.plot(decomp_mov.seasonal)
    plt.xlabel('Date')
    plt.ylabel('USD (in millions)')
    plt.title('Seasonal decomposition of trending revenue (no period)')

    plot_acf(decomp_mov.resid.dropna(), lags=max_lags, ax=fig1.add_subplot(4, 4, 7))
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation on non-stationary residuals (no period))')

    plot_acf(decomp_mov.seasonal, lags=max_lags, ax=fig1.add_subplot(4, 4, 8))
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation on non-stationary seasonal component (no period)')

    decomp_stat = seasonal_decompose(df_statn['Revenue'], period=perd)
    fig1.add_subplot(4, 4, 9)
    plt.plot(decomp_stat.resid)
    plt.xlabel('Date')
    plt.ylabel('USD (in millions)')
    plt.title(f'Residuals of seasonal decomposition of stationary revenue (period {perd})')

    fig1.add_subplot(4, 4, 10)
    plt.plot(decomp_stat.seasonal)
    plt.xlabel('Date')
    plt.ylabel('USD (in millions)')
    plt.title(f'Seasonal decomposition of stationary revenue (period {perd})')

    plot_acf(decomp_stat.resid.dropna(), lags=max_lags, ax=fig1.add_subplot(4, 4, 11))
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation on stationary residuals with period {perd}')

    plot_acf(decomp_stat.seasonal, lags=max_lags, ax=fig1.add_subplot(4, 4, 12))
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation on stationary seasonal component with period {perd}')

    decomp_stat = seasonal_decompose(df_statn['Revenue'])
    fig1.add_subplot(4, 4, 13)
    plt.plot(decomp_stat.resid)
    plt.xlabel('Date')
    plt.ylabel('USD (in millions)')
    plt.title(f'Residuals of seasonal decomposition of stationary revenue (no period)')

    fig1.add_subplot(4, 4, 14)
    plt.plot(decomp_stat.seasonal)
    plt.xlabel('Date')
    plt.ylabel('USD (in millions)')
    plt.title('Seasonal decomposition of stationary revenue (no period)')

    plot_acf(decomp_stat.resid.dropna(), lags=max_lags, ax=fig1.add_subplot(4, 4, 15))
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation on stationary residuals (no period))')

    plot_acf(decomp_stat.seasonal, lags=max_lags, ax=fig1.add_subplot(4, 4, 16))
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation on stationary seasonal component (no period))')

    plt.tight_layout()
    if save:
        plt.savefig(img_dir+f'seasonal_plots_{perd}.png')
    plt.show()
    return


# Creates a SARIMAX model using input 'data' with order 'ordr' as a tuple (p, d, q) and trend 'f_trend' and
# seasonal_order 'season_param'. The trend and seasonal_order default to None. Calculates results, RSS, RMSE, MAE,
# and Ljung-Box values. Returns the model, its results from applying .fit(), and the created dataframe df_results for
# additional statistics (when simple=False).
def model_creation(data, ordr, f_trend=None, season_param=None, simple=False):
    fq = 'D'
    fit_method = 'lbfgs'
    iter_max = 100
    max_lag = 10
    p, d, q = ordr

    model0 = SARIMAX(data, order=(p, d, q), seasonal_order=season_param, trend=f_trend, freq=fq)
    results0 = model0.fit(method=fit_method, maxiter=iter_max)

    if simple:
        return model0, results0
    else:
        rss = sum((results0.fittedvalues - data['Revenue']) ** 2)
        ljung_box_results = acorr_ljungbox(results0.filter_results.standardized_forecasts_error[0], lags=max_lag, return_df=True)
        lb_min = ljung_box_results['lb_pvalue'].min()
        rmse = np.sqrt(mean_squared_error(results0.fittedvalues, data['Revenue']))
        mae = np.mean(np.abs(results0.resid))

        if d == 1:
            rss_statn = sum((results0.fittedvalues.diff(d).dropna() - data['Revenue'].diff(d).dropna()) ** 2)
            df_results = pd.DataFrame([rmse, mae, rss, rss_statn, lb_min], columns=['Value'], index=['RMSE', 'MAE', 'RSS', 'RSS (stationary)', 'Lowest Ljung-Box p-value'])
        else:
            df_results = pd.DataFrame([rmse, mae, rss, lb_min], columns=['Value'], index=['RMSE', 'MAE', 'RSS', 'Lowest Ljung-Box p-value'])
        print(results0.summary())
        print(f"\nAdditional model statistics:\n{df_results}\n")
        print(f"Ljung-Box statistics for lags 1 to {max_lag}:\n{ljung_box_results}\n")
        results0.plot_diagnostics(figsize=[16, 10])
        plt.show()
        return model0, results0, df_results


# Passes order, trend, and season_order to model_creation using training data from an 80/20 split of df. Calculates
# predictions and forecasts three months beyond the end of the dataset. Produces plots of the model's residuals and
# predictions/forecasts. Then calculates model evaluation statistics and returns it as a dataframe.
def model_predictions(order, trend=None, season_order=None, df=df_med, save=False):
    split = 0.2
    forecast_steps = 237
    fig_dim = (12, 6)
    pred_color = 'r'
    fill_color = 'pink'
    forecast_end = '03-31-2022'

    train_data, test_data = train_test_split(df, test_size=split, shuffle=False)
    model_train, model_results = model_creation(train_data, order, trend, season_order, True)
    train_fitted = model_results.fittedvalues

    predictions = model_results.predict(len(train_data), end=(len(df) - 1))
    prediction = model_results.get_prediction(start=len(train_data), end=(len(df)-1))
    confidence_intervals_pr = prediction.conf_int()
    lower_lims_pr = confidence_intervals_pr.loc[:, 'lower Revenue']
    upper_lims_pr = confidence_intervals_pr.loc[:, 'upper Revenue']
    x_pr = matplotlib.dates.date2num(predictions.index)
    L_pr = np.poly1d(np.polyfit(x_pr, predictions, 1))

    md_forecasting = model_results.forecast(forecast_steps)
    md_forecast = model_results.get_forecast(forecast_steps)
    confidence_intervals_ft = md_forecast.conf_int()
    lower_lims_ft = confidence_intervals_ft.loc[:, 'lower Revenue']
    upper_lims_ft = confidence_intervals_ft.loc[:, 'upper Revenue']
    x_ft = matplotlib.dates.date2num(md_forecasting.index)
    L_ft = np.poly1d(np.polyfit(x_ft, md_forecasting, 1))

    model_updated = model_results.append(test_data)
    updated_forecast = model_updated.forecast(forecast_steps-len(test_data))
    md_forecast_upd = model_updated.get_forecast(forecast_steps-len(test_data))
    confidence_intervals_ft_upd = md_forecast_upd.conf_int()
    lower_lims_ft_upd = confidence_intervals_ft_upd.loc[:, 'lower Revenue']
    upper_lims_ft_upd = confidence_intervals_ft_upd.loc[:, 'upper Revenue']
    x_ft_upd = matplotlib.dates.date2num(updated_forecast.index)
    L_ft_upd = np.poly1d(np.polyfit(x_ft_upd, updated_forecast, 1))

    print(f"Predicted values across test data:\n{predictions}\n")
    print(f"Forecasted values through {forecast_end}:\n{md_forecasting}\n")
    print(f"Forecasted values from 01-01-2022 through {forecast_end} "
          f"after appending test data (model is still trained on training data):\n{updated_forecast}\n")

    resid_train = model_results.resid
    resid_test = predictions - test_data['Revenue']
    resid_all = pd.concat([train_fitted, predictions]) - df['Revenue']

    fig_rsd = plt.figure(figsize=(24, 12))
    fig_rsd.add_subplot(311)
    plt.plot(resid_train.index, resid_train)
    plt.title('Residuals of training data')
    plt.xlabel('Date')
    plt.ylabel('Revenue (in millions USD)')

    fig_rsd.add_subplot(312)
    plt.plot(resid_test.index, resid_test)
    plt.title('Residuals of test data')
    plt.xlabel('Date')
    plt.ylabel('Revenue (in millions USD)')

    fig_rsd.add_subplot(313)
    plt.plot(resid_all.index, resid_all)
    plt.title('Residuals of all data')
    plt.xlabel('Date')
    plt.ylabel('Revenue (in millions USD)')
    plt.tight_layout()
    if save:
        plt.savefig(img_dir+f'model_residuals_compilation_{order}_{trend}_{season_order}.png')
    plt.show()

    plt.figure(figsize=fig_dim)
    plt.plot(test_data.index, test_data, label='Observed (test data)')
    plt.plot(predictions.index, predictions, color=pred_color, label='Predicted')
    plt.plot(x_pr, L_pr(x_pr), 'k--', label='Predicted trend')
    plt.fill_between(lower_lims_pr.index, lower_lims_pr, upper_lims_pr, color=fill_color)
    plt.title('Comparing test data with model predictions and shaded confidence intervals')
    plt.xlabel('Date')
    plt.ylabel('Revenue (in millions USD)')
    plt.legend()
    if save:
        plt.savefig(img_dir+f'model_predictions_{order}_{trend}_{season_order}.png')
    plt.show()

    plt.figure(figsize=fig_dim)
    plt.plot(model_results.fittedvalues, label='Fitted by model', color='orange', linestyle='dashed')
    plt.plot(train_data, label='Training', color='b')
    plt.plot(test_data, label='Test', color='g')
    plt.plot(md_forecasting, label='Forecast', color=pred_color)
    plt.plot(x_ft, L_ft(x_ft), 'k--', label='Forecasted trend')
    plt.fill_between(lower_lims_ft.index, lower_lims_ft, upper_lims_ft, color=fill_color)
    plt.title('Revenue with forecasted projections for 2022 Q1 and shaded confidence intervals')
    plt.xlabel('Date')
    plt.ylabel('Revenue (in millions USD)')
    plt.legend()
    if save:
        plt.savefig(img_dir+f'model_forecast_confint_{order}_{trend}_{season_order}.png')
    plt.show()

    plt.figure(figsize=fig_dim)
    plt.plot(model_results.fittedvalues, label='Fitted by model', color='orange', linestyle='dashed')
    plt.plot(train_data, label='Training', color='b')
    plt.plot(test_data, label='Test', color='g')
    plt.plot(md_forecasting, label='Forecast', color=pred_color)
    plt.plot(x_ft, L_ft(x_ft), 'k--', label='Forecasted trend')
    plt.title('Revenue with forecasted projections for 2022 Q1')
    plt.xlabel('Date')
    plt.ylabel('Revenue (in millions USD)')
    plt.legend()
    if save:
        plt.savefig(img_dir+f'model_forecast_plain_{order}_{trend}_{season_order}.png')
    plt.show()

    plt.figure(figsize=fig_dim)
    plt.plot(model_results.fittedvalues, label='Fitted by model', color='orange', linestyle='dashed')
    plt.plot(train_data, label='Training', color='b')
    plt.plot(test_data, label='Test', color='g')
    plt.plot(updated_forecast, label='Forecast from last data point', color=pred_color)
    plt.plot(x_ft_upd, L_ft_upd(x_ft_upd), 'k--', label='Forecasted trend from last data point')
    plt.fill_between(lower_lims_ft_upd.index, lower_lims_ft_upd, upper_lims_ft_upd, color=fill_color)
    plt.title('Revenue with forecasted projections for 2022 Q1 (accounting for test data) and shaded confidence intervals')
    plt.xlabel('Date')
    plt.ylabel('Revenue (in millions USD)')
    plt.legend()
    if save:
        plt.savefig(img_dir+f'model_forecast_updated_confint_{order}_{trend}_{season_order}.png')
    plt.show()

    model_results.plot_diagnostics(figsize=(16, 10))
    if save:
        plt.savefig(img_dir+f'model_diagnostics_{order}_{trend}_{season_order}.png')
    plt.show()

    fig_tr = plt.figure(figsize=fig_dim)
    ax_tr = fig_tr.add_subplot(111)
    fig_tr = plot_acf(resid_train, lags=(len(train_data) - 2), ax=ax_tr)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation of residuals (training)')
    if save:
        plt.savefig(img_dir+f'model_acf_resid_train_{order}_{trend}_{season_order}.png')
    plt.show()

    fig_te = plt.figure(figsize=fig_dim)
    ax_te = fig_te.add_subplot(111)
    fig_te = plot_acf(resid_test, lags=(len(test_data) - 2), ax=ax_te)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation of residuals (test)')
    if save:
        plt.savefig(img_dir+f'model_acf_resid_test_{order}_{trend}_{season_order}.png')
    plt.show()

    fig_a = plt.figure(figsize=fig_dim)
    ax_a = fig_a.add_subplot(111)
    fig_a = plot_acf(resid_all, lags=(len(df) - 2), ax=ax_a)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation of residuals (all)')
    if save:
        plt.savefig(img_dir+f'model_acf_resid_all_{order}_{trend}_{season_order}.png')
    plt.show()

    total_predictions = pd.concat([model_results.fittedvalues, predictions])
    rss_train = sum(resid_train ** 2)
    rss_test = sum(resid_test ** 2)
    rss_all = rss_train + rss_test
    rss_statn = sum((total_predictions.diff().dropna() - df['Revenue'].diff().dropna()) ** 2)

    mae_train = np.mean(np.abs(resid_train))
    mae_test = np.mean(np.abs(resid_test))
    mae_all = mae_train + mae_test

    rmse_train = np.sqrt(mean_squared_error(model_results.fittedvalues, train_data['Revenue']))
    rmse_test = np.sqrt(mean_squared_error(predictions, test_data['Revenue']))
    rmse_all = rmse_train + rmse_test

    rsq_train = r2_score(train_data, train_fitted)
    rsq_test = r2_score(test_data, predictions)
    rsq_all = r2_score(df, total_predictions)

    df_results = pd.DataFrame([rmse_train, rmse_test, rmse_all,
                                    mae_train, mae_test, mae_all,
                                    rsq_train, rsq_test, rsq_all,
                                    rss_train, rss_test, rss_all, rss_statn],
                              columns=['Value'],
                              index=[['RMSE', 'RMSE', 'RMSE', 'MAE', 'MAE', 'MAE',
                                      'R^2', 'R^2', 'R^2', 'RSS', 'RSS', 'RSS', 'RSS'],
                                     ['train', 'test', 'all', 'train', 'test', 'all',
                                      'train', 'test', 'all', 'train', 'test', 'all', 'stationary']])
    print(model_results.summary())
    print(df_results)
    return df_results




if use_auto_arima:
    model_season = auto_arima(df_med, seasonal=True, m=104, d=1, D=1, start_p=0, max_p=2, start_q=2, max_q=2, start_P=2,
                              max_P=2, start_Q=0, max_Q=1, trace=True, error_action='ignore', suppress_warnings=True)
    print(model_season.summary())


if run_verbose:
    plots(df_med, df_stationary, False, False)
    plots(X_train, X_statn_train, False, True, '_train')
    b_model, b_aic = aic_arima_search(3, 1, X_train, (2, 1, 0, 104))
    seasonal_fft(7, df_stationary)
    inspect_seasonal(104, 600, df_med, False)
    seasonal_plots(104, False, df_med, df_stationary)
    seasonal_plots(8, False, df_med, df_stationary)
    model_0, results_0, df_results_0 = model_creation(X_train, (0, 1, 2), None, (2, 1, 0, 104), False)
    df_model_results = model_predictions((0, 1, 2), None, (2, 1, 0, 104), df_med, False)




