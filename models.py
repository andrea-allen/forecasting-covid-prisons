import numpy as np
import pandas as pd
from statsmodels.regression import linear_model
import statsmodels.graphics.tsaplots as tsa
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA as ARIMA
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
sns.set(palette="icefire")
# sns.set(palette="CMRmap")
sns.set_style("whitegrid", {'axes.grid': False})
palette = sns.color_palette("icefire", 6)


class ArimaX:

    def __init__(self, p, d, q, endog, exog=None):
        self.p = p
        self.d = d
        self.q = q
        self.endog = endog
        self.exog = exog
        self.model = ARIMA(endog=self.endog,
                           exog=self.exog,
                           order=(self.p, self.d, self.q))
        self.fitted = None

    def fit(self):
        fitted_model = self.model.fit()
        self.fitted = fitted_model
        summary = fitted_model.summary()
        return fitted_model

    def predict(self, Y_test, X_test=None):
        predictions = self.fitted.get_forecast(steps=len(Y_test), exog=X_test)
        aic = self.fitted.aic
        accuracy = forecast_accuracy(predictions.summary_frame()["mean"], Y_test)
        return {"predictions" : predictions, "accuracy" : accuracy, "aic": aic}


class Analyzer:
    """
    Should analyze a given time series for stationarity, ACF, PACF
    """
    @staticmethod
    def adf_test(timeseries, data_title):
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#lags used', 'number of observations used'])
        dfoutput["Data"] = data_title
        for key, value in dftest[4].items():
            dfoutput['critical value (%s)' % key] = value
        print(dfoutput)
        return dfoutput

    @staticmethod
    def plot_acf(timeseries, ax, lags=20):
        tsa.plot_acf(timeseries.dropna(), alpha=.05, lags=lags, ax=ax, title=None)
        ax.set_xlabel("Lag")
        ax.legend(["ACF"])

    @staticmethod
    def plot_pacf(timeseries, ax, lags=20):
        tsa.plot_pacf(timeseries.dropna(), alpha=.05, lags=lags, ax=ax, title=None, label="PACF")
        ax.legend(["PACF"])
        ax.set_xlabel("Lag")

    @staticmethod
    def plot_lag_corr(endog, exog, lags, ax):
        corr_df = pd.DataFrame(columns=["lag", "correlation"])
        for l in range(len(lags)):
            if lags[l] == 0:
                corr = np.corrcoef(exog.shift(-lags[l]).dropna(), y=endog)[0][1]
            else:
                corr = np.corrcoef(exog.shift(-lags[l]).dropna(), y=endog[:-lags[l]])[0][1]
            corr_df.loc[l] = [-lags[l], corr]
            ax.scatter(exog.shift(-lags[l]), endog, s=9, alpha=0.6, label=f"lag {-lags[l]}")
        ax.legend()
        print(corr_df)
        return corr_df


    @staticmethod
    def plot_summary(timeseries, data_title, label=None):
        fig, axs = plt.subplots(2,2, figsize=(8,6))
        adf_results = Analyzer.adf_test(timeseries, data_title)
        axs[0,0].plot(timeseries, label=label)
        axs[0,0].set_ylabel(data_title)
        axs[0,0].set_xticks([])
        axs[0,0].legend(loc='upper left', fontsize=10)

        axs[0,1].vlines(list([adf_results[f'critical value ({x}%)'] for x in [1,5,10]]),
                        ymin=0, ymax=10, label='critical values', color=palette[0])
        axs[0,1].vlines([adf_results['Test Statistic']],
                        ymin=0, ymax=10, label=f'test statistic\np-value: {np.round(adf_results["p-value"], 3)}',
                        color=palette[2])
        axs[0,1].legend(loc='upper left')

        Analyzer.plot_acf(timeseries, axs[1,0])
        # axs[1,0].set_ylabel('ACF')

        Analyzer.plot_pacf(timeseries, axs[1, 1])
        # axs[1, 1].set_ylabel('PACF')
        return fig, axs


class Results:
    def __init__(self, fitted_model):
        self.fitted_model = fitted_model

    def get_results_df(self):
        coef_results = pd.DataFrame(columns=['Coefficients', 'P-values'])
        try:
            coefs = self.fitted_model.params
            pvalues = self.fitted_model.pvalues
            aic = np.round(self.fitted_model.aic, 2)
            ljb_stat = self.fitted_model.summary().tables[2].data[0][1]
            ljb_pvalue = self.fitted_model.summary().tables[2].data[1][1]
        except AttributeError:
            coefs = self.fitted_model.fitted.params
            pvalues = self.fitted_model.fitted.pvalues
            aic = np.round(self.fitted_model.fitted.aic,2)
            ljb_stat = self.fitted_model.fitted.summary().tables[2].data[0][1]
            ljb_pvalue = self.fitted_model.fitted.summary().tables[2].data[1][1]

        coef_results.iloc[:, 0] = coefs
        coef_results.iloc[:, 1] = pvalues
        coef_results['Variable'] = coef_results.index
        coef_results['Coefficients'] = np.round(coef_results['Coefficients'], 4)
        coef_results['P-values'] = np.round(coef_results['P-values'], 4)

        coef_results = coef_results[["Variable", "Coefficients", "P-values"]]


        diagnostics_summary = pd.DataFrame([[ljb_stat, ljb_pvalue, aic]], columns=['Ljung-Box (Q)','Prob(Q)', 'AIC'])
        return coef_results, diagnostics_summary


class ModelFitter:
    """
    A class to store ALL training and testing data, parameters, fitted model, results, and pre-fit diagnostics
    """
    def __init__(self, y_data, x_data=None):
        self.model = None
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None
        self.x_data = x_data
        self.y_data = y_data
        if x_data is None:
            self.exog = False
        else:
            self.exog = True

    ## 3 main methods:
    # split into train and test
    def split_train_test(self):
        dataset_len = len(self.y_data)
        split_index = round(dataset_len * 0.75)
        train_set_end_date = self.y_data.index[split_index]
        self.y_train = self.y_data.loc[self.y_data.index <= train_set_end_date].copy()
        self.y_test = self.y_data.loc[self.y_data.index > train_set_end_date].copy()
        if self.exog:
            self.x_train = self.x_data.loc[self.x_data.index <= train_set_end_date].copy()
            self.x_test = self.x_data.loc[self.x_data.index > train_set_end_date].copy()


    # pre-fit: if the model is ARIMAX, then do an OLSR regression and analyze resid.
    def olsr_prefit(self):
        olsr_results = linear_model.OLS(self.y_train, self.x_train).fit()
        # residuals = olsr_results.resid
        return olsr_results

    # show the beta in a plot, maybe return the beta?
    # Otherwise, just analyze the time series at different diffs (1-3)
    # do all the pre-fit analysis plots and show results
    def pre_fit_analyze(self):
        if self.exog:
            fig, ax = plt.subplots(len(self.x_data.columns), 1)

            for i, col in enumerate(self.x_data.columns):
                if len(self.x_data.columns)==1:
                    current_ax = ax
                else:
                    current_ax = ax[i]
                corr_table = Analyzer.plot_lag_corr(self.y_data, self.x_data[col], lags=[0,1,2,3], ax =current_ax)
                current_ax.set_xlabel(col)
                current_ax.set_ylabel("Residents.Confirmed.Normed")
                # TODO save corr_table somewhere
            prefit = self.olsr_prefit()
            timeseries = prefit.resid
            coeffs = prefit.params
            coeff_names = list(coeffs.index)
            sb = "Resids via OLSR w/: \n"
            for c in coeff_names:
                sb = sb + f"Coeff {c}: {np.round(coeffs[c],1)}\n"
        else:
            timeseries = self.y_data
            sb = "Residents.Confirmed.Normed"
        Analyzer.plot_summary(timeseries, "Raw data", label=sb)
        Analyzer.plot_summary(timeseries.diff(1).dropna(), "1st difference", label=sb)
        # Analyzer.plot_summary(timeseries.diff(1).diff(1).dropna(), "2nd difference", label=sb)

    def fit_model(self, p, d, q):
        model = ArimaX(p, d, q, endog=self.y_train, exog=self.x_train)
        model.fit()
        self.model = model

    def get_results(self, show=False):
        # Prediction results:
        prediction_results = self.model.predict(Y_test=self.y_test, X_test=self.x_test)
        mape = np.round(prediction_results["accuracy"]["mape"], 2)

        if show:
            ## Residual plots
            # fig, ax = plt.subplots(1, 3, figsize=(10,8))
            fig, ax = plt.subplots(2,2, figsize=(14,8))
            plot_predictions(prediction_results["predictions"], self.y_test, self.y_test.index, mape, ax[0,0])
            plot_resids(self.model.fitted, ax[1,0])
            ax[1,1].set_title(f"Order: ({self.model.p}, {self.model.d}, {self.model.q})", fontsize=16)

        # General fit results
        resulter = Results(self.model)
        coef_table, diagnostic_table = resulter.get_results_df()
        if show:
            # fig, ax = plt.subplots(2, 1)
            cell_text = []
            for row in range(len(coef_table)):
                if len(coef_table.iloc[row]["Variable"]) > 9:
                    coef_table.iloc[row]["Variable"] = coef_table.iloc[row]["Variable"][:9]+"\n"+coef_table.iloc[row]["Variable"][9:]
                cell_text.append(coef_table.iloc[row])
            tab1 = ax[0,1].table(cellText=cell_text, colLabels=coef_table.columns, loc='center')
            tab1.auto_set_font_size(False)
            tab1.set_fontsize(14)
            tab1.scale(1,4)
            ax[0,1].axis('off')

            cell_text = []
            for row in range(len(diagnostic_table)):
                cell_text.append(diagnostic_table.iloc[row])

            tab2 = ax[1,1].table(cellText=cell_text, colLabels=diagnostic_table.columns, loc='center')
            tab2.auto_set_font_size(False)
            tab2.set_fontsize(16)
            tab2.scale(1,4)
            ax[1,1].axis('off')
            plt.tight_layout()

        return {"coef_table": coef_table, "diagnostics": diagnostic_table, "predictions": prediction_results}

    # then have a method that you can call to do the predictions (plot separately)

    # make a results object that can take the model and plot the diagnostics


class ArimaGridSearch:
    """
    Fit multiple models for a range of parameters and report results and best fits
    """
    def __init__(self, y_data, x_data=None):
        self.y_data = y_data
        self.x_data = x_data
        self.results = None

    def search(self, p_range, d_range, q_range):
        results = pd.DataFrame(columns=["p", "d", "q", "coeffs significant", "AIC", "MAPE", "Ljung-Box"])
        dimension = len(p_range)*len(d_range)*len(q_range)
        entry = 0
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    model_fitter = ModelFitter(self.y_data,
                                                         x_data=self.x_data)
                    model_fitter.split_train_test()
                    try:
                        model_fitter.fit_model(p, d, q)
                        res = model_fitter.get_results()
                        p_values = res["coef_table"]['P-values']
                        signif = p_values_significant(p_values)
                        new_row = [p, d, q, signif, res["diagnostics"]["AIC"][0],
                                             np.round(res["predictions"]["accuracy"]["mape"], 4),
                                             res["diagnostics"]['Prob(Q)'][0]]
                    except np.linalg.LinAlgError:
                        new_row = [p, d, q, False, None,
                                             None,
                                             None]
                    results.loc[entry] = new_row
                    print(f"Fitting {entry} out of {dimension} models.")
                    entry += 1
        self.results = results
        return results


def p_values_significant(p_value_series):
    for pval in p_value_series:
        if pval > .05:
            return False
    return True


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    # acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse,
            'corr':corr})


def plot_predictions(predictions, true, x_index, mape, ax):
    x_ticks = x_index[::6]
    ax.plot(x_index, predictions.summary_frame()['mean'], ls="--",
                          label='Predicted')
    ax.plot(x_index, true,
            label='Actual')
    ax.plot(x_index, predictions.summary_frame()['mean_ci_lower'],
                      marker='.', linestyle=':', label='Lower 95%')

    ax.plot(x_index, predictions.summary_frame()['mean_ci_upper'],
                      marker='.', linestyle=':', label='Upper 95%')

    ax.fill_between(x_index, predictions.summary_frame()['mean_ci_lower'],
                     predictions.summary_frame()['mean_ci_upper'], alpha=0.2,
                    label='95% CI')

    ax.legend(frameon=False, loc='upper left')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, rotation=35)
    ax.set_title(f"Out-of-time sample MAPE: {mape}%")
    ax.set_ylabel("Fraction of infected cumulative population")

def plot_resids(fitted_model, ax):
    sns.kdeplot(fitted_model.resid, ax=ax, label='ARIMA residuals')
    # Create a set of inset Axes: these should fill the bounding box allocated to
    # them.
    ax2 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax, [0.35, 0.5, 0.6, 0.25])
    ax2.set_axes_locator(ip)
    ax2.scatter(np.arange(len(fitted_model.resid)), fitted_model.resid, label='ARIMA residuals', s=10, alpha=0.6)
    ax.legend(frameon=False, loc='upper left')