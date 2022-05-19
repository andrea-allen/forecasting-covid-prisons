import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

font = {'family': 'normal',
        'size': 14}

matplotlib.rc('font', **font)


def plot_grid_search(result_df, ax, label, symbol):
    ax[0].scatter(result_df.index, result_df["MAPE"], label=label, s=80, alpha=0.7, marker=symbol)
    ax[1].scatter(result_df.index, result_df["AIC"], s=80, alpha=0.7, marker=symbol)
    ax[2].scatter(result_df.index, result_df["AIC"] + result_df["MAPE"], s=80, alpha=0.7, marker=symbol)


def significant_results(result_df):
    result_df = result_df[result_df["coeffs significant"] == True]
    result_df = result_df[result_df["Ljung-Box"] > .05]
    result_df = result_df.sort_values(by="MAPE")
    result_df["order"] = [f"({result_df.iloc[i]['p']}, {result_df.iloc[i]['d']}, {result_df.iloc[i]['q']})" for i in
                          range(len(result_df))]
    return result_df

def plot_gridsearch_results():
    single_result = pd.read_csv("results/WA_ARIMA_grid_diff1.csv", index_col=0)
    single_result["order"] = [f"({single_result.iloc[i]['p']}, {single_result.iloc[i]['d']}, {single_result.iloc[i]['q']})" for i in
                          range(len(single_result))]
    pdq_range = single_result[["order"]]
    wa_results_univar = significant_results(pd.read_csv("results/WA_ARIMA_grid_diff1.csv", index_col=0))
    wa_results_staff = significant_results(pd.read_csv("results/WA_ARIMAX_staff_grid_diff1.csv", index_col=0))
    wa_results_state = significant_results(pd.read_csv("results/WA_ARIMAX_state_grid_diff1.csv", index_col=0))
    wa_results_2var = significant_results(pd.read_csv("results/WA_ARIMAX_staff_state_grid_diff1.csv", index_col=0))

    ### Making final plots of results from ARIMA grid search
    ca_results_univar = significant_results(pd.read_csv("results/CA_ARIMA_grid_diff1.csv", index_col=0))
    ca_results_staff = significant_results(pd.read_csv("results/CA_ARIMAX_staff_grid_diff1.csv", index_col=0))
    ca_results_state = significant_results(pd.read_csv("results/CA_ARIMAX_state_grid_diff1.csv", index_col=0))
    ca_results_2var = significant_results(pd.read_csv("results/CA_ARIMAX_staff_state_grid_diff1.csv", index_col=0))

    latex_tables = {"WA" : {"Univar": wa_results_univar[["order","AIC", "MAPE", "Ljung-Box"]].to_latex(escape=False), "ARIMAX_staff": wa_results_staff[["order","AIC", "MAPE", "Ljung-Box"]].to_latex(escape=False),
                            "ARIMAX_state": wa_results_state[["order","AIC", "MAPE", "Ljung-Box"]].to_latex(escape=False), "ARIMAX_both": wa_results_2var[["order","AIC", "MAPE", "Ljung-Box"]].to_latex(escape=False)},
                    "CA": {"Univar": ca_results_univar[["order","AIC", "MAPE", "Ljung-Box"]].to_latex(escape=False), "ARIMAX_staff": ca_results_staff[["order","AIC", "MAPE", "Ljung-Box"]].to_latex(escape=False),
                           "ARIMAX_state": ca_results_state[["order","AIC", "MAPE", "Ljung-Box"]].to_latex(escape=False), "ARIMAX_both": ca_results_2var[["order","AIC", "MAPE", "Ljung-Box"]].to_latex(escape=False)}}

    for k,v in latex_tables["CA"].items():
        print(f'{k}\n')
        print(v)
    fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    plot_grid_search(ca_results_univar, ax, label="Univar", symbol="o")
    plot_grid_search(ca_results_state, ax, label="State", symbol="^")
    plot_grid_search(ca_results_staff, ax, label="Staff", symbol="v")
    plot_grid_search(ca_results_2var, ax, label="Both", symbol="s")

    ax[0].vlines(pdq_range.index[:17], ymin=0, ymax=3, lw=1, ls="--", alpha=0.3)
    ax[1].vlines(pdq_range.index[:17], ymin=-620, ymax=-560, lw=1, ls="--", alpha=0.3)
    ax[2].vlines(pdq_range.index[:17], ymin=-610, ymax=-550, lw=1, ls="--", alpha=0.3)

    # SOMETHING IS REALLY WRONG WITH THE XTICKS AND WHERE THE RESULTS ARE PLOTTED
    ax[0].set_xticks(pdq_range.index[:17])
    ax[1].set_xticks(pdq_range.index[:17])
    ax[2].set_xticks(pdq_range.index[:17])
    ax[2].set_xticklabels(pdq_range["order"][:17], rotation=90)
    ax[0].set_ylabel("MAPE")
    ax[0].set_yticks([0, .5, 1, 1.5, 2, 2.5])
    ax[0].set_yticklabels(['0%', '.5%', '1%', '1.5%', '2%', '2.5%'])
    ax[1].set_ylabel("AIC")
    ax[2].set_ylabel("Sum AIC + MAPE")
    ax[0].legend()
    # plt.show()

    fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    plot_grid_search(wa_results_univar, ax, label="Univar", symbol="o")
    plot_grid_search(wa_results_state, ax, label="State", symbol="^")
    plot_grid_search(wa_results_staff, ax, label="Staff", symbol="v")
    plot_grid_search(wa_results_2var, ax, label="Both", symbol="s")

    ax[0].vlines(pdq_range.index[:10], ymin=0, ymax=12, lw=1, ls="--", alpha=0.3)
    ax[1].vlines(pdq_range.index[:10], ymin=-530, ymax=-480, lw=1, ls="--", alpha=0.3)
    ax[2].vlines(pdq_range.index[:10], ymin=-520, ymax=-450, lw=1, ls="--", alpha=0.3)

    ax[0].set_xticks(pdq_range.index[:10])
    ax[1].set_xticks(pdq_range.index[:10])
    ax[2].set_xticks(pdq_range.index[:10])
    ax[2].set_xticklabels(pdq_range["order"][:10], rotation=90)
    ax[0].set_ylabel("MAPE")
    ax[0].set_yticks([0, 2, 4, 6, 8, 10])
    ax[0].set_yticklabels(['0%', '2%', '4%', '6%', '8%', '10%'])
    ax[1].set_ylabel("AIC")
    ax[2].set_ylabel("Sum AIC + MAPE")
    ax[0].legend()
    plt.show()

    # all_results_wa = pd.read_csv("results/WA_ARIMA_staff_grid_diff1.csv", index_col=0).to_latex(escape=False)
    # all_results_ca = pd.read_csv("results/CA_ARIMA_staff_grid_diff1.csv", index_col=0).to_latex(escape=False)

    return latex_tables

def plot_multiple_predictions(results, x_index, true, labels):
    n_results = len(results)
    line_styles = ["--", ":", "-.", "--", ":", "-."]
    fig, ax = plt.subplots(1, n_results, figsize=(14,7), sharey=True)

    x_ticks = x_index[::6]

    for n in range(n_results):
        result = results[n]
        predictions = result["predictions"]["predictions"]
        ax[n].plot(x_index, true, color='red', lw=2, label='Actual')

        ax[n].plot(x_index, predictions.summary_frame()['mean'], ls=line_styles[n], label = labels[n])

        ax[n].plot(x_index, predictions.summary_frame()['mean_ci_lower'],marker = '.', linestyle = ':')
        ax[n].plot(x_index, predictions.summary_frame()['mean_ci_upper'],marker = '.', linestyle = ':')

        ax[n].fill_between(x_index, predictions.summary_frame()['mean_ci_lower'],predictions.summary_frame()['mean_ci_upper'], alpha = 0.1)

        ax[n].legend(frameon=False, loc='upper left')
        ax[n].set_xticks(x_ticks)
        ax[n].set_xticklabels(x_ticks, rotation=35)
    ax[0].set_ylabel("Fraction of infected cumulative population")

    plt.show()
