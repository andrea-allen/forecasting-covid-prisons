import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import statsmodels.tsa.stattools
import matplotlib
import statistics
from statsmodels.tsa.stattools import adfuller
font = {'family' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

main_palette = sns.color_palette("icefire", n_colors=6)

## TODO should make a pre-clean-EDA function that can then be called by project main

def load_raw_data(saved_file_date=None):
    # saved_file_date = 04-06-22
    if saved_file_date is None:
        today = datetime.date.today().strftime("%m-%d-%y")
        state_historic_data_df = pd.read_csv(
            "https://raw.githubusercontent.com/uclalawcovid19behindbars/data/master/historical-data/historical_state_counts.csv")
        state_historic_data_df.to_csv(f"./raw_data/historic_state_counts_{today}.csv")
        anchored_df = pd.read_csv(
            "https://raw.githubusercontent.com/uclalawcovid19behindbars/data/master/anchored-data/state_aggregate_denominators.csv")
        anchored_df.to_csv(f"./raw_data/state_aggregate_denominators_{today}.csv")
    else:
        anchored_df = pd.read_csv(f"./raw_data/state_aggregate_denominators_{saved_file_date}.csv")
        state_historic_data_df = pd.read_csv(f"./raw_data/historic_state_counts_{saved_file_date}.csv")
    nyt = load_nyt_data()
    return {"cases" : state_historic_data_df, "anchored": anchored_df, "nyt":nyt}

def get_california(df):
    return df[df["State"]=="California"]

def get_washington(df):
    return df[df["State"]=="Washington"]

def plot_ts(df, state_name, date_range_tuple, column_name, ax, norm_pop=False, anchored_df=None, diff=False, color=None):
    state_mask = df[df["State"]==state_name]
    mask = (state_mask['Date'] > date_range_tuple[0]) & (state_mask['Date'] <= date_range_tuple[1])
    date_mask = state_mask.loc[mask]
    date_mask = date_mask[["Date", column_name]].dropna()
    res_denom = 1
    staff_denom = 1
    if norm_pop:
        res_denom = anchored_df[anchored_df.State == state_name]['Residents.Population'].iloc[0]
        staff_denom = anchored_df[anchored_df.State == state_name]['Staff.Population'].iloc[0]
        if np.isnan(res_denom) or np.isnan(staff_denom):
            print(f"One of resident or staff population was NaN for {state_name}, setting denominator to 1 and 1")
            res_denom = 1
            staff_denom = 1
    if diff:
        ax.plot(date_mask["Date"], date_mask[column_name].diff(1), lw=1, label=f"{state_name}_{column_name}", color=color)
    else:
        if 'Staff' in column_name:
            denom = staff_denom
        elif 'Resid' in column_name:
            denom = res_denom
        else:
            denom = 1
        ax.plot(date_mask["Date"], date_mask[column_name]/denom,  lw=1, label=f"{state_name}", color=color)
    ax.set_xticks(ticks=date_mask["Date"][::10])
    ax.set_xticklabels(labels = [ date.date() for date in date_mask["Date"][::10]], rotation=45)
    ax.legend(frameon=False, loc="upper left")
    # ax.set_xlabel('Date')
    if norm_pop:
        ax.set_ylabel(f'Pct population {column_name}')
    else:
        ax.set_ylabel(f'Population {column_name}')
    if diff:
        ax.set_ylabel(f'Pop {column_name} DIFF')
    return ax

def plot_hist(df, state_name, date_range_tuple, column_name, ax, norm_pop=False, anchored_df=None, diff=False):
    state_mask = df[df["State"]==state_name]
    mask = (state_mask['Date'] > date_range_tuple[0]) & (state_mask['Date'] <= date_range_tuple[1])
    date_mask = state_mask.loc[mask]
    date_mask = date_mask[["Date", column_name]].dropna()
    res_denom = 1
    staff_denom = 1
    if norm_pop:
        res_denom = anchored_df[anchored_df.State == state_name]['Residents.Population'].iloc[0]
        staff_denom = anchored_df[anchored_df.State == state_name]['Staff.Population'].iloc[0]
        if np.isnan(res_denom) or np.isnan(staff_denom):
            print(f"One of resident or staff population was NaN for {state_name}, setting denominator to 1 and 1")
            res_denom = 1
            staff_denom = 1
    if diff:
        ax.hist(date_mask[column_name].diff(1),  alpha=0.6, bins='auto', label=f"{state_name}_{column_name}")
    else:
        if 'Staff' in column_name:
            denom = staff_denom
        elif 'Resid' in column_name:
            denom = res_denom
        else:
            denom = 1
        ax.hist(date_mask[column_name]/denom,  alpha=0.6, bins='auto', label=f"{state_name}_{column_name}")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    return ax

# For plotting all the timeseries:
def plot_all_state_DOC(df, anchored_df, state_names, column_name, axs, diff=False):
    current_row = 0
    current_col = 0
    max_col_idx = len(axs[0])-1
    for state in state_names:
        plot_ts(df, state, (datetime.datetime(2020, 3, 20), datetime.datetime(2022, 4, 2)), f"Residents.{column_name}", axs[current_row, current_col], norm_pop=True, diff=diff, anchored_df=anchored_df)
        plot_ts(df, state, (datetime.datetime(2020, 3, 20), datetime.datetime(2022, 4, 2)), f"Staff.{column_name}", axs[current_row, current_col], norm_pop=True, diff=diff, anchored_df=anchored_df)
        if current_col < max_col_idx:
            current_col += 1
        elif current_col == max_col_idx:
            current_col = 0
            current_row += 1
    plt.tight_layout()

def plot_all_state_hists(df, state_names, column_name,  axs, diff=False):
    current_row = 0
    current_col = 0
    max_col_idx = 2
    for state in state_names:
        plot_hist(df, state, (datetime.datetime(2020, 3, 20), datetime.datetime(2022, 4, 2)),
                  f"Residents.{column_name}", axs[current_col], norm_pop=False, diff=diff)
        plot_hist(df, state, (datetime.datetime(2020, 3, 20), datetime.datetime(2022, 4, 2)),
                  f"Staff.{column_name}", axs[current_col], norm_pop=False, diff=diff)
        if current_col < max_col_idx:
            current_col += 1
        elif current_col == max_col_idx:
            current_col = 0
            current_row += 1
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    # plt.tight_layout()


def missingness(df, state_names, axs):
    state_df = df[df.State == state_names[0]]
    date_labels = [f"{state_df.Date.iloc[i].month}-{state_df.Date.iloc[i].year}" for i in range(len(state_df))]
    state_df = state_df.drop(columns=['Unnamed: 0', "Staff.Vadmin", "Residents.Vadmin", "State", "Date"])
    sns.heatmap(state_df.isna().transpose(), cmap="YlGnBu", cbar=False, cbar_kws={"label": "Missing Data"}, ax=axs[0, 0])
    x_ticks = axs[0,0].get_xticks()
    tick_interval = round(len(state_df)/len(x_ticks))
    axs[0,0].set_xticklabels(date_labels[::tick_interval], rotation=90)
    axs[0,0].set_title(f"{state_names[0]}")
    state_df = df[df.State == state_names[1]]
    date_labels = [f"{state_df.Date.iloc[i].month}-{state_df.Date.iloc[i].year}" for i in range(len(state_df))]
    state_df = state_df.drop(columns=['Unnamed: 0', "Staff.Vadmin", "Residents.Vadmin", "State", "Date"])
    sns.heatmap(state_df.isna().transpose(), cmap="YlGnBu", cbar=False, cbar_kws={"label": "Missing Data"}, ax=axs[0, 1])
    x_ticks = axs[0,1].get_xticks()
    tick_interval = round(len(state_df)/len(x_ticks))
    axs[0,1].set_xticklabels(date_labels[::tick_interval], rotation=90)
    axs[0, 1].set_title(f"{state_names[1]}")
    state_df = df[df.State == state_names[2]]
    date_labels = [f"{state_df.Date.iloc[i].month}-{state_df.Date.iloc[i].year}" for i in range(len(state_df))]
    state_df = state_df.drop(columns=['Unnamed: 0', "Staff.Vadmin", "Residents.Vadmin", "State", "Date"])
    sns.heatmap(state_df.isna().transpose(), cmap="YlGnBu", cbar=False, cbar_kws={"label": "Missing Data"}, ax=axs[1, 0])
    x_ticks = axs[1,0].get_xticks()
    tick_interval = round(len(state_df)/len(x_ticks))
    axs[1,0].set_xticklabels(date_labels[::tick_interval], rotation=90)
    axs[1, 0].set_title(f"{state_names[2]}")
    state_df = df[df.State == state_names[3]]
    date_labels = [f"{state_df.Date.iloc[i].month}-{state_df.Date.iloc[i].year}" for i in range(len(state_df))]
    state_df = state_df.drop(columns=['Unnamed: 0', "Staff.Vadmin", "Residents.Vadmin", "State", "Date"])
    sns.heatmap(state_df.isna().transpose(), cmap="YlGnBu", cbar=False, cbar_kws={"label": "Missing Data"}, ax=axs[1, 1])
    x_ticks = axs[1,1].get_xticks()
    tick_interval = round(len(state_df)/len(x_ticks))
    axs[1,1].set_xticklabels(date_labels[::tick_interval], rotation=90)
    axs[1, 1].set_title(f"{state_names[3]}")
    plt.tight_layout()

def date_distribution_multistate(df, state_names, axs):
    max_col = len(axs[0])-1
    curr_row = 0
    curr_col = 0
    for state in state_names:
        state_df = df[df.State == state][["Date", "Residents.Confirmed"]]
        state_df = state_df.dropna()
        values, counts = np.unique([state_df["Date"].diff(1).iloc[i].days
                                                                     for i in range(1, len(state_df))], return_counts=True)
        axs[curr_row, curr_col].vlines(values, 0, counts, color='green', lw=4, alpha=0.75,label="res confirmed daily gap")
        # axs[curr_row, curr_col].hist([state_df["Date"].diff(1).iloc[i].days
        #                                                              for i in range(1, len(state_df))],
        #                              color='green', alpha=0.5, bins='auto',  label="res confirmed daily gap")
        # axs[curr_row, curr_col].scatter(np.arange(len(state_df)-1), [state_df["Date"].diff(1).iloc[i].days
        #                                                              for i in range(1, len(state_df))], s=4,
        #                                 fc="none", marker='^',
        #                                 color='green', label="res confirmed daily gap")
        state_df = df[df.State == state][["Date", "Staff.Confirmed"]]
        state_df = state_df.dropna()
        values, counts = np.unique([state_df["Date"].diff(1).iloc[i].days
                                    for i in range(1, len(state_df))], return_counts=True)
        axs[curr_row, curr_col].vlines(values, 0, counts, color='orange', lw=4, alpha=0.75, label="staff confirmed daily gap")
        # axs[curr_row, curr_col].hist([state_df["Date"].diff(1).iloc[i].days
        #                                                              for i in range(1, len(state_df))],
        #                              color='orange', alpha=0.5, bins='auto',  label="staff confirmed daily gap")
        # axs[curr_row, curr_col].scatter(np.arange(len(state_df)-1), [state_df["Date"].diff(1).iloc[i].days
        #                                                              for i in range(1, len(state_df))], s=4,
        #                                 fc="none", marker='v',
        #                                 color='orange', label="staff confirmed daily gap")
        axs[curr_row, curr_col].set_title(f"{state}")
        axs[curr_row, curr_col].legend(frameon=False, loc='upper left')
        axs[curr_row, curr_col].set_ylabel('Count')
        axs[curr_row, curr_col].set_xlabel('days between observations')
        if curr_col < max_col:
            curr_col += 1
        else:
            curr_col = 0
            curr_row += 1
    plt.tight_layout()

def eda(data):
    df = data["cases"]
    anchored = data["anchored"]
    # convert dates to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    fig, axs = plt.subplots(2,2, sharex=True)
    plot_all_state_DOC(df, anchored, ["California", "Washington", "West Virginia", "Pennsylvania"], "Confirmed", axs)
    fig, axs = plt.subplots(2,2, sharex=True)
    plot_all_state_DOC(df, anchored, ["California", "Washington", "West Virginia", "Pennsylvania"], "Confirmed", axs, diff=True)
    fig, axs = plt.subplots(2,2)
    plot_all_state_hists(df, ["California", "Washington", "West Virginia", "Pennsylvania"], "Confirmed", axs, diff=True)

    fig, axs = plt.subplots(2,2, sharex=True)
    plot_all_state_DOC(df, anchored, ["California", "Washington", "West Virginia", "Pennsylvania"], "Initiated", axs)
    fig, axs = plt.subplots(2,2, sharex=True)
    plot_all_state_DOC(df, anchored, ["California", "Washington", "West Virginia", "Pennsylvania"], "Initiated", axs, diff=True)
    fig, axs = plt.subplots(2,2)
    plot_all_state_hists(df, ["California", "Washington", "West Virginia", "Pennsylvania"], "Initiated", axs, diff=True)
    plt.show()

    fig, axs = plt.subplots(2,2, figsize=(12,8))
    missingness(df, ["California", "Washington", "West Virginia", "Pennsylvania"], axs)

    figure2, axs2 = plt.subplots(2, 2)
    date_distribution_multistate(df, ["California", "Washington", "West Virginia", "Pennsylvania"], axs2)

    fig, axs = plt.subplots(2,2, figsize=(12,8), sharex=True)
    statistics.rolling_corr_plots(df[df.State=="California"], axs[0,0]) #TODO fix state
    statistics.rolling_corr_plots(df[df.State=="Washington"], axs[0,1])
    statistics.rolling_corr_plots(df[df.State=="West Virginia"], axs[1,0])
    statistics.rolling_corr_plots(df[df.State=="Pennsylvania"], axs[1,1])
    plt.show()

    statistics.acf_plot("California")
    plt.show()
    ca_res_conf = statistics.ad_fuller_results("California", "Residents.Confirmed")
    ca_staff_conf = statistics.ad_fuller_results("California", "Staff.Confirmed")
    ca_res_init = statistics.ad_fuller_results("California", "Residents.Initiated")
    ca_staff_init = statistics.ad_fuller_results("California", "Staff.Initiated")
    wv_res_conf = statistics.ad_fuller_results("West Virginia", "Residents.Confirmed")
    wv_staff_conf = statistics.ad_fuller_results("West Virginia", "Staff.Confirmed")
    wv_res_init = statistics.ad_fuller_results("West Virginia", "Residents.Initiated")
    wv_staff_init = statistics.ad_fuller_results("West Virginia", "Staff.Initiated")
    pen_res_conf = statistics.ad_fuller_results("Pennsylvania", "Residents.Confirmed")
    pen_staff_conf = statistics.ad_fuller_results("Pennsylvania", "Staff.Confirmed")
    pen_res_init = statistics.ad_fuller_results("Pennsylvania", "Residents.Initiated")
    pen_staff_init = statistics.ad_fuller_results("Pennsylvania", "Staff.Initiated")
    wa_res_conf = statistics.ad_fuller_results("Washington", "Residents.Confirmed")
    wa_staff_conf = statistics.ad_fuller_results("Washington", "Staff.Confirmed")
    wa_res_init = statistics.ad_fuller_results("Washington", "Residents.Initiated")
    wa_staff_init = statistics.ad_fuller_results("Washington", "Staff.Initiated")
    stationarity_df = pd.DataFrame([ca_res_conf, ca_staff_conf, wa_res_conf, wa_staff_conf])
    tex_string = stationarity_df.to_latex()

    plt.show()

def eda_plots_for_paper(data):
    df = data["cases"]
    anchored = data["anchored"]
    nyt = data["nyt"]
    # convert dates to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    nyt["date"] = pd.to_datetime(nyt["date"])

    palette = sns.color_palette("icefire", 4)

    # fig, axs = plt.subplots(3, 1, sharex=True)

    # plot_ts(df, "California", (datetime.datetime(2020, 3, 20), datetime.datetime(2022, 4, 2)),
    #         "Residents.Confirmed", axs[0], anchored_df=anchored, color=palette[0])
    # plot_ts(df, "California", (datetime.datetime(2020, 3, 20), datetime.datetime(2022, 4, 2)),
    #         "Staff.Confirmed", axs[1], anchored_df=anchored, color=palette[1])
    # state_cases = get_nyt_state(nyt, "California")
    # axs[2].plot(state_cases["date"], state_cases["cases"], lw=1, color=palette[2], label="State")
    # axs[2].legend(frameon=False)
    # axs[0].set_ylabel('Residents')
    # axs[1].set_ylabel('Staff')
    # axs[2].set_ylabel('State')
    # axs[2].set_xticks(state_cases["date"][::45])
    # axs[2].set_xticklabels(state_cases["date"][::45].dt.date, rotation=35)

    # fig, axs = plt.subplots(3, 1, sharex=True)
    # plot_ts(df, "Washington", (datetime.datetime(2020, 3, 20), datetime.datetime(2022, 4, 2)),
    #         "Residents.Confirmed", axs[0], anchored_df=anchored, color=palette[0])
    # plot_ts(df, "Washington", (datetime.datetime(2020, 3, 20), datetime.datetime(2022, 4, 2)),
    #         "Staff.Confirmed", axs[1], anchored_df=anchored, color=palette[1])
    # state_cases = get_nyt_state(nyt, "Washington")
    # axs[2].plot(state_cases["date"], state_cases["cases"], lw=1, color=palette[2], label="State")
    # axs[2].legend(frameon=False)
    # axs[0].set_ylabel('Residents')
    # axs[1].set_ylabel('Staff')
    # axs[2].set_ylabel('State')
    # axs[2].set_xticks(state_cases["date"][::45])
    # axs[2].set_xticklabels(state_cases["date"][::45].dt.date, rotation=35)

    fig, axs = plt.subplots(3, 1, sharex=True)
    plot_ts(df, "California", (datetime.datetime(2020, 3, 20), datetime.datetime(2022, 4, 2)), "Residents.Confirmed",
            axs[0], anchored_df=anchored, diff=True, color=palette[0])
    plot_ts(df, "California", (datetime.datetime(2020, 3, 20), datetime.datetime(2022, 4, 2)), "Staff.Confirmed",
            axs[1], anchored_df=anchored, diff=True, color=palette[1])
    state_cases = get_nyt_state(nyt, "California")
    axs[2].plot(state_cases["date"][1:], state_cases["cases"].diff(1).dropna(), lw=1, color=palette[2], label="State")
    axs[2].legend(frameon=False)
    axs[0].set_ylabel('Residents')
    axs[1].set_ylabel('Staff')
    axs[2].set_ylabel('State')
    axs[2].set_xticks(state_cases["date"][::45])
    axs[2].set_xticklabels(state_cases["date"][::45].dt.date, rotation=35)

    fig, axs = plt.subplots(3, 1, sharex=True)
    plot_ts(df, "Washington", (datetime.datetime(2020, 3, 20), datetime.datetime(2022, 4, 2)), "Residents.Confirmed",
            axs[0], anchored_df=anchored, diff=True, color=palette[0])
    plot_ts(df, "Washington", (datetime.datetime(2020, 3, 20), datetime.datetime(2022, 4, 2)), "Staff.Confirmed",
            axs[1], anchored_df=anchored, diff=True, color=palette[1])
    state_cases = get_nyt_state(nyt, "Washington")
    axs[2].plot(state_cases["date"][1:], state_cases["cases"].diff(1).dropna(), lw=1, color=palette[2], label="State")
    axs[2].legend(frameon=False)
    axs[0].set_ylabel('Residents')
    axs[1].set_ylabel('Staff')
    axs[2].set_ylabel('State')
    axs[2].set_xticks(state_cases["date"][::45])
    axs[2].set_xticklabels(state_cases["date"][::45].dt.date, rotation=35)


    fig, axs = plt.subplots(1,2, figsize=(9,5))
    plot_all_state_hists(df, ["California", "Washington"], "Confirmed", axs, diff=True)

    # plt.tight_layout()
    plt.show()

def eda_post_cleaning(data):
    sns.set_palette("icefire", n_colors=5)
    # fig, ax = plt.subplots(2, 1, sharex=True)
    #
    # ax[0].plot(get_california(data).index, get_california(data)["Residents.Confirmed.Normed"].diff(1), lw=1.5, alpha=0.8, label="CA Residents (normalized)")
    # ax[0].plot(get_california(data).index, get_california(data)["Staff.Confirmed.Normed"].diff(1), lw=1.5, alpha=0.8, label="CA Staff (normalized)")
    # ax[0].plot(get_california(data).index, get_california(data)["State.Cases.Normed"].diff(1), lw=1.5, alpha=0.8, label="CA State (normalized)")
    # ax[0].legend()
    # # ax[0].set_xlabel("Time")
    # # ax[0].set_ylabel("New weekly fraction of infected population")
    # ax[1].plot(get_washington(data).index, get_washington(data)["Residents.Confirmed.Normed"].diff(1), lw=1.5, alpha=0.8, label="WA Residents (normalized)")
    # ax[1].plot(get_washington(data).index, get_washington(data)["Staff.Confirmed.Normed"].diff(1), lw=1.5, alpha=0.8, label="WA Staff (normalized)")
    # ax[1].plot(get_washington(data).index, get_washington(data)["State.Cases.Normed"].diff(1), lw=1.5, alpha=0.8, label="WA State (normalized)")
    # ax[1].legend()
    #
    # x_ticks = get_california(data).index[::10]
    # ax[0].set_ylabel("New cases as % of population")
    # ax[1].set_ylabel("New cases as % of population")
    #
    # ax[1].set_xticks(x_ticks)
    # ax[1].set_xticklabels(x_ticks, rotation=-35)
    # plt.show()

    #smoothed
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10,6))

    smoothing_window = 3
    ax[0].plot(get_california(data).index, get_california(data)["Residents.Confirmed.Normed"].rolling(window=smoothing_window).mean().diff(1), lw=1.5,
               alpha=0.8, label="CA Residents (normalized)")
    ax[0].plot(get_california(data).index, get_california(data)["Staff.Confirmed.Normed"].rolling(window=smoothing_window).mean().diff(1), lw=1.5, alpha=0.8,
               label="CA Staff (normalized)")
    ax[0].plot(get_california(data).index, get_california(data)["State.Cases.Normed"].rolling(window=smoothing_window).mean().diff(1), lw=1.5, alpha=0.8,
               label="CA State (normalized)")
    ax[0].legend()
    # ax[0].set_xlabel("Time")
    # ax[0].set_ylabel("New weekly fraction of infected population")
    ax[1].plot(get_washington(data).index, get_washington(data)["Residents.Confirmed.Normed"].rolling(window=smoothing_window).mean().diff(1), lw=1.5,
               alpha=0.8, label="WA Residents (normalized)")
    ax[1].plot(get_washington(data).index, get_washington(data)["Staff.Confirmed.Normed"].rolling(window=smoothing_window).mean().diff(1), lw=1.5, alpha=0.8,
               label="WA Staff (normalized)")
    ax[1].plot(get_washington(data).index, get_washington(data)["State.Cases.Normed"].rolling(window=smoothing_window).mean().diff(1), lw=1.5, alpha=0.8,
               label="WA State (normalized)")
    ax[1].legend()

    x_ticks = get_california(data).index[::10]
    ax[0].set_ylabel("New cases as % of population")
    ax[1].set_ylabel("New cases as % of population")

    ax[1].set_xticks(x_ticks)
    ax[1].set_xticklabels(x_ticks, rotation=-35)
    plt.show()

def plot_staff_state_corr(ca_df, wa_df):
    ## Correlation plot for staff and state
    fig, axs = plt.subplots(2, 1, sharex=True)
    coef_ca = np.corrcoef(ca_df["State.Cases.Normed"].diff(1).dropna(),
                          ca_df["Staff.Confirmed.Normed"].diff(1).dropna())[0][1]
    axs[0].scatter(ca_df["State.Cases.Normed"].diff(1).dropna(),
                   ca_df["Staff.Confirmed.Normed"].diff(1).dropna(),
                   alpha=0.7, label=f"CA, corr={np.round(coef_ca, 2)}",
                   color=main_palette[4])
    axs[0].legend()
    coef_wa = np.corrcoef(wa_df["State.Cases.Normed"].diff(1).dropna(),
                          wa_df["Staff.Confirmed.Normed"].diff(1).dropna())[0][1]
    axs[1].scatter(wa_df["State.Cases.Normed"].diff(1).dropna(),
                   wa_df["Staff.Confirmed.Normed"].diff(1).dropna(),
                   alpha=0.7, label=f"WA, corr={np.round(coef_wa, 2)}",
                   color=main_palette[5])
    axs[1].legend()
    axs[1].set_xlabel("New weekly cases in state as % of population")
    axs[0].set_ylabel("New weekly cases in \nstaff as % of staff pop")
    axs[1].set_ylabel("New weekly cases in \nstaff as % of staff pop")
    plt.show()



def load_nyt_data():
    nyt_url_states = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
    nyt_url_counties = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    df = pd.read_csv(nyt_url_states, na_filter=True)
    return df

def get_nyt_state(df, state):
    return df[df["state"]==state]
