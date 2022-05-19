import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import statsmodels.tsa.stattools
import matplotlib
font = {'family' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)
sns.set_palette("icefire")


def check_neg_diffs(df, col):
    neg_diff_idx = np.where(df[col].diff(1) < 0)[0]
    start_idx = df.index[0]
    corrections = {}
    if len(neg_diff_idx) == 0:
        print("There were no negative diffs to correct")
        return df, corrections
    for d in neg_diff_idx:
        problem_idx = start_idx + d
        last_idx = problem_idx - 1
        next_idx = problem_idx + 1
        new_val = (df[col].loc[next_idx] + df[col].loc[last_idx]) / 2
        new_val = df[col].loc[last_idx]
        df[col].loc[problem_idx] = new_val
        corrections[problem_idx] = new_val
    return df, corrections

def recursively_fix_diffs(df, col, iterations):
    corrs = []
    for i in range(iterations):
        corrections = check_neg_diffs(df, col)
        corrs.append(corrections)
    return df, corrs

def clean_california_residents(df):
    ca_df = df[df["State"]=="California"]
    print(ca_df["Residents.Confirmed"].loc[54:57])
    ca_df, corrs = check_neg_diffs(ca_df, "Residents.Confirmed")
    ca_df, corrs2 = check_neg_diffs(ca_df, "Residents.Confirmed")
    ca_df, corrs3 = check_neg_diffs(ca_df, "Residents.Confirmed")
    ca_df, corrs4 = check_neg_diffs(ca_df, "Residents.Confirmed")
    ca_df, corrs5 = check_neg_diffs(ca_df, "Residents.Confirmed")
    print(ca_df["Residents.Confirmed"].loc[54:57])
    print(corrs)
    print(np.where(ca_df["Residents.Confirmed"].diff(1) < 0)[0])
    return df

def clean_california_staff(df):
    # TODO: need to ensure that the entire df is being returned after
    ca_df = df[df["State"] == "California"]
    print(np.where(ca_df["Staff.Confirmed"].diff(1) < 0)[0])
    ca_df, c_staff1 = check_neg_diffs(ca_df, "Staff.Confirmed")
    print(np.where(ca_df["Staff.Confirmed"].diff(1) < 0)[0])
    ca_df, c_staff2 = check_neg_diffs(ca_df, "Staff.Confirmed")
    print(np.where(ca_df["Staff.Confirmed"].diff(1) < 0)[0])
    ca_df, c_staff3 = check_neg_diffs(ca_df, "Staff.Confirmed")
    print(np.where(ca_df["Staff.Confirmed"].diff(1) < 0)[0])
    ca_df, c_staff4 = check_neg_diffs(ca_df, "Staff.Confirmed")
    print(np.where(ca_df["Staff.Confirmed"].diff(1) < 0)[0])
    ca_df, c_staff5 = check_neg_diffs(ca_df, "Staff.Confirmed")
    print(np.where(ca_df["Staff.Confirmed"].diff(1) < 0)[0])
    return df

def clean_washington_residents(df):
    wa_df = df[df["State"] == "Washington"]
    wa_df, corrs = check_neg_diffs(wa_df, "Residents.Confirmed")
    print(np.where(wa_df["Residents.Confirmed"].diff(1) < 0)[0])
    return df

def clean_washington_staff(df):
    wa_df = df[df["State"] == "Washington"]
    print(np.where(wa_df["Staff.Confirmed"].diff(1) < 0)[0])
    wa_df, c_staff1 = check_neg_diffs(wa_df, "Staff.Confirmed")
    return df


def cleaning_process(data):
    state_historic_data_df = data

    # convert dates to datetime
    state_historic_data_df["Date"] = pd.to_datetime(state_historic_data_df["Date"])

    # Cleaning the data and imputing missing/bad values for residents, CA
    fig, ax = plt.subplots(1, 2)
    diffs_raw = state_historic_data_df[state_historic_data_df["State"]=="California"]["Residents.Confirmed"].diff(1)
    diffs_raw_pos = diffs_raw[diffs_raw >= 0]
    diffs_raw_neg = diffs_raw[diffs_raw < 0]
    ax[0].hist(diffs_raw_neg/len(diffs_raw_neg), bins=3, density=False, label="Negative valued diff", color='red')
    ax[0].hist(diffs_raw_pos/len(diffs_raw_pos), bins=100,density=False, label="Positive valued diff")
    ax[0].legend()
    ax[0].set_xlim([-10, 30])
    ax[0].set_xlabel("California confirmed residents")

    df = clean_california_residents(state_historic_data_df)

    cleaned_diff = df[df["State"]=="California"]["Residents.Confirmed"].diff(1)
    ax[1].hist(cleaned_diff/len(cleaned_diff), bins=100)
    ax[1].set_xlim([-10, 30])
    ax[1].set_xlabel("Corrected data")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # Cleaning the data and imputing missing/bad values for staff, CA
    diffs_raw = df[df["State"]=="California"]["Staff.Confirmed"].diff(1)
    diffs_raw_pos = diffs_raw[diffs_raw >= 0]
    diffs_raw_neg = diffs_raw[diffs_raw < 0]
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(diffs_raw_neg/len(diffs_raw_neg), bins=20, density=False, label="Negative valued diff", color='red')
    ax[0].hist(diffs_raw_pos/len(diffs_raw_pos), bins=100, density=False, label="Positive valued diff")
    ax[0].set_xlim([-10, 30])
    ax[0].set_xlabel("California confirmed staff")
    ax[0].legend()

    df = clean_california_staff(df)

    cleaned_diff = df[df["State"]=="California"]["Staff.Confirmed"].diff(1)
    ax[1].hist(cleaned_diff/len(cleaned_diff), bins=100)
    ax[1].set_xlim([-10, 30])
    ax[1].legend()
    ax[1].set_xlabel("Corrected data")
    plt.tight_layout()
    plt.show()

    # # Cleaning the data and imputing missing/bad values for residents, WA
    # # All values for Washington were valid
    # fig, ax = plt.subplots(1, 2)
    # diffs_raw = df[df["State"]=="Washington"]["Residents.Confirmed"].diff(1)
    # diffs_raw_pos = diffs_raw[diffs_raw >= 0]
    # diffs_raw_neg = diffs_raw[diffs_raw < 0]
    # ax[0].hist(diffs_raw_neg/len(diffs_raw_neg), density=False, )
    # ax[0].hist(diffs_raw_pos/len(diffs_raw_pos), density=False, )
    # ax[0].set_xlim([-10, 30])
    #
    # df = clean_washington_residents(df)
    #
    # cleaned_diff = df[df["State"]=="Washington"]["Residents.Confirmed"].diff(1)
    # ax[1].hist(cleaned_diff/len(cleaned_diff), )
    # ax[1].set_xlim([-10, 30])
    # plt.show()

    # Cleaning the data and imputing missing/bad values for staff, WA
    # diffs_raw = df[df["State"]=="Washington"]["Staff.Confirmed"].diff(1)
    # diffs_raw_pos = diffs_raw[diffs_raw >= 0]
    # diffs_raw_neg = diffs_raw[diffs_raw < 0]
    # fig, ax = plt.subplots(1, 2)
    # ax[0].hist(diffs_raw_neg/len(diffs_raw_neg), density=False)
    # ax[0].hist(diffs_raw_pos/len(diffs_raw_pos), density=False)
    # ax[0].set_xlim([-10, 30])
    #
    # df = clean_washington_staff(df)
    #
    # cleaned_diff = df[df["State"]=="Washington"]["Staff.Confirmed"].diff(1)
    # ax[1].hist(cleaned_diff/len(cleaned_diff),)
    # ax[1].set_xlim([-10, 30])
    # plt.show()

    return df


def merge_data(prison_df, anchored_df, nyt_df):
    ### Just get the columns we want
    simple_df = prison_df[prison_df["State"].isin(["California", "Washington"])]
    nyt_df = nyt_df[nyt_df["state"].isin(["California", "Washington"])]
    simple_df = simple_df[["Date", "State", "Residents.Confirmed", "Staff.Confirmed"]]
    static_pop_df = anchored_df[["State", "Residents.Population", "Staff.Population"]]
    nyt_df["Date"] = pd.to_datetime(nyt_df["date"])
    nyt_df["State"] = nyt_df["state"]
    nyt_df = nyt_df[["Date", "State", "cases"]]

    ### add new column that is normalized by population
    merged_df = pd.merge(simple_df, static_pop_df, how="left", on="State")
    merged_df["Residents.Confirmed.Normed"] = merged_df["Residents.Confirmed"] / merged_df["Residents.Population"]
    merged_df["Staff.Confirmed.Normed"] = merged_df["Staff.Confirmed"] / merged_df["Staff.Population"]

    merged_df = merged_df.merge(nyt_df, how="left", on=["Date", "State"])
    state_populations = pd.DataFrame([["California", 39500000], ["Washington", 7615000]], columns=["State", "State.Population"])
    merged_df = merged_df.merge(state_populations, how="left", on="State")
    merged_df["State.Cases.Normed"] = merged_df["cases"] / merged_df["State.Population"]

    ## Save transformed data
    merged_df.to_csv("./transformed_data/df_transformed_nyt.csv", index=False)
