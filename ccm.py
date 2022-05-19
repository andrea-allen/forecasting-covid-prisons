import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import skccm
import wrangle_data
from skccm.utilities import train_test_split
import skccm.data as data

sns.set_palette("icefire")
palette = sns.color_palette("icefire")

def get_data():
    merged_df = pd.read_csv("./transformed_data/df_transformed_nyt.csv")
    # merged_df = merged_df.set_index(merged_df["Date"])

    ca_df = wrangle_data.get_california(merged_df)
    ca_df = ca_df.set_index(pd.DatetimeIndex(ca_df['Date']))

    # Smoothing and up-sampling
    resampled = ca_df["Residents.Confirmed.Normed"].diff(1).dropna().resample('D')
    res_smoothed = resampled.interpolate(method='polynomial', order=3)
    resampled = ca_df["Staff.Confirmed.Normed"].diff(1).dropna().resample('D')
    staff_smoothed = resampled.interpolate(method='polynomial', order=3)
    resampled = ca_df["State.Cases.Normed"].diff(1).dropna().resample('D')
    state_smoothed = resampled.interpolate(method='polynomial', order=3)

    wa_df = wrangle_data.get_washington(merged_df)
    return res_smoothed, staff_smoothed, state_smoothed



def convergent_cross_map(x1, x2, x1_label, x2_label,):
    """
    :param x1: the variable that is hypothesized to be the driver
    :param x2: the response variable hypothesized
    :return:
    """
    fig, ax_pred = plt.subplots(2, 1, figsize=(14,5))
    ax_pred[0].plot(x1, label=x1_label, color=palette[0])
    ax_pred[0].plot(x2, label=x2_label, color=palette[4])
    ax_pred[0].legend(loc="best")
    ax_pred[0].set_ylabel("Smoothed Up-sampled\n New Cases")
    xticks = ax_pred[0].get_xticklabels()
    ax_pred[0].set_xticklabels(xticks, rotation=35)
    # plt.show()

    e1 = skccm.Embed(x1)
    e2 = skccm.Embed(x2)
    fig, ax = plt.subplots()
    ax.plot(e1.mutual_information(20))
    ax.plot(e2.mutual_information(20))
    # plt.show()

    lag = 1
    embed = 6 # should do a heatmap for lag robustness
    X1 = e1.embed_vectors_1d(lag,embed)
    X2 = e2.embed_vectors_1d(lag,embed)
    fig, ax = plt.subplots(1, 2, figsize=(14,2.5))
    ax[0].scatter(X1[:, 0], X1[:, 1], label=x1_label, color=palette[0])
    ax[0].set_xlabel('X1(t)', fontweight='bold')
    ax[0].set_ylabel('X1(t-1)', fontweight='bold')
    ax[0].legend(loc='best')
    ax[1].scatter(X2[:, 0], X2[:, 1], label=x2_label, color=palette[4])
    ax[1].set_xlabel('X2(t)', fontweight='bold')
    ax[1].set_ylabel('X2(t-1)', fontweight='bold')
    ax[1].legend(loc='best')
    plt.tight_layout()
    # plt.show()

    #split the embedded time series
    x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

    CCM = skccm.CCM() #initiate the class

    #library lengths to test
    len_tr = len(x1tr)
    lib_lens = np.arange(embed+1, len_tr, len_tr/50, dtype='int')

    #test causation
    CCM.fit(x1tr,x2tr)
    x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)

    sc1,sc2 = CCM.score()

    # fig, ax = plt.subplots()
    ax_pred[1].plot(lib_lens, sc1, label=x1_label, color=palette[0])
    ax_pred[1].plot(lib_lens, sc2, label=x2_label, color=palette[4])
    ax_pred[1].set_xlabel('Prediction vector length')
    ax_pred[1].set_ylabel('Prediction skill')
    ax_pred[1].legend()
    plt.show()

def robust(x1, x2):
    robustness_lags = np.zeros((20,55))
    robustness_embeddings = np.zeros((10, 55))

    lag_range =np.arange(1, 20)
    embedding_range =np.arange(2, 12)
    for e in embedding_range:
        e1 = skccm.Embed(x1)
        e2 = skccm.Embed(x2)
        lag = 2
        embed = e
        X1 = e1.embed_vectors_1d(lag, embed)
        X2 = e2.embed_vectors_1d(lag, embed)
        x1tr, x1te, x2tr, x2te = train_test_split(X1, X2, percent=.75)
        len_tr = len(x1tr)
        lib_lens = np.arange(embed + 1, len_tr, len_tr / (50-embed+1), dtype='int')
        CCM = skccm.CCM()
        CCM.fit(x1tr, x2tr)
        x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)

        sc1, sc2 = CCM.score()
        result = np.array(sc1)-np.array(sc2)
        robustness_embeddings[embed-2][embed+1:embed+len(result)+1] = result
    for l in lag_range:
        e1 = skccm.Embed(x1)
        e2 = skccm.Embed(x2)
        lag = l
        embed = 5
        X1 = e1.embed_vectors_1d(lag, embed)
        X2 = e2.embed_vectors_1d(lag, embed)
        x1tr, x1te, x2tr, x2te = train_test_split(X1, X2, percent=.75)
        len_tr = len(x1tr)
        lib_lens = np.arange(embed + 1, len_tr, len_tr / (50-embed+1), dtype='int')
        CCM = skccm.CCM()
        CCM.fit(x1tr, x2tr)
        x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)

        sc1, sc2 = CCM.score()
        # fig, ax = plt.subplots()
        # ax.plot(lib_lens, sc1, color=palette[0])
        # ax.plot(lib_lens, sc2, color=palette[1])
        # ax.legend()
        # plt.show()
        result = np.array(sc1)-np.array(sc2)
        robustness_lags[l][embed+1:embed+len(result)+1] = result

    fig, ax = plt.subplots(1, 2)
    sns.heatmap(robustness_embeddings, cmap="icefire", center=0, ax = ax[0], cbar_kws={'label': 'x1 skill - x2 skill'})
    sns.heatmap(robustness_lags, cmap="icefire", center=0, ax = ax[1], cbar_kws={'label': 'x1 skill - x2 skill'})
    ax[0].set_xlabel('prediction vector length')
    ax[0].set_ylabel('embedding dimension')
    ax[1].set_xlabel('prediction vector length')
    ax[1].set_ylabel('lag dimension')
    lib_lens = np.arange(2, len_tr, len_tr / (50 - 2), dtype='int')
    ax[1].set_xticks(np.arange(2, len(lib_lens) + 2)[::5])
    ax[0].set_xticks(np.arange(2, len(lib_lens) + 2)[::5])
    ax[1].set_xticklabels(lib_lens[::5])
    ax[0].set_xticklabels(lib_lens[::5])
    ax[0].set_yticks(np.arange(10))
    ax[0].set_yticklabels(np.arange(2,12))
    plt.show()
    return robustness_embeddings







def examples():
    rx1 = 3.72  # determines chaotic behavior of the x1 series
    rx2 = 3.72  # determines chaotic behavior of the x2 series
    b12 = 0.2  # Influence of x1 on x2
    b21 = 0.01  # Influence of x2 on x1
    ts_length = 1000
    x1, x2 = data.coupled_logistic(rx1, rx2, b12, b21, ts_length)
    # xs = np.arange(0,ts_length)
    # x1 = np.sin(xs) + .1*np.random.randint(0,100, ts_length)
    # x2 = np.zeros(ts_length)
    # x2[2:] = x1[2:] + x1[1:ts_length-1]
    ## Using these parameters, x1 has more of an influence on x2 than x2 has on x1
    fig, ax = plt.subplots(1,2)
    ax[0].plot(x1)
    ax[1].plot(x2)
    #As is clearly evident from the figure above, there is no way to tell if one series is influencing the other just by examining the time series.


    e1 = skccm.Embed(x1)
    e2 = skccm.Embed(x2)
    fig, ax = plt.subplots()
    ax.plot(e1.mutual_information(20))
    ax.plot(e2.mutual_information(20))
    plt.show()

    lag = 2
    embed = 2
    X1 = e1.embed_vectors_1d(lag, embed)
    X2 = e2.embed_vectors_1d(lag, embed)

    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(X1[:, 0], X1[:, 1], color='blue', label='X1(t)')
    ax[0].set_xlabel('X1(t)', fontweight='bold')
    ax[0].set_ylabel('X1(t-1)', fontweight='bold')
    ax[0].legend(loc='best')
    ax[1].scatter(X2[:, 0], X2[:, 1], color='red', label='X2(t)')
    ax[1].set_xlabel('X2(t)', fontweight='bold')
    ax[1].set_ylabel('X2(t-1)', fontweight='bold')
    ax[1].legend(loc='best')
    plt.show()

    # check the forecast skill as a function of library length.

    #This package diverges from the paper above in that a training set is used to rebuild the shadow manifold and
    # the testing set is used to see if nearby points on one manifold can be used to make accurate predictions
    # about the other manifold. This removes the problem of autocorrelated time series.

    # split the embedded time series
    x1tr, x1te, x2tr, x2te = train_test_split(X1, X2, percent=.75)

    CCM = skccm.CCM()  # initiate the class

    # library lengths to test
    len_tr = len(x1tr)
    lib_lens = np.arange(10, len_tr, len_tr / 20, dtype='int')

    # test causation
    CCM.fit(x1tr, x2tr)
    x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)

    sc1, sc2 = CCM.score()

    fig, ax = plt.subplots()
    ax.plot(lib_lens, sc1, label="x1")
    ax.plot(lib_lens, sc2, label="x2")
    ax.legend()
    plt.show()
    # As can be seen from the image above, x1 has a higher prediction skill. Another way to view this is that
    # information about x1 is present in the x2 time series. This leads to better forecasts for x1 using x2’s
    # reconstructed manifold. This means that x1 is driving x2 which is exactly how we set the initial conditions
    # when we generated these time series.

    ##3 Robustness:
    # test a range of values and plot a heatmap of the difference between x1 and x2,
    # for me it would be w/ different lags




# examples()

# robust(staff_smoothed, res_smoothed)
# robust(state_smoothed, res_smoothed)
# robust(res_smoothed, staff_smoothed)


