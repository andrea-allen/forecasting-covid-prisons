import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import clean_data
import wrangle_data
import statistics
import models
from ccm import *
import numpy as np
import visualize_results
import seaborn as sns

main_palette = sns.color_palette("icefire", 6)

font = {'family': 'normal',
        'size': 14}

matplotlib.rc('font', **font)

CLEAN = False
RUN = False

"""
Pre-clean EDA
"""
data = wrangle_data.load_raw_data("04-06-22")
wrangle_data.eda_plots_for_paper(data)
wrangle_data.eda(data)

"""
Loading and automating cleaning the data
"""
if CLEAN:
    data = wrangle_data.load_raw_data("04-06-22")
    cleaned_prison_df = clean_data.cleaning_process(data["cases"])
    clean_data.merge_data(cleaned_prison_df, data["anchored"], wrangle_data.load_nyt_data())

"""
Loading transformed
"""
merged_df = pd.read_csv("./transformed_data/df_transformed_nyt.csv") # Includes NYT state data
merged_df = merged_df.set_index(merged_df["Date"])

"""
Exploring cleaned merged data
"""
statistics.evaluate_stationarity(merged_df)
wrangle_data.eda_post_cleaning(merged_df)

"""
Separating CA and WA
"""
ca_df = wrangle_data.get_california(merged_df)
wa_df = wrangle_data.get_washington(merged_df)


wrangle_data.plot_staff_state_corr(ca_df, wa_df)

"""
Modeling
"""

#### Investigative fits for potential manual fit

ca_model_fitter = models.ModelFitter(ca_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=ca_df[["Staff.Confirmed.Normed"]].diff(1).dropna())

ca_model_fitter.split_train_test()
ca_model_fitter.pre_fit_analyze()
ca_model_fitter.fit_model(1,0,0)
ca_model_fitter.get_results(show=True)
plt.show()
#
# ca_model_fitter = models.ModelFitter(ca_df["Residents.Confirmed.Normed"].diff(1).dropna())
#
# ca_model_fitter.split_train_test()
# ca_model_fitter.pre_fit_analyze()
# ca_model_fitter.fit_model(1,1,0)
# ca_model_fitter.get_results(show=True)
# plt.show()
#
# ca_model_fitter = models.ModelFitter(ca_df["Residents.Confirmed.Normed"].diff(1).dropna(),
#                                      x_data=ca_df[["Staff.Confirmed.Normed", "State.Cases.Normed"]].diff(1).dropna())
#
# ca_model_fitter.split_train_test()
# ca_model_fitter.pre_fit_analyze()
# ca_model_fitter.fit_model(1,0,1)
# ca_model_fitter.get_results(show=True)
# plt.show()


"""
Grid search over parameter space for p,d,q
"""
# # California
if RUN:
    gridsearch = models.ArimaGridSearch(ca_df["Residents.Confirmed.Normed"].diff(1).dropna())
    results_univar = gridsearch.search(p_range=[0, 1, 2, 3, 4], d_range=[0,1], q_range=[0, 1, 2])
    results_univar.to_csv("./results/CA_ARIMA_grid_diff1.csv")

    gridsearch = models.ArimaGridSearch(ca_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=ca_df["Staff.Confirmed.Normed"].diff(1).dropna())
    results_1var = gridsearch.search(p_range=[0, 1, 2, 3, 4], d_range=[0,1], q_range=[0, 1, 2])
    results_1var.to_csv("./results/CA_ARIMAX_staff_grid_diff1.csv")

    gridsearch = models.ArimaGridSearch(ca_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=ca_df["State.Cases.Normed"].diff(1).dropna())
    results_1var = gridsearch.search(p_range=[0, 1, 2, 3, 4], d_range=[0,1], q_range=[0, 1, 2])
    results_1var.to_csv("./results/CA_ARIMAX_state_grid_diff1.csv")

    gridsearch = models.ArimaGridSearch(ca_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=ca_df[["Staff.Confirmed.Normed",
                                                                                          "State.Cases.Normed"]].diff(1).dropna())
    results_2var = gridsearch.search(p_range=[0, 1, 2, 3, 4], d_range=[0,1], q_range=[0, 1, 2])
    results_2var.to_csv("./results/CA_ARIMAX_staff_state_grid_diff1.csv")


# # Washington
if RUN:
    gridsearch = models.ArimaGridSearch(wa_df["Residents.Confirmed.Normed"].diff(1).dropna())
    results_univar = gridsearch.search(p_range=[0, 1, 2, 3, 4], d_range=[0,1], q_range=[0, 1, 2])
    results_univar.to_csv("./results/WA_ARIMA_grid_diff1.csv")

    gridsearch = models.ArimaGridSearch(wa_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=wa_df["Staff.Confirmed.Normed"].diff(1).dropna())
    results_1var = gridsearch.search(p_range=[0, 1, 2, 3, 4], d_range=[0,1], q_range=[0, 1, 2])
    results_1var.to_csv("./results/WA_ARIMAX_staff_grid_diff1.csv")
    #
    gridsearch = models.ArimaGridSearch(wa_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=wa_df["State.Cases.Normed"].diff(1).dropna())
    results_1var = gridsearch.search(p_range=[0, 1, 2, 3, 4], d_range=[0,1], q_range=[0, 1, 2])
    results_1var.to_csv("./results/WA_ARIMAX_state_grid_diff1.csv")
    #
    gridsearch = models.ArimaGridSearch(wa_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=wa_df[["Staff.Confirmed.Normed",
                                                                                          "State.Cases.Normed"]].diff(1).dropna())
    results_2var = gridsearch.search(p_range=[0, 1, 2, 3, 4], d_range=[0,1], q_range=[0, 1, 2])
    results_2var.to_csv("./results/WA_ARIMAX_staff_state_grid_diff1.csv")


"""
Visualizing and summarizing results
"""
tables = visualize_results.plot_gridsearch_results()

import seaborn as sns
sns.set_palette("icefire")

# Washington, ARIMAX(0,1,0) (Staff)
wa_model_fitter = models.ModelFitter(wa_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=wa_df[["Staff.Confirmed.Normed"]].diff(1).dropna())
wa_model_fitter.split_train_test()
# wa_model_fitter.fit_model(0,2,0)
wa_model_fitter.fit_model(0,1,0)
staff1 = wa_model_fitter.get_results(show=True)
results_table_staff1 = ["arimax(0,1,0)", staff1["coef_table"].loc['Staff.Confirmed.Normed']['Coefficients'],
                         "-", "-", "-", 1, "-", "-", staff1["coef_table"].loc['sigma2']['Coefficients'],
                        staff1["predictions"]["accuracy"]["mape"], staff1["predictions"]["aic"]]

# Washington, ARIMAX(0,1,2) (Staff)
wa_model_fitter = models.ModelFitter(wa_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=wa_df[["Staff.Confirmed.Normed"]].diff(1).dropna())
wa_model_fitter.split_train_test()
wa_model_fitter.fit_model(0,1,2)
staff2 = wa_model_fitter.get_results(show=True)
results_table_staff2 = ["arimax(0,1,2)", staff2["coef_table"].loc['Staff.Confirmed.Normed']['Coefficients'],
                         "-", "-", "-", 1, staff2["coef_table"].loc['ma.L1']['Coefficients'],
                         staff2["coef_table"].loc['ma.L2']['Coefficients'],
                         staff2["coef_table"].loc['sigma2']['Coefficients'],
                        staff2["predictions"]["accuracy"]["mape"], staff2["predictions"]["aic"]]


# Washington, ARIMAX(0,1,0) (State) # Should also do 3,1,0
wa_model_fitter = models.ModelFitter(wa_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=wa_df[["State.Cases.Normed"]].diff(1).dropna())
wa_model_fitter.split_train_test()
# wa_model_fitter.fit_model(3,1,0)
wa_model_fitter.fit_model(0,1,0)
state1 = wa_model_fitter.get_results(show=True)
results_table_state1 = ["arimax(0,1,0)", "-", state1["coef_table"].loc['State.Cases.Normed']['Coefficients'],
                         "-", "-", 1, "-",
                         "-",
                         state1["coef_table"].loc['sigma2']['Coefficients'],
                        state1["predictions"]["accuracy"]["mape"], state1["predictions"]["aic"]]

# Washington, ARIMA(0,1,0) # lowest MAPE
wa_model_fitter = models.ModelFitter(wa_df["Residents.Confirmed.Normed"].diff(1).dropna())
wa_model_fitter.split_train_test()
wa_model_fitter.fit_model(0,1,0)
self1 = wa_model_fitter.get_results(show=True)
results_table_self1 = ["arima(0,1,0)", "-",
                         "-", "-", "-", 1, "-",
                         "-",
                         self1["coef_table"].loc['sigma2']['Coefficients'],
                       self1["predictions"]["accuracy"]["mape"], self1["predictions"]["aic"]]

summary_table_WA = pd.DataFrame([results_table_self1, results_table_state1, results_table_staff1, results_table_staff2],
                                columns=["model", "Staff", "State", "ar.L1", "ar.L2", "d", "ma.L1", "ma.L2", "sigma2",
                                         "MAPE", "AIC"])

plt.show()

visualize_results.plot_multiple_predictions([staff2, state1, self1],
                                            wa_model_fitter.y_test.index, wa_model_fitter.y_test,
                                            labels=["ARIMAX(0,1,2)-Staff", "ARIMAX(0,1,0)-State", "ARIMA(0,1,0)"])


# California
#DIFFERENCED:
# CA: Univariate: (0,1,2), Staff: (0,1,1) and (1,1,0), State: (0,1,1), Both: non-significant

sns.set_palette("CMRmap")
# ARIMAX(0,1,1) (Staff)
ca_model_fitter = models.ModelFitter(ca_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=ca_df[["Staff.Confirmed.Normed"]].diff(1).dropna())
ca_model_fitter.split_train_test()
ca_model_fitter.fit_model(0,1,1)
staff1 = ca_model_fitter.get_results(show=True)
results_table_staff1 = ["arimax(0,1,1)", staff1["coef_table"].loc['Staff.Confirmed.Normed']['Coefficients'],
                         "-", "-", "-", 1, staff1["coef_table"].loc['ma.L1']['Coefficients'],
                         "-",
                         staff1["coef_table"].loc['sigma2']['Coefficients'],
                        staff1["predictions"]["accuracy"]["mape"], staff1["predictions"]["aic"]]

## BEST ONE
# ARIMAX(1,1,0) (Staff),
ca_model_fitter = models.ModelFitter(ca_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=ca_df[["Staff.Confirmed.Normed"]].diff(1).dropna())
ca_model_fitter.split_train_test()
ca_model_fitter.fit_model(1,1,0)
staff2 = ca_model_fitter.get_results(show=True)
results_table_staff2 = ["arimax(1,1,0)", staff2["coef_table"].loc['Staff.Confirmed.Normed']['Coefficients'],
                         "-", staff2["coef_table"].loc['ar.L1']['Coefficients'], "-", 1, "-",
                         "-",
                         staff2["coef_table"].loc['sigma2']['Coefficients'],
                        staff2["predictions"]["accuracy"]["mape"], staff2["predictions"]["aic"]]


# ARIMAX(0,1,1) (State)
ca_model_fitter = models.ModelFitter(ca_df["Residents.Confirmed.Normed"].diff(1).dropna(), x_data=ca_df[["State.Cases.Normed"]].diff(1).dropna())
ca_model_fitter.split_train_test()
ca_model_fitter.fit_model(0,1,1)
state1 = ca_model_fitter.get_results(show=True)
results_table_state1 = ["arimax(0,1,1)", "-",
                         state1["coef_table"].loc['State.Cases.Normed']['Coefficients'],
                            "-", "-", 1, state1["coef_table"].loc['ma.L1']['Coefficients'],
                         "-",
                         state1["coef_table"].loc['sigma2']['Coefficients'],
                        state1["predictions"]["accuracy"]["mape"], state1["predictions"]["aic"]]

# ARIMA(2,1,0)
ca_model_fitter = models.ModelFitter(ca_df["Residents.Confirmed.Normed"].diff(1).dropna())
ca_model_fitter.split_train_test()
ca_model_fitter.fit_model(2,1,0)
self1 = ca_model_fitter.get_results(show=True)
results_table_self1 = ["arima(2,1,0)", "-", "-",
                            self1["coef_table"].loc['ar.L1']['Coefficients'],
                            self1["coef_table"].loc['ar.L2']['Coefficients'], 1,
                            "-", "-", self1["coef_table"].loc['sigma2']['Coefficients'],
                       self1["predictions"]["accuracy"]["mape"], self1["predictions"]["aic"]]

summary_table_CA = pd.DataFrame([results_table_self1, results_table_state1, results_table_staff1, results_table_staff2],
                                columns=["model", "Staff", "State", "ar.L1", "ar.L2", "d", "ma.L1", "ma.L2", "sigma2",
                                         "MAPE", "AIC"])
plt.show()

visualize_results.plot_multiple_predictions([staff1, staff2, state1, self1],
                                            ca_model_fitter.y_test.index, ca_model_fitter.y_test,
                                            labels=["ARIMAX(0,1,1)-Staff", "ARIMAX(1,1,0)-Staff", "ARIMAX(0,1,1)-State", "ARIMA(2,1,0)"])

tex_CA_models = summary_table_CA.to_latex(escape=False)
tex_WA_models = summary_table_WA.to_latex(escape=False)
print(tex_CA_models)
print(tex_WA_models)


######

"""
CCM Analysis
"""

res_smoothed, staff_smoothed, state_smoothed = get_data()

convergent_cross_map(state_smoothed, staff_smoothed, "state", "staff")
convergent_cross_map(staff_smoothed, state_smoothed, "staff", "state")

convergent_cross_map(staff_smoothed, res_smoothed, "staff", "residents")
convergent_cross_map(state_smoothed, res_smoothed, "state", "residents")

convergent_cross_map(res_smoothed, staff_smoothed, "residents", "staff")
convergent_cross_map(res_smoothed, state_smoothed, "residents", "state")




