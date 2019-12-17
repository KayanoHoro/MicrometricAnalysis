# The code below imports a few packages we will need for this analysis
import pandas as pd
import numpy as np
import pylab as pl
import statsmodels.api as sm

# load the data
df = pd.read_csv("african_crises.csv")
# check out the first 5 rows of the dataset
# print(df.head())
# summarize the data
# print(df.describe())

# let several parameters become dummy
dummy_domestic = pd.get_dummies(df["domestic_debt_in_default"], prefix="domestic_debt_in_default")
dummy_sovereign = pd.get_dummies(df["sovereign_external_debt_default"], prefix="sovereign_external_debt_default")
dummy_independence = pd.get_dummies(df["independence"], prefix="independence")
# print(dummy_domestic,dummy_sovereign,dummy_independence)

# create a clean data frame for the regression
cols_to_keep = ["systemic_crisis", "currency_crises", "inflation_crises", "banking_crisis", "exch_usd", "gdp_weighted_default", "inflation_annual_cpi"]
data = df[cols_to_keep].join(dummy_domestic.loc[:,: "domestic_debt_in_default_0"]).join(dummy_sovereign.loc[:,: "sovereign_external_debt_default_0"]).join(dummy_independence.loc[:,: "independence_0"])
# print(data.head())

# manually add the intercept
data["intercept"] = 1.0

# extract independent variables
train_cols = data.columns[4:]
# print(train_cols)

# fit the binary logit model
logit = sm.Logit(data["systemic_crisis"], data[train_cols])
result = logit.fit()

# display the results
print(result.summary())
# odds ratios only
# print(np.exp(result.params))
i = int(input("Please input the number you want to analyze："))
systemic_crisis_Xib = result.params[6] + result.params[0] * data["exch_usd"][i] + result.params[1] * data["gdp_weighted_default"][i] \
                      + result.params[4] * data["sovereign_external_debt_default_0"][i] + result.params[5] * data["independence_0"][i]
print(np.exp(systemic_crisis_Xib)/(1 + np.exp(systemic_crisis_Xib))**2)
# print(1/(1 + np.exp(-systemic_crisis_Xib)))

# the same display with different dependent variable
logit = sm.Logit(data["currency_crises"], data[train_cols])
result = logit.fit()
print(result.summary())
# print(np.exp(result.params))
logit = sm.Logit(data["inflation_crises"], data[train_cols])
result = logit.fit()
print(result.summary())
# print(np.exp(result.params))
logit = sm.Logit(data["banking_crisis"], data[train_cols])
result = logit.fit()
print(result.summary())
# print(np.exp(result.params))