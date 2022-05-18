from shapflex import shapFlex_plus
from catboost import CatBoostClassifier 

import pandas as pd
import numpy as np
num_users = 10000
num_months = 12

signup_months = np.random.choice(np.arange(1, num_months), num_users) * np.random.randint(0,2, size=num_users) # signup_months == 0 means customer did not sign up
df = pd.DataFrame({
    'user_id': np.repeat(np.arange(num_users), num_months),
    'signup_month': np.repeat(signup_months, num_months), # signup month == 0 means customer did not sign up
    'month': np.tile(np.arange(1, num_months+1), num_users), # months are from 1 to 12
    'spend': np.random.poisson(500, num_users*num_months) #np.random.beta(a=2, b=5, size=num_users * num_months)*1000 # centered at 500
})
# A customer is in the treatment group if and only if they signed up
df["treatment"] = df["signup_month"]>0
# Simulating an effect of month (monotonically decreasing--customers buy less later in the year)
df["spend"] = df["spend"] - df["month"]*10
# Simulating a simple treatment effect of 100
after_signup = (df["signup_month"] < df["month"]) & (df["treatment"])
df.loc[after_signup,"spend"] = df[after_signup]["spend"] + 100
i = 3 
causal_graph = """digraph {
treatment[label="Program Signup in month i"];
pre_spends;
post_spends;
Z->treatment;
pre_spends -> treatment;
treatment->post_spends;
signup_month->post_spends;
signup_month->treatment;
}"""

# Post-process the data based on the graph and the month of the treatment (signup)
# For each customer, determine their average monthly spend before and after month i
df_i_signupmonth = (
    df[df.signup_month.isin([0, i])]
    .groupby(["user_id", "signup_month", "treatment"])
    .apply(
        lambda x: pd.Series(
            {
                "pre_spends": x.loc[x.month < i, "spend"].mean(),
                "post_spends": x.loc[x.month > i, "spend"].mean(),
            }
        )
    )
    .reset_index()
)
print(df_i_signupmonth)

outcome_name = 'post_spends'
data = df_i_signupmonth
outcome_col = pd.Series(data.columns)[data.columns==outcome_name].index[0]
X, y = data.drop(outcome_name, axis=1), data[outcome_name].values
cat_features = [inx for inx, value in zip(X.dtypes.index, X.dtypes) if value =='object']
model = CatBoostClassifier(iterations=5)
model.fit(X, y, cat_features=cat_features, verbose=False)
def predict_function(model, data):
  #pd.DataFrame(model.predict_proba(X)).loc[:, 0][9] если запустить будет результат 0.98, что соответствует
  #выводу для 9 номера который равен 0.98, неважно какой алгоритм, такая высокая степень уверенности
  #позволяет идентифицировать выводимую колонку однозначно
  return pd.DataFrame(model.predict_proba(data)[:, [0]])



explain, reference = data.iloc[:10, :data.shape[1]-1], data.iloc[:, :data.shape[1]-1]
sample_size = 5
target_features = pd.Series(["treatment",'pre_spends',  'signup_month'])
causal = pd.DataFrame(
    {'cause': ['pre_spends',  'signup_month',  ], 'effect': [ 
    'treatment', 'treatment']}
)

exmpl_of_test = shapFlex_plus(explain,  model, predict_function, target_features=target_features, causal=causal, causal_weights = [1. for x in range(len(causal))])
yes_cause = exmpl_of_test.forward()
