# shapflex


[![image](https://img.shields.io/pypi/v/shapflex.svg)](https://pypi.python.org/pypi/shapflex)

[![image](https://img.shields.io/conda/vn/conda-forge/shapflex.svg)](https://anaconda.org/conda-forge/shapflex)


**A python version of R package for computing asymmetric Shapley values to assess causality in any trained machine learning model**


-   Free software: MIT license
-   Documentation: https://gregory-ch.github.io/shapflex
 
## Warnings 

This is the alpfa version of porting https://github.com/nredell/shapFlex
 
## Examples

***
#02.05.22


import pandas as pd
import numpy as np
data = pd.read_csv('https://kolodezev.ru/download/data_adult.csv', index_col=0)
outcome_name = 'income'
outcome_col = pd.Series(data.columns)[data.columns==outcome_name].index[0]
X, y = data.drop(outcome_name, axis=1), data[outcome_name].values
cat_features = [inx for inx, value in zip(X.dtypes.index, X.dtypes) if value =='object']
model = CatBoostClassifier(iterations=100)
model.fit(X, y, cat_features=cat_features, verbose=False)
def predict_function(model, data):
  #pd.DataFrame(model.predict_proba(X)).loc[:, 0][9] если запустить будет результат 0.98, что соответствует
  #выводу для 9 номера который равен 0.98, неважно какой алгоритм, такая высокая степень уверенности
  #позволяет идентифицировать выводимую колонку однозначно
  return pd.DataFrame(model.predict_proba(data)[:, [0]])


explain, reference = data.iloc[:300, :data.shape[1]-1], data.iloc[:, :data.shape[1]-1]
sample_size = 50
target_features = pd.Series(["marital_status", "education", "relationship",  "native_country",
                     "age", "sex", "race", "hours_per_week"])
causal = pd.DataFrame(
  dict(cause=pd.Series(["age", "sex", "race", "native_country",
              "age", "sex", "race", "native_country", "age",
              "sex", "race", "native_country"]),
  effect = pd.Series(np.concatenate([np.tile("marital_status", 4), np.tile("education", 4), np.tile("relationship", 4)])))
)
exmpl_of_test = shapFlex_plus(explain,  model, predict_function, target_features=pd.Series(["marital_status", "education", "relationship", "native_country",
"age", "sex", "race", "hours_per_week"]), causal=causal, causal_weights = [1. for x in range(len(causal))])
data_predict = exmpl_of_test.loop_over_monte_carlo_samples()
'''data_predict = pd.read_csv('r_data_predict.csv', index_col=0)
data_predict = data_predict.rename(columns={'index':'index_in_sample'})'''
print(data_predict.shape)
data_predicted = exmpl_of_test.predict_shapFlex(data_predict)
print(data_predicted.shape)
data_merge = pd.melt(explain)
data_merge.columns = ["feature_name", "feature_value"]
data_merge['index_in_sample'] = np.tile(np.arange(explain.shape[0]), exmpl_of_test.n_features)
data_out = data_merge.merge(data_predicted, how='right', on=['index_in_sample', 'feature_name'])
#result = exmpl_of_test.forward()
#print(result.shape)

*** 


## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
