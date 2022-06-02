# shapflex


[![image](https://img.shields.io/pypi/v/shapflex.svg)](https://pypi.python.org/pypi/shapflex)

[![image](https://img.shields.io/conda/vn/conda-forge/shapflex.svg)](https://anaconda.org/conda-forge/shapflex)


**A python version of R package for computing asymmetric Shapley values to assess causality in any trained machine learning model**


-   Free software: MIT license
-   Documentation: https://gregory-ch.github.io/shapflex
 
## Warnings 

This is the alpha version of porting https://github.com/nredell/shapFlex

## Example with discovery on real data
 
 https://github.com/gregory-ch/shap_flex_porting/blob/main/shap_joint.ipynb
 
## reproducing original example



```
#02.05.22


import pandas as pd
import numpy as np
from shapflex.shapflex import shapFlex_plus
from catboost import CatBoostClassifier 

data = pd.read_csv('https://kolodezev.ru/download/data_adult.csv', index_col=0)
outcome_name = 'income'
outcome_col = pd.Series(data.columns)[data.columns==outcome_name].index[0]
X, y = data.drop(outcome_name, axis=1), data[outcome_name].values
cat_features = [inx for inx, value in zip(X.dtypes.index, X.dtypes) if value =='object']
model = CatBoostClassifier(iterations=100)
model.fit(X, y, cat_features=cat_features, verbose=False)
def predict_function(model, data):
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
result = exmpl_of_test.forward()
print(result.groupby('feature_name').mean())


```


## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
