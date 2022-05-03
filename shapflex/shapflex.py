"""Main module."""
#02.05.22
import numpy as np
import pandas as pd
import igraph
import itertools
from catboost import CatBoostClassifier

class shapFlex_plus:
    def __init__(self, explain,  model, predict_function, reference = None, target_features = None, \
                     causal = None, causal_weights = None, sample_size = None, use_future = None):
        self.explain = explain
        self.reference = reference if reference else explain
        self.model = model
        self.predict_function = predict_function
        self.target_features = target_features if isinstance(target_features, pd.core.series.Series) else explain.columns.tolist()
        self.causal = causal #if causal else None
        self.causal_weights = causal_weights #if causal_weights else None
        self.sample_size = sample_size if sample_size else 60
        self.use_future = use_future if isinstance(target_features, pd.core.series.Series) else False
        
        self.n_features = self.explain.shape[1]
        self.n_instances = self.reference.shape[0]

        self.causal_graph = igraph.Graph.DataFrame(self.causal, directed=True) if isinstance(self.causal, pd.core.frame.DataFrame) else [None]
        self.nodes = [v for v in self.causal_graph.vs] if isinstance(self.causal, pd.core.frame.DataFrame) else [None]
        self.each_node_causes = {v['name']: [succ['name'] for succ in v.successors()] for v in self.nodes if v.successors()} if isinstance(self.causal, pd.core.frame.DataFrame) else [None]# надо уточнить, мб здесь не только "прямые" successors и predecessors ищутся 
        self.each_node_is_an_effect_from = {v['name']: [pred['name'] for pred in v.predecessors()] for v in self.nodes if v.predecessors()} if isinstance(self.causal, pd.core.frame.DataFrame) else [None]# но и вообще все
        self.causal_nodes = [v for v in self.each_node_causes.keys()] if isinstance(self.causal, pd.core.frame.DataFrame) else [None]
        self.effect_nodes = [v for v in self.each_node_is_an_effect_from.keys()] if isinstance(self.causal, pd.core.frame.DataFrame) else [None]
        self.nodes = [v['name'] for v in self.nodes] if isinstance(self.causal, pd.core.frame.DataFrame) else [None]

    @staticmethod
    def unlist_df(data):
      unlisted_df = pd.Series(
                  data,
                  index=[
                  index_col + index_row for index_col, index_row in itertools.product(
                      [str(x) for x in range(data.shape[0])], 
                      [str(x) for x in data.columns])]
              )
      return unlisted_df
      
    def loop_over_monte_carlo_samples(self):
      i_size = self.sample_size
      j_size = len(self.target_features)
      data_sample = []

      for i in range(i_size):
        reference_index = np.random.choice(np.arange(0, self.n_features ), size=1, replace=False)
        feature_indices_random = np.random.choice(np.arange(0, self.n_features), size=self.n_features, replace=False)
        feature_names_random = self.explain.columns[feature_indices_random].values
        reference_instance = self.reference.iloc[reference_index, feature_indices_random]
        explain_instances = self.explain.iloc[:, feature_indices_random]
        data_sample_feature = []
        for j in range(j_size):
          target_feature_index =  self.explain.columns.get_loc(self.target_features[j])
          target_feature_index_shuffled = list(self.explain.columns.values[feature_indices_random]).index(self.target_features[j])
         
          if self.target_features[j] in self.nodes:
            target_feature_causes_these_features =  [self.target_features[j]] + self.each_node_causes.get(self.target_features[j], []) 
            target_feature_is_caused_by =  [self.target_features[j]] + self.each_node_is_an_effect_from.get(self.target_features[j], []) 
            target_index = target_feature_index_shuffled
            causes_indices = np.where(np.in1d(feature_names_random, target_feature_is_caused_by[1:]))[0]
            effects_indices  = np.where(np.in1d(feature_names_random, target_feature_causes_these_features[1:]))[0]
            sample_indices = feature_indices_random[~np.isin(feature_indices_random, 
                np.concatenate([[target_index], causes_indices, effects_indices]))]
            sample_real_indices = sample_indices[sample_indices < target_index]  # Not in causal diagram, feature data from 'explain'.
            sample_fake_indices = sample_indices[sample_indices > target_index]  # Not in causal diagram, feature data from 'reference'.

            feature_indices_real_causes_real_effects = np.concatenate([sample_real_indices, causes_indices, effects_indices, [target_index], sample_fake_indices])
            feature_indices_real_causes_fake_effects = np.concatenate([sample_real_indices, causes_indices, [target_index], effects_indices, sample_fake_indices])
            feature_indices_fake_causes_real_effects = np.concatenate([sample_real_indices, effects_indices, [target_index], causes_indices, sample_fake_indices])
            feature_indices_fake_causes_fake_effects = np.concatenate([sample_real_indices, [target_index], causes_indices, effects_indices, sample_fake_indices])
          
          if not self.target_features[j] in self.nodes:
            explain_instance_real_target = explain_instances.copy()

            # Only create a Frankenstein instance if the target is not the last feature and there is actually
            # one or more features to the right of the target to replace with the reference.
            if (target_feature_index_shuffled < self.n_features):
              explain_instance_real_target.iloc[:, target_feature_index_shuffled+1: ] =\
                 pd.concat([reference_instance.iloc[:, target_feature_index_shuffled+1: ]] * self.explain.shape[0], axis=0).reset_index(drop=True)
              
            # These instances are otherwise the same as the Frankenstein instance created above with the
            # exception that the target feature is now replaced with the target feature in the random reference
            # instance. The difference in model predictions between these two Frankenstein instances is
            # what gives us the stochastic Shapley value approximation.
            explain_instance_fake_target = explain_instance_real_target.copy()
            explain_instance_fake_target.iloc[:, [target_feature_index_shuffled]] =\
               pd.concat([reference_instance.iloc[:, [target_feature_index_shuffled]]]  * self.explain.shape[0], axis=0).reset_index(drop=True)
          
          else:

            if self.target_features[j] in self.causal_nodes:
              reference_instance_real_causes_fake_effects = reference_instance.iloc[:, feature_indices_real_causes_fake_effects]
              explain_instance_real_causes_fake_effects_real_target = explain_instances.iloc[:, feature_indices_real_causes_fake_effects]
              target_index_temp = explain_instance_real_causes_fake_effects_real_target.columns.get_loc(self.target_features[j])
              if target_index_temp < self.n_features:
                explain_instance_real_causes_fake_effects_real_target.iloc[:, target_index_temp + 1: self.n_features + 1] =\
                pd.concat([reference_instance_real_causes_fake_effects.iloc[:, target_index_temp + 1: self.n_features + 1]]  * self.explain.shape[0], axis=0).reset_index(drop=True)
                

              explain_instance_real_causes_fake_effects_fake_target = explain_instance_real_causes_fake_effects_real_target.copy()
              explain_instance_real_causes_fake_effects_fake_target.iloc[:, target_index_temp] =\
              pd.concat([reference_instance_real_causes_fake_effects.iloc[:, target_index_temp]]  * self.explain.shape[0], axis=0).reset_index(drop=True)

              reference_instance_fake_causes_real_effects = reference_instance.iloc[:, feature_indices_fake_causes_real_effects]
              explain_instance_fake_causes_real_effects_real_target_cause = explain_instances.iloc[:, feature_indices_fake_causes_real_effects]
              target_index_temp = explain_instance_real_causes_fake_effects_real_target.columns.get_loc(self.target_features[j])

              if target_index_temp < self.n_features:
                explain_instance_fake_causes_real_effects_real_target_cause.iloc[:, target_index_temp + 1: self.n_features + 1] =\
                pd.concat([reference_instance_fake_causes_real_effects.iloc[:, target_index_temp + 1: self.n_features+1]]*self.explain.shape[0],
                axis=0).reset_index(drop=True)
              
              explain_instance_fake_causes_real_effects_fake_target_cause = explain_instance_fake_causes_real_effects_real_target_cause.copy()
              explain_instance_fake_causes_real_effects_fake_target_cause.iloc[:, target_index_temp] =\
              pd.concat([reference_instance_fake_causes_real_effects.iloc[:, target_index_temp]]*self.explain.shape[0],
                axis=0).reset_index(drop=True)

            if self.target_features[j] in self.effect_nodes:
              reference_instance_real_causes_fake_effects = reference_instance.iloc[:, feature_indices_real_causes_fake_effects]
              explain_instance_real_causes_fake_effects_real_target_effect = explain_instances.iloc[:, feature_indices_real_causes_fake_effects]

              target_index_temp = explain_instance_real_causes_fake_effects_real_target_effect.columns.get_loc(self.target_features[j])

              if (target_index_temp < self.n_features):
                explain_instance_real_causes_fake_effects_real_target_effect.iloc[:, target_index_temp + 1: self.n_features + 1] =\
                pd.concat([reference_instance_real_causes_fake_effects.iloc[:, target_index_temp + 1: self.n_features + 1]]*self.explain.shape[0],
                axis=0).reset_index(drop=True)
              
              explain_instance_real_causes_fake_effects_fake_target_effect = explain_instance_real_causes_fake_effects_real_target_effect.copy()
              explain_instance_real_causes_fake_effects_fake_target_effect.iloc[:, target_index_temp] =\
                pd.concat([reference_instance_real_causes_fake_effects.iloc[:, target_index_temp]]*self.explain.shape[0],
                axis=0).reset_index(drop=True)
              reference_instance_fake_causes_real_effects = reference_instance.iloc[:, feature_indices_fake_causes_real_effects]
              explain_instance_fake_causes_real_effects_real_target = explain_instances.iloc[:, feature_indices_fake_causes_real_effects]
              target_index_temp = explain_instance_fake_causes_real_effects_real_target.columns.get_loc(self.target_features[j])

              if target_index_temp < self.n_features:
                explain_instance_fake_causes_real_effects_real_target.iloc[:, target_index_temp + 1: self.n_features + 1] =\
                pd.concat([reference_instance_fake_causes_real_effects.iloc[:, target_index_temp + 1: self.n_features + 1]]*self.explain.shape[0],
                axis=0).reset_index(drop=True)
                

              explain_instance_fake_causes_real_effects_fake_target = explain_instance_fake_causes_real_effects_real_target.copy()
              explain_instance_fake_causes_real_effects_fake_target.iloc[:, target_index_temp] =\
              pd.concat([reference_instance_fake_causes_real_effects.iloc[:, target_index_temp]]*self.explain.shape[0],
                axis=0).reset_index(drop=True)

          if not self.target_features[j] in self.nodes:
            explain_instance_real_target = explain_instance_real_target.loc[:, self.explain.columns]
            explain_instance_fake_target = explain_instance_fake_target.loc[:, self.explain.columns]
            data_explain_instance = pd.concat([explain_instance_real_target, explain_instance_fake_target], axis=0).reset_index(drop=True)#, ignore_index=True)
            data_explain_instance['index_in_sample'] = np.tile(np.arange(0, self.explain.shape[0]), 2) 
            data_explain_instance['feature_group'] = np.repeat(['real_target', 'fake_target'], repeats=self.explain.shape[0])
            data_explain_instance['feature_name'] = self.target_features[j]
            data_explain_instance['causal'] = 0
            data_explain_instance['causal_type'] = None

          else:
            if self.target_features[j] in self.causal_nodes:
              explain_instance_real_causes_fake_effects_real_target =\
              explain_instance_real_causes_fake_effects_real_target.loc[:, self.explain.columns]
              explain_instance_real_causes_fake_effects_fake_target =\
              explain_instance_real_causes_fake_effects_fake_target.loc[:, self.explain.columns]
              explain_instance_fake_causes_real_effects_real_target_cause =\
              explain_instance_fake_causes_real_effects_real_target_cause.loc[:, self.explain.columns]
              explain_instance_fake_causes_real_effects_fake_target_cause =\
              explain_instance_fake_causes_real_effects_fake_target_cause.loc[:, self.explain.columns]
            if self.target_features[j] in self.effect_nodes:
              explain_instance_real_causes_fake_effects_real_target_effect =\
              explain_instance_real_causes_fake_effects_real_target_effect.loc[:, self.explain.columns]
              explain_instance_real_causes_fake_effects_fake_target_effect =\
              explain_instance_real_causes_fake_effects_fake_target_effect.loc[:, self.explain.columns]
              explain_instance_fake_causes_real_effects_real_target =\
              explain_instance_fake_causes_real_effects_real_target.loc[:, self.explain.columns]
              explain_instance_fake_causes_real_effects_fake_target =\
              explain_instance_fake_causes_real_effects_fake_target.loc[:, self.explain.columns]
            
            if self.target_features[j] in self.causal_nodes:
              data_explain_instance = pd.concat([
                explain_instance_real_causes_fake_effects_real_target,
                explain_instance_real_causes_fake_effects_fake_target,
                explain_instance_fake_causes_real_effects_real_target_cause,
                explain_instance_fake_causes_real_effects_fake_target_cause], axis=0
              ).reset_index(drop=True)
              data_explain_instance['index_in_sample'] = np.tile(np.arange(1, self.explain.shape[0] + 1), 4)  # Four Frankenstein instances per explained instance.
              data_explain_instance['feature_group'] = np.repeat(["real_causes_fake_effects_real_target", "real_causes_fake_effects_fake_target",
                                                          "fake_causes_real_effects_real_target_cause", "fake_causes_real_effects_fake_target_cause"],
                                                        self.explain.shape[0])
              data_explain_instance['causal_type'] = "target_is_a_cause"
            if self.target_features[j] in self.effect_nodes:
              data_explain_instance = pd.concat([
                explain_instance_real_causes_fake_effects_real_target_effect,
                explain_instance_real_causes_fake_effects_fake_target_effect,
                explain_instance_fake_causes_real_effects_real_target,
                explain_instance_fake_causes_real_effects_fake_target
              ], axis=0).reset_index(drop=True)
              data_explain_instance['index_in_sample'] = np.tile(np.arange(1, self.explain.shape[0] + 1), 4)  # Four Frankenstein instances per explained instance.
              data_explain_instance['feature_group'] = np.repeat(["real_causes_fake_effects_real_target_effect", "real_causes_fake_effects_fake_target_effect",
                                                          "fake_causes_real_effects_real_target", "fake_causes_real_effects_fake_target"],
                                                        self.explain.shape[0])
              data_explain_instance['causal_type'] = "target_is_an_effect"

            if (self.target_features[j] in self.causal_nodes) and (self.target_features[j] in self.effect_nodes):
              data_explain_instance = pd.concat([
                explain_instance_real_causes_fake_effects_real_target,
                explain_instance_real_causes_fake_effects_fake_target,
                explain_instance_fake_causes_real_effects_real_target_cause,
                explain_instance_fake_causes_real_effects_fake_target_cause,
                explain_instance_real_causes_fake_effects_real_target_effect,
                explain_instance_real_causes_fake_effects_fake_target_effect,
                explain_instance_fake_causes_real_effects_real_target,
                explain_instance_fake_causes_real_effects_fake_target
              ], axis=0).reset_index(drop=True)
              data_explain_instance['index_in_sample'] = np.tile(np.arange(1, self.explain.shape[0] + 1), 8)  # Eight Frankenstein instances per explained instance.
              data_explain_instance['feature_group'] = np.repeat([
                "real_causes_fake_effects_real_target", "real_causes_fake_effects_fake_target",  # Target is a causal node.
                "fake_causes_real_effects_real_target_cause", "fake_causes_real_effects_fake_target_cause",  # Target is a causal node.
                "real_causes_fake_effects_real_target_effect", "real_causes_fake_effects_fake_target_effect",  # Target is an effect node.
                "fake_causes_real_effects_real_target", "fake_causes_real_effects_fake_target"  # Target is an effect node.
                ],
              self.explain.shape[0])
              data_explain_instance['causal_type'] = np.repeat([
                "target_is_a_cause", "target_is_a_cause", "target_is_a_cause", "target_is_a_cause",
                "target_is_an_effect", "target_is_an_effect", "target_is_an_effect", "target_is_an_effect"],
              self.explain.shape[0])
            
            data_explain_instance['feature_name'] = self.target_features[j]
            data_explain_instance['causal'] = 1

          data_explain_instance['sample'] = i
          data_sample_feature.append(data_explain_instance)

        data_sample.append(data_sample_feature)

      data_sample = pd.concat([pd.concat(data_sample_i, axis=0) for data_sample_i in data_sample], axis=0).reset_index(drop=True)
      return data_sample

    def predict_shapFlex(self, data_predict):
      '''есть self.reference, self.model, self.predict_function, self.n_features, self.causal, self.causal_weights'''
      data_model = data_predict.iloc[:, :self.n_features].copy()
      data_meta = data_predict.iloc[:, self.n_features:].copy()
      data_predicted = pd.DataFrame(predict_function(self.model, data_model), index=data_model.index)
      data_predicted = pd.concat([data_meta, data_predicted], axis=1)
      intercept = predict_function(self.model, self.reference).mean(skipna=True)
      user_fun_y_pred_name = data_predicted.columns[-1]
      variables_of_interest = list(set(data_predicted.columns) - set(['feature_group', user_fun_y_pred_name]))
      data_predicted.loc[:, variables_of_interest] =\
        data_predicted.loc[:, variables_of_interest].fillna(0)
      
      
      data_predicted = data_predicted.pivot_table(
        index=set(data_predicted.columns) - set(['feature_group', user_fun_y_pred_name]),
        columns=['feature_group'],
        values=user_fun_y_pred_name
      ).reset_index()
      
      data_non_causal = data_predicted.loc[data_predicted['causal']==0]
      data_non_causal['shap_effect'] = data_non_causal['real_target'] - data_non_causal['fake_target']
      data_causal = data_predicted.loc[data_predicted['causal']==1]

      if isinstance(self.causal, pd.core.frame.DataFrame):
        data_target_is_a_cause = data_causal[data_causal['causal_type'] == 'target_is_a_cause']
        data_target_is_an_effect = data_causal[data_causal['causal_type'] == 'target_is_an_effect']

        data_target_is_a_cause['shap_u_1_12'] = data_target_is_a_cause.loc[:, 'real_causes_fake_effects_real_target'] -\
          data_target_is_a_cause.loc[:, 'real_causes_fake_effects_fake_target']
        data_target_is_a_cause['shap_u_1_21'] = data_target_is_a_cause.loc[:, 'fake_causes_real_effects_real_target_cause'] -\
          data_target_is_a_cause.loc[:, 'fake_causes_real_effects_fake_target_cause']
        data_target_is_an_effect['shap_u_2_12'] = data_target_is_an_effect.loc[:, 'real_causes_fake_effects_real_target_effect'] -\
          data_target_is_an_effect.loc[:, 'real_causes_fake_effects_fake_target_effect']
        data_target_is_an_effect['shap_u_2_21'] = data_target_is_an_effect.loc[:, 'fake_causes_real_effects_real_target'] -\
          data_target_is_an_effect.loc[:, 'fake_causes_real_effects_fake_target']
        
        data_weights = pd.concat([self.causal, pd.Series(self.causal_weights)], axis=1)
        data_weights.columns = ["target_is_a_cause", "target_is_an_effect", "weight"]
        data_weights = pd.melt(data_weights,  id_vars='weight')
        data_weights.columns= ['weight', "causal_type", "feature_name"]
        data_weights = data_weights.groupby(['causal_type', 'feature_name']).apply(np.mean).reset_index()
        data_target_is_a_cause = data_target_is_a_cause.merge(data_weights, on=['causal_type', 'feature_name'], how='left')
        data_target_is_an_effect = data_target_is_an_effect.merge(data_weights, on=['causal_type', 'feature_name'], how='left')
        
        shap_u_1 = np.sum(data_target_is_a_cause[['shap_u_1_12', 'shap_u_1_21']].values *\
           np.hstack([data_target_is_a_cause[['weight']].values, 1 - data_target_is_a_cause[['weight']].values]), axis=-1)
        data_target_is_a_cause['shap_effect'] = shap_u_1
        if data_target_is_an_effect.shape[0] > 0:
          shap_u_2 = np.sum(data_target_is_an_effect[['shap_u_2_12', 'shap_u_2_21']].values *\
           np.hstack([data_target_is_an_effect[['weight']].values, 1 - data_target_is_an_effect[['weight']].values]), axis=-1)
          data_target_is_an_effect['shap_effect'] = shap_u_2

        data_causal = pd.concat([data_target_is_a_cause, data_target_is_an_effect], axis=0)
        data_causal = data_causal.groupby(['index_in_sample', 'sample', 'feature_name']).mean().reset_index()# мб докинуть условие на skipna

      data_predicted = pd.concat([data_causal, data_non_causal], ignore_index=True, axis=0)
      data_predicted = data_predicted.loc[:, ['index_in_sample', 'sample', 'feature_name', 'shap_effect']]

      data_predicted = data_predicted.reset_index().dropna(axis=0).groupby(['index_in_sample', 'feature_name']).agg({'shap_effect': [np.std, np.mean]})
      data_predicted[('shap_effect', 'intercept')] = intercept[0]
      data_predicted = data_predicted.reset_index()
      data_predicted.columns = ['index_in_sample', 'feature_name', 'shap_effect sd', 'shap_effect', 'shap_effect intercept']

      return data_predicted

    def forward(self):
      data_predict = self.loop_over_monte_carlo_samples()
      data_predicted = self.predict_shapFlex(data_predict)
      data_merge = pd.melt(self.explain)
      data_merge.columns = ["feature_name", "feature_value"]
      data_merge['index_in_sample'] = np.tile(np.arange(self.explain.shape[0]), self.n_features)
      data_out = data_merge.merge(data_predicted, how='right', on=['index_in_sample', 'feature_name'])

      return data_out