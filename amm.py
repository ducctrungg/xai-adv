import numpy
import json
import pandas as pd

FEATURE_ADV = [
  'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
  'Bwd IAT Mean', 'Bwd IAT Max', 'Bwd IAT Std', 'Bwd IAT Min', 
  'Active Mean', 'Active Std', 'Active Max', 'Active Min',
  'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
  'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
  'Init Fwd Win Byts', 'Init Bwd Win Byts'
]

class AMMGenerator:
  def AMMFeatureSelection(self, x_dataset_matrix, s_matrix_shap: numpy.ndarray, feature_names: list, trigger_size):
    x_local = x_dataset_matrix.to_numpy()
    shap_matrix_t = s_matrix_shap.T
    distances_feature = numpy.array([item.max() - item.min() for item in shap_matrix_t])

    feature_count = shap_matrix_t.shape[0]
    sample_count = x_local.shape[0]

    d_p = []
    for index in range(0, feature_count):
      distance = distances_feature[index]
      shap_vec = shap_matrix_t[index]
      mean = numpy.mean(shap_vec)
      p = numpy.sum(shap_vec > mean) / sample_count  # find vi > mean
      d_p.append(distance * p)

    order_feature = numpy.argsort(-numpy.array(d_p))
    output = {}
    for _index in range(0, trigger_size):
      order = order_feature[_index]
      feature = feature_names[order]
      shap_vec = shap_matrix_t[order]
      row = numpy.argmin(shap_vec)
      value = x_local[row][order]
      output[feature] = value
    return output

  def manipulateFeature(self, amm_file_patch, df_x, df_y):
    with open(amm_file_patch, "r") as file:
      features = json.load(file) # Read from json file
      df_mask = pd.DataFrame([features])
    df_temp = df_x.copy()
    for idx in range(len(df_y)):
      if df_y[idx] == 1:
        for column in df_mask.columns:
          if column in FEATURE_ADV: 
            df_temp.at[idx, column] = df_mask[column].values[0]
    return df_temp

  def saveAmmPatch(self, type, dataset, results):
    with open(f'dataset/{dataset}/phase1/{type}_amm.json', 'w') as file:
      json.dump(results, file)
    return file.name