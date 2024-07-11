import explainer  # custom library
import utils  # custom library

import pandas as pd
import pickle

import tensorflow as tf

FILE_TRAIN = "dataset/insdn/train.csv"
FILE_TEST = "dataset/insdn/test.csv"

def calculateFrequency(shap_values):
  df_shap = pd.DataFrame(shap_values[:,:,0], columns=x_train.columns)
  data_dict = {}
  for idx in range(len(df_shap)):
    # Select the desired row and get the top 10 values's column names
    row_values = df_shap.iloc[idx].nlargest(10).index.tolist()
    # Create a dictionary and add the list with filename as the key
    data_dict['Normal sample ' + str(idx)] = row_values
  return data_dict

if __name__ == '__main__':
  (x_train, y_train) = utils.getDataFraction(FILE_TRAIN, 0.5)
  mlp = tf.keras.models.load_model(f'dataset/insdn/phase1/mlp', compile=False)
  utils.compile(mlp)
  cnn = tf.keras.models.load_model(f'dataset/insdn/model/cnn_4layer', compile=False)
  utils.compile(cnn)

  for counter, model in enumerate([mlp, cnn]):
    model_name = "mlp" if counter == 0 else "cnn"
    print(f"\n### CALCULATE SHAP_VALUES FOR {model_name} ###")
    shap_values = explainer.calculateSHAP(model, x_train)
    with open(f"dataset/insdn/defense/{model_name}_shap", "wb") as file:
      pickle.dump(shap_values, file)

    data_dict = calculateFrequency(shap_values)
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10'])
    freq = {}
    for column in df.columns:
      for feature in df[column]:
        if feature not in freq:
          count = sum(df[column_2] == feature for column_2 in df.columns)
          freq[feature] = count.sum()
          print(freq[feature])
    freq_df = pd.DataFrame(list(freq.items()), columns=['Feature', 'Frequency'])
    freq_df.sort_values(by=['Frequency'], ascending=False, inplace=True)
    # nho doi link, day la whitelist minh can
    freq_df.to_csv('dataset/insdn/defense/cnn_top_feature.csv', index=False)