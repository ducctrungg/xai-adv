from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import json
import pickle
import tensorflow as tf

def evaluate(y_test, y_pred):
  # Evaluate
  cm=confusion_matrix(y_test, y_pred)
  cr=classification_report(y_test, y_pred)

  print("Confusion Matrix:")
  print(cm)

  print("Performance Matrix:")
  print(cr)

  print("Individual metrics:")
  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("Precision:", precision_score(y_test, y_pred))
  print("Recall:", recall_score(y_test, y_pred))
  print("F1-Score:", f1_score(y_test, y_pred))

def getData(file):
  df = pd.read_csv(file)
  return (df.iloc[:,:-1], df.iloc[:,-1])

def getDataFraction(file, fraction):
  """
  Get feature and classes based on fraction
  """
  df = pd.read_csv(file)
  df = df.sample(frac=fraction, replace=False, random_state=10, ignore_index=True)
  return (df.iloc[:,:-1], df.iloc[:,-1])

def manipulateFeature(df_x, df_y, feature_path):
  with open(feature_path, "r") as file:
    features = json.load(file) # Read from json file
    df_mask = pd.DataFrame([features])
  for idx in range(len(df_x)):
    if df_y[idx] == 1:
      for column in df_mask.columns:
        if df_mask[column].values[0] != 0:
          df_x.at[idx, column] = df_mask[column].values[0]
  return df_x

def saveModel(model, model_path):
  with open(model_path, "wb") as file:
    pickle.dump(model, file)

# Load model
def loadModel(model_path):
  with open(model_path, 'rb') as file:
    model = pickle.load(file)
  return model

def getClasses(classes, label, dataframe):
  """
  Get only samples from specific class in given dataframe
  """
  return dataframe[dataframe[classes] == label]

def mergeXY(df_x, df_y):  
  return pd.concat([df_x, df_y], axis=1)

def mergeDF(df_a, df_b):
  return pd.concat([df_a, df_b], axis=0)

def compile(model):
  model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['binary_accuracy']
  )