"""
Kịch bản: 
  1. Huấn luyện các mô hình DeepLearning: MLP, CNN 2 lớp, CNN 4 lớp
  2. Thực nghiệm 2 kịch bản 1 và 2

Kịch bản 1: 
  1. Tính shap_values cho mô hình MLP
  2. Tính amm và tạo amm_patch để tạo adversarial samples
  3. Đánh giá lại mẫu adversarial này với Black-box model là CNN 2 lớp và CNN 4 lớp  
"""

import json
import pickle

import tensorflow as tf
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import explainer  # custom library
import utils  # custom library
from amm import AMMGenerator  # custom library

FILE_TRAIN = "dataset/cicids2018/train.csv"
FILE_TEST = "dataset/cicids2018/test.csv"
FRACTION = [0.1, 0.2]

def trainDT():
  dt = DecisionTreeClassifier()
  dt.fit(x_train, y_train)
  return dt

def trainXB():
  params = {
    'tree_method': 'approx',
    'objective': 'binary:logistic',
  }
  num_boost_round = 20
  xb = XGBClassifier(n_estimators=num_boost_round, **params)
  xb.fit(x_train, y_train)
  return xb

def trainRF():
  rf = RandomForestClassifier()
  rf.fit(x_train, y_train)
  return rf

def trainLB():
  lb = LGBMClassifier()
  lb.fit(x_train, y_train)
  return lb

def phase1():
  """
  Thực nghiệm kịch bản 1
  """
  # Train MLP model with full dataset
  dt = trainDT()
  xb = trainXB()

  # Calculate + Save shap_values (IMPORTANT)
  # DT -> XGBoost
  print("\n### CALCULATE SHAP_VALUES ###")
  for i, model in enumerate([dt, xb]):
    model_name = "dt" if i == 0 else "xb"
    shap_values = explainer.treeShap(model, x_train)
    with open(f"dataset/cicids2018/phase1/{model_name}_shap", "wb") as file:
      pickle.dump(shap_values, file)

  # Using test dataset to generate adversarial samples using AMM method
  for i in ["dt", "xb"]:
    print(f"\n### CALCULATE AMM - {i} ###")
    generator = AMMGenerator()
    shap_values = explainer.loadShap(f"dataset/cicids2018/phase1/{i}_shap")
    if i == "dt": 
      result_feature = generator.AMMFeatureSelection(x_train, shap_values[:,:,1], x_train.columns.to_list(), trigger_size = len(x_train.columns))
    else:
      result_feature = generator.AMMFeatureSelection(x_train, shap_values, x_train.columns.to_list(), trigger_size = len(x_train.columns))
    results = {key: result_feature[key] for key in result_feature}
    amm_path = generator.saveAmmPatch(i, "cicids2018", results)

    print("\n### MANIPULATE FEATURE ###")
    # Generate adversarial samples by manipulate subset of feature
    x_amm = generator.manipulateFeature(amm_path, x_test, y_test)

    print("\n### PHASE 1 - EVALUATE ADVERSARIALS ###")
    # Evaluate CNN 2 layers and 4 layers model with adversarials
    for i in [2,4]:
      print(f"\n### PHASE 1 - CNN {i} Layers ###")
      cnn = tf.keras.models.load_model(f'dataset/cicids2018/model/cnn_{i}layer', compile=False)
      utils.compile(cnn)
      y_pred = cnn.predict(x_amm, verbose = 2) > 0.5
      utils.evaluate(y_test, y_pred)
  return

def phase2():
  """
  Thực nghiệm kịch bản 2
  """
  for i in FRACTION:
    # get data and train models
    (x_frac, y_frac) = utils.getDataFraction(FILE_TRAIN, i)
    
    dt = trainDT(x_frac, y_frac)  
    xb = trainXB(x_frac, y_frac)
    
    # calculate shap_values
    print(f"\n### PHASE 2 - CALCULATE SHAP_VALUES FOR FRACTION {i} ###")

    for counter, model in enumerate([dt, xb]):
      model_name = "dt" if counter == 0 else "rf"
      shap_values = explainer.treeShap(model, x_frac)

      # save shap_values
      with open(f"dataset/cicids2018/phase2/{counter}_shap_{i}", "ab") as file:
        pickle.dump(shap_values, file)
  
    # Calculate amm values
    for counter in range(2):
      print(f"\n### CALCULATE AMM - {counter} ###")
      generator = AMMGenerator()
      shap_values = explainer.loadShap(f"dataset/cicids2018/phase2/{counter}_shap_{i}")
      if counter == 0: 
        result_feature = generator.AMMFeatureSelection(x_frac, shap_values[:,:,0], x_frac.columns.to_list(), trigger_size = len(x_frac.columns))
      else:
        result_feature = generator.AMMFeatureSelection(x_frac, shap_values, x_frac.columns.to_list(), trigger_size = len(x_frac.columns))
    
      results = {key: result_feature[key] for key in result_feature}
      with open(f'dataset/cicids2018/phase2/{counter}_{i}_patch.json', 'w') as file:
        json.dump(results, file)

      # Generate adversarial samples by manipulate subset of feature
      x_amm = generator.manipulateFeature(f"dataset/cicids2018/phase2/{counter}_{i}_patch.json", x_test, y_test)

      print(f"\n### PHASE 2 - EVALUATE WITH FRACTION {i} ###")

      # Evaluate CNN 2 layers and 4 layers model with adversarials
      for temp in [2,4]:
        print(f"\n### PHASE 2 - MLP >< CNN {temp} Layers ###")
        cnn = tf.keras.models.load_model(f'dataset/cicids2018/model/cnn_{temp}layer', compile=False)
        utils.compile(cnn)
        y_pred = cnn.predict(x_amm, verbose = 2) > 0.5  
        utils.evaluate(y_test, y_pred)
  return

def dtFunction():
  """
  Only use this function for surrogate model Decision Tree
  """
  def trainDTLocal(x_train, y_train):
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    return dt
  
  for fraction in FRACTION:
    (x_frac, y_frac) = utils.getDataFraction(FILE_TRAIN, fraction) 
    dt = trainDTLocal(x_frac, y_frac)
    # calculate shap_values
    print(f"\n### PHASE 2 - CALCULATE SHAP_VALUES FOR FRACTION {fraction} ###")
    shap_values = explainer.treeShap(dt, x_frac)

    # save shap_values
    with open(f"dataset/cicids2018/phase2/0_shap_{fraction}", "ab") as file:
      pickle.dump(shap_values, file)

    print(f"\n### PHASE 2 - CALCULATE AMM FOR FRACTION {fraction} ###")
    
    generator = AMMGenerator()
    # Calculate amm values
    shap_values = explainer.loadShap(f"dataset/cicids2018/phase2/0_shap_{fraction}")
    for counter in range(2):
      print(f"TEST CASE FOR DECISION TREE - {counter}")
      result_feature = generator.AMMFeatureSelection(x_frac, shap_values[:,:,counter], x_frac.columns.to_list(), trigger_size = len(x_frac.columns))
      results = {key: result_feature[key] for key in result_feature}
      with open(f'dataset/cicids2018/phase2/0_{fraction}_case{counter}_patch.json', 'w') as file:
        json.dump(results, file)

      # Generate adversarial samples by manipulate subset of feature
      x_amm = generator.manipulateFeature(f"dataset/cicids2018/phase2/0_{fraction}_case{counter}_patch.json", x_test, y_test)
      print(f"\n### PHASE 2 - EVALUATE WITH FRACTION {fraction} TEST CASE {counter} ###")

      # # Evaluate CNN 2 layers and 4 layers model with adversarials
      for temp in [2,4]:
        print(f"\n### PHASE 2 - Decision Tree >< CNN {temp} Layers ###")
        cnn = tf.keras.models.load_model(f'dataset/cicids2018/model/cnn_{temp}layer', compile=False)
        utils.compile(cnn)
        y_pred = cnn.predict(x_amm, verbose = 2) > 0.5  
        utils.evaluate(y_test, y_pred)

      # Evaluate CNN 2 layers and 4 layers model with adversarials
      print(f"\n### PHASE 2 - Decision Tree >< LightGBM ###")
      lb = utils.loadModel("dataset/cicids2018/model/lightgbm")
      y_pred = lb.predict(x_amm)
      utils.evaluate(y_test, y_pred)

      print(f"\n### PHASE 2 - Decision Tree >< Random Forest ###")
      rf = utils.loadModel("dataset/cicids2018/model/rf")
      y_pred = rf.predict(x_amm)
      utils.evaluate(y_test, y_pred)
  return

def xbFunction():
  """
  Only use this function for surrogate model Decision Tree
  """
  def trainXBLocal():
    params = {
    'tree_method': 'approx',
    'objective': 'binary:logistic',
    }
    num_boost_round = 20
    xb = XGBClassifier(n_estimators=num_boost_round, **params)
    xb.fit(x_frac, y_frac)
    return xb
  
  for fraction in FRACTION:
    (x_frac, y_frac) = utils.getDataFraction(FILE_TRAIN, fraction) 
    xb = trainXBLocal()
    # calculate shap_values
    print(f"\n### PHASE 2 - CALCULATE SHAP_VALUES FOR FRACTION {fraction} ###")
    shap_values = explainer.treeShap(xb, x_frac)

    # save shap_values
    with open(f"dataset/cicids2018/phase2/xb_shap_{fraction}", "ab") as file:
      pickle.dump(shap_values, file)

    print(f"\n### PHASE 2 - CALCULATE AMM FOR FRACTION {fraction} ###")
    
    generator = AMMGenerator()
    # Calculate amm values
    shap_values = explainer.loadShap(f"dataset/cicids2018/phase2/xb_shap_{fraction}")
    print(f"TEST CASE FOR XGBoost")
    result_feature = generator.AMMFeatureSelection(x_frac, shap_values, x_frac.columns.to_list(), trigger_size = len(x_frac.columns))
    results = {key: result_feature[key] for key in result_feature}
    with open(f'dataset/cicids2018/phase2/xb_{fraction}_patch.json', 'w') as file:
      json.dump(results, file)

    # Generate adversarial samples by manipulate subset of feature
    x_amm = generator.manipulateFeature(f"dataset/cicids2018/phase2/xb_{fraction}_patch.json", x_test, y_test)
    print(f"\n### PHASE 2 - EVALUATE WITH FRACTION {fraction} ###")

    # # Evaluate CNN 2 layers and 4 layers model with adversarials
    for temp in [2,4]:
      print(f"\n### PHASE 2 - XGBoost >< CNN {temp} Layers ###")
      cnn = tf.keras.models.load_model(f'dataset/cicids2018/model/cnn_{temp}layer', compile=False)
      utils.compile(cnn)
      y_pred = cnn.predict(x_amm, verbose = 2) > 0.5  
      utils.evaluate(y_test, y_pred)

    print(f"\n### PHASE 2 - XGBoost >< LightGBM ###")
    lb = utils.loadModel("dataset/cicids2018/model/lightgbm")
    y_pred = lb.predict(x_amm)
    utils.evaluate(y_test, y_pred)

    print(f"\n### PHASE 2 - XGBoost >< Random Forest ###")
    rf = utils.loadModel("dataset/cicids2018/model/rf")
    y_pred = rf.predict(x_amm)
    utils.evaluate(y_test, y_pred)
  return

if __name__ == '__main__':
  (x_test, y_test) = utils.getDataFraction(FILE_TRAIN, 0.5) 
  dtFunction()

  # print("\n### Train + Test LightGBM model ###")
  # lb = trainLB()
  # utils.saveModel(lb, "dataset/cicids2018/model/lightgbm")
  # y_pred = lb.predict(x_test)
  # utils.evaluate(y_test, y_pred)

  # print("\n### Train + Test Random Forest model ###")
  # rf = trainRF()
  # y_pred = rf.predict(x_test)
  # utils.evaluate(y_test, y_pred)
  # utils.saveModel(rf, "dataset/cicids2018/model/rf")

  # input_shape = (x_test.shape[1],)
  # print("\n### TRAIN + EVALUATE + SAVE MODEL CNN2 | CNN4 ###")
  # trainCNN2()
  # trainCNN4()
  # print('\n### RUN PHASE 1 ###')
  # phase1()
  # print("\n### RUN PHASE 2 ###")
  # phase2()
