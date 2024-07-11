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

import explainer  # custom library
import utils  # custom library
from amm import AMMGenerator  # custom library
from mlp import MLP  # custom library
from cnn import CNN2, CNN4

FILE_TRAIN = "dataset/insdn/train.csv"
FILE_TEST = "dataset/insdn/test.csv"
FRACTION = [0.2, 0.4, 0.6, 0.8, 1.0]

def TE(model, x, y):
  utils.compile(model)
  model.fit(
    x, 
    y,
    epochs = 20,
    batch_size = 128,
    verbose = 2
  )
  y_pred = model.predict(x_test, verbose = 2) > 0.5
  print("\n### EVALUATE ###")
  utils.evaluate(y_test, y_pred)  

def trainMLP(x_train, y_train):
  """
  Train and return MLP model
  """
  print("\n### TRANNING MLP MODEL ###")
  mlp = MLP(input_shape)
  TE(mlp, x_train, y_train)
  mlp.save(f"dataset/insdn/phase1/mlp", save_format='tf')
  return mlp

def trainCNN2():
  """
  Train and return CNN 2 layers model
  """
  (x_train, y_train) = utils.getData(FILE_TRAIN)
  print("\n### TRANNING CNN2 MODEL ###")
  cnn2 = CNN2(input_shape)
  TE(cnn2, x_train, y_train)
  cnn2.save('dataset/insdn/model/cnn_2layer', save_format='tf')
  return cnn2
 
def trainCNN4():
  """
  Train and return CNN 4 layers model
  """
  (x_train, y_train) = utils.getData(FILE_TRAIN)
  print("\n### TRANNING CNN4 MODEL ###")
  cnn4 = CNN4(input_shape)
  TE(cnn4, x_train, y_train)
  cnn4.save('dataset/insdn/model/cnn_4layer', save_format='tf')
  return cnn4

def phase1():
  """
  Thực nghiệm kịch bản 1
  """
  # Train MLP model with full dataset
  (x_train, y_train) = utils.getData(FILE_TRAIN)
  mlp = trainMLP(x_train, y_train)

  # Calculate + Save shap_values (IMPORTANT)
  print("\n### CALCULATE SHAP_VALUES ###")
  shap_values = explainer.calculateSHAP(mlp, x_train)
  with open("dataset/insdn/phase1/mlp_shap", "wb") as file:
    pickle.dump(shap_values, file)

  # Using test dataset to generate adversarial samples using AMM method
  print("\n### CALCULATE AMM ###")
  generator = AMMGenerator()
  shap_values = explainer.loadShap("dataset/insdn/phase1/mlp_shap")
  result_feature = generator.AMMFeatureSelection(x_train, shap_values[:,:,0], x_train.columns.to_list(), trigger_size = len(x_train.columns))
  results = {key: result_feature[key] for key in result_feature}
  amm_path = generator.saveAmmPatch("mlp", "insdn", results)
  
  print("\n### MANIPULATE FEATURE ###")
  # Generate adversarial samples by manipulate subset of feature
  x_amm = generator.manipulateFeature(amm_path, x_test, y_test)

  print("\n### PHASE 1 - EVALUATE ADVERSARIALS ###")
  # Evaluate CNN 2 layers and 4 layers model with adversarials
  for i in [2,4]:
    print(f"\n### PHASE 1 - MLP >< CNN {i} Layers ###")
    cnn = tf.keras.models.load_model(f'dataset/insdn/model/cnn_{i}layer', compile=False)
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
    mlp = trainMLP(x_frac, y_frac)

    # save model
    mlp.save(f"dataset/insdn/phase2/mlp_{i}", save_format='tf')

    # calculate shap_values
    print(f"\n### PHASE 2 - CALCULATE SHAP_VALUES FOR FRACTION {i} ###")
    shap_values = explainer.calculateSHAP(mlp, x_frac)

    # save shap_values
    with open(f"dataset/insdn/phase2/mlp_shap_{i}", "ab") as file:
      pickle.dump(shap_values, file)
  
    # Calculate amm values
    generator = AMMGenerator()
    shap_values = explainer.loadShap(f"dataset/insdn/phase2/mlp_shap_{i}")
    result_feature = generator.AMMFeatureSelection(x_frac, shap_values[:,:,0], x_frac.columns.to_list(), trigger_size = len(x_frac.columns))
    results = {key: result_feature[key] for key in result_feature}
    with open(f'dataset/insdn/phase2/amm_{i}_patch.json', 'w') as file:
      json.dump(results, file)

    # Generate adversarial samples by manipulate subset of feature
    x_amm = generator.manipulateFeature(f"dataset/insdn/phase2/amm_{i}_patch.json", x_test, y_test)

    print(f"\n### PHASE 2 - EVALUATE WITH FRACTION {i} ###")

    # Evaluate CNN 2 layers and 4 layers model with adversarials
    for i in [2,4]:
      print(f"\n### PHASE 2 - MLP >< CNN {i} Layers ###")
      cnn = tf.keras.models.load_model(f'dataset/insdn/model/cnn_{i}layer', compile=False)
      utils.compile(cnn)
      y_pred = cnn.predict(x_amm, verbose = 2) > 0.5  
      utils.evaluate(y_test, y_pred)

      print(f"\n### PHASE 2 - MLP >< LightGBM ###")
      lb = utils.loadModel("dataset/insdn/model/lightgbm")
      y_pred = lb.predict(x_amm)
      utils.evaluate(y_test, y_pred)

      print(f"\n### PHASE 2 - MLP >< Random Forest ###")
      rf = utils.loadModel("dataset/insdn/model/rf")
      y_pred = rf.predict(x_amm)
      utils.evaluate(y_test, y_pred)
  return

def superFunction():
  # superFunction use for deeplearning surrogate model attack machine learning model
  def phase1():
    # Generate adversarial samples by manipulate subset of feature
    print("\n### MANIPULATE FEATURE ###")
    generator = AMMGenerator()
    x_amm = generator.manipulateFeature("dataset/insdn/phase1/mlp_amm.json", x_test, y_test)

    print("\n### PHASE 1 - EVALUATE ADVERSARIALS ###")
    # Evaluate CNN 2 layers and 4 layers model with adversarials
    print(f"\n### PHASE 1 - MLP >< LightGBM ###")
    lb = utils.loadModel("dataset/insdn/model/lightgbm")
    y_pred = lb.predict(x_amm)
    utils.evaluate(y_test, y_pred)

    print(f"\n### PHASE 1 - MLP >< Random Forest ###")
    rf = utils.loadModel("dataset/insdn/model/rf")
    y_pred = rf.predict(x_amm)
    utils.evaluate(y_test, y_pred)

  def phase2():
    for i in FRACTION:
      print(f"\n### PHASE 2 - EVALUATES FOR FRACTION {i} ###")
    
      # Calculate amm values
      generator = AMMGenerator()
      # Generate adversarial samples by manipulate subset of feature
      x_amm = generator.manipulateFeature(f"dataset/insdn/phase2/amm_{i}_patch.json", x_test, y_test)

      # Evaluate CNN 2 layers and 4 layers model with adversarials
      print(f"\n### PHASE 2 - MLP >< LightGBM ###")
      lb = utils.loadModel("dataset/insdn/model/lightgbm")
      y_pred = lb.predict(x_amm)
      utils.evaluate(y_test, y_pred)

      print(f"\n### PHASE 2 - MLP >< Random Forest ###")
      rf = utils.loadModel("dataset/insdn/model/rf")
      y_pred = rf.predict(x_amm)
      utils.evaluate(y_test, y_pred)

  phase1()
  # phase2()

if __name__ == '__main__':
  (x_test, y_test) = utils.getData(FILE_TEST)
  # (x_train, y_train) = utils.getData(FILE_TRAIN)
  input_shape = (x_test.shape[1],)
  # print("\n### TRAIN + EVALUATE + SAVE MODEL CNN2 | CNN4 ###")
  # trainCNN2()
  # trainCNN4()
  # print('\n### RUN PHASE 1 ###')
  # phase1()
  print("\n### RUN PHASE 2 ###")
  phase2()
  # superFunction()
