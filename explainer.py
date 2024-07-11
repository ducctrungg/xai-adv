import shap
import pickle

def calculateSHAP(model, dataset):
  """
  Caculate and return shap_values for Deep Learning model
  """
  shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
  explainer = shap.DeepExplainer(model, data=shap.utils.sample(dataset.to_numpy(), 500))
  shap_values = explainer.shap_values(dataset.to_numpy())
  return shap_values

def treeShap(model, dataset):
  """
  Caculate and return shap_values for Machine Learning tree-based model
  """
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(dataset.to_numpy())
  return shap_values

def loadShap(file_path):
  with open(file_path, 'rb') as file:
    shap_values = pickle.load(file)
  return shap_values