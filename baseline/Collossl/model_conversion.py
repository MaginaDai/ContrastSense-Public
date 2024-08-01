import tensorflow as tf
import tf2onnx

import onnx
from onnx2pytorch import ConvertModel
import pdb

loaded_model = tf.keras.models.load_model("scripts/runs/lr/Shoaib/model_best_lowest.hdf5", compile=False)
pdb.set_trace()
onnx_model, _ = tf2onnx.convert.from_keras(loaded_model)
pytorch_model = ConvertModel(onnx_model)
pytorch_model