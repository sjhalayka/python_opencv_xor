import cv2
import numpy as np

ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([2, 5, 2], dtype=np.uint8))
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1)
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ann.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 1, 0.000001 ))

input_array0 = np.array([[0.0, 0.0]], dtype=np.float32)
output_array0 = np.array([[0.0, 1.0]], dtype=np.float32)

input_array1 = np.array([[1.0, 0.0]], dtype=np.float32)
output_array1 = np.array([[0.0, 1.0]], dtype=np.float32)

input_array2 = np.array([[0.0, 1.0]], dtype=np.float32)
output_array2 = np.array([[0.0, 1.0]], dtype=np.float32)

input_array3 = np.array([[1.0, 1.0]], dtype=np.float32)
output_array3 = np.array([[1.0, 0.0]], dtype=np.float32)

td0 = cv2.ml.TrainData_create(input_array0, cv2.ml.ROW_SAMPLE, output_array0)
td1 = cv2.ml.TrainData_create(input_array1, cv2.ml.ROW_SAMPLE, output_array1)
td2 = cv2.ml.TrainData_create(input_array2, cv2.ml.ROW_SAMPLE, output_array2)
td3 = cv2.ml.TrainData_create(input_array3, cv2.ml.ROW_SAMPLE, output_array3)

ann.train(td0, cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)

for i in range(0, 10000):
    ann.train(td0, cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
    ann.train(td1, cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
    ann.train(td2, cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
    ann.train(td3, cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)

print(ann.predict(input_array0))
print(ann.predict(input_array1))
print(ann.predict(input_array2))
print(ann.predict(input_array3))
