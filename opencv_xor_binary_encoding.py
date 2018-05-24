import cv2
import numpy as np

ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([2, 5, 1], dtype=np.uint8))
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1)
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ann.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 1, 0.000001 ))

input_array = np.array([ [0.0, 0.0],
                         [1.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 1.0]
                         ], dtype=np.float32)
                        
output_array = np.array([ [0.0],
                          [1.0],
                          [1.0],
                          [0.0]
                          ], dtype=np.float32)

td = cv2.ml.TrainData_create(input_array, cv2.ml.ROW_SAMPLE, output_array)

ann.train(td, cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)

for i in range(0, 10000):
    ann.train(td, cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)

print(ann.predict(input_array))
