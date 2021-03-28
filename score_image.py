#! /usr/bin/python3
import warnings
# Supress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import tfmodel

home_dir = os.path.dirname(sys.argv[0])
model_file = os.path.join(home_dir, 'static/tflite/model.tflite')
label_file = os.path.join(home_dir, 'static/tflite/dict.txt')

tf_classifier = tfmodel.Model(model_file, label_file)

for path_to_image in sys.argv[1:]:
    results = tf_classifier.classify(path_to_image, min_confidence=0.01)
    for i,r in enumerate(results):
        print('{},{},{},{:.2f}'.format(path_to_image, i+1, r['label'], float(r['confidence'])))

