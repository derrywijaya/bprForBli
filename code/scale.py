import tensorflow.python.platform

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

# Global variables.
tf.app.flags.DEFINE_string('input', None,
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_string('output', None,
                           'File containing the training data (labels & features).')
FLAGS = tf.app.flags.FLAGS

# Extract numpy representations of the labels and features given rows consisting of:
#   label, feat_0, feat_1, ..., feat_n
def extract_data():
    inputfile = FLAGS.input
    fvecs = []

    for line in file(inputfile):
	row = line.split(" ")
        fvecs.append([float(x) for x in row[1:]])

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.matrix(fvecs).astype(np.float32)

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs_np

def main(argv=None):
    outputfile = FLAGS.output
    train_data = extract_data()
    train_scaled = preprocessing.scale(train_data)
    max_abs_scaler = preprocessing.MaxAbsScaler()
    train_data = max_abs_scaler.fit_transform(train_scaled)
    np.savetxt(outputfile,train_data,delimiter=" ")

if __name__ == '__main__':
    tf.app.run()
