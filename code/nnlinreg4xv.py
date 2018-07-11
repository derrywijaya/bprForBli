import tensorflow.python.platform
import math
import numpy as np
import tensorflow as tf


# Global variables.
BATCH_SIZE = 100  # The number of training examples to use per training step.

tf.app.flags.DEFINE_string('traininput', None,
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_string('trainoutput', None,
                           'File containing the test data (labels & features).')
tf.app.flags.DEFINE_string('testinput', None,
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_string('testoutput', None,
                           'File containing the test data (labels & features).')
tf.app.flags.DEFINE_string('project', None,
                           'File containing the test data (labels & features).')
tf.app.flags.DEFINE_string('towrite', None,
                           'File containing the test data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of passes over the training data.')
tf.app.flags.DEFINE_integer('num_hidden', 1,
                            'Number of nodes in the hidden layer.')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
FLAGS = tf.app.flags.FLAGS

# Extract numpy representations of the labels and features given rows consisting of:
#   label, feat_0, feat_1, ..., feat_n
def extract_data():
    traininputfile = FLAGS.traininput
    trainoutputfile = FLAGS.trainoutput
    testinputfile = FLAGS.testinput
    testoutputfile = FLAGS.testoutput
    projectfile = FLAGS.project

    labels = []
    fvecs = []
    testlabels = []
    testfvecs = []
    project = []

    for line in file(traininputfile):
	row = line.split(" ")
        fvecs.append([float(x) for x in row[0:]])

    for line in file(trainoutputfile):
        row = line.split(" ")
        labels.append([float(x) for x in row[0:]])

    for line in file(testinputfile):
	row = line.split(" ")
        testfvecs.append([float(x) for x in row[0:]])

    for line in file(testoutputfile):
        row = line.split(" ")
        testlabels.append([float(x) for x in row[0:]])

    for line in file(projectfile):
	row = line.split(" ")
        project.append([float(x) for x in row[1:]])


    # Convert the array of float arrays into a numpy float matrix.
    project_np = np.matrix(project).astype(np.float32)

    fvecs_np = np.matrix(fvecs).astype(np.float32)
    labels_np = np.matrix(labels).astype(np.float32)
    labels_o = labels_np.transpose()

    testfvecs_np = np.matrix(testfvecs).astype(np.float32)
    testlabels_np = np.matrix(testlabels).astype(np.float32)
    testlabels_o = testlabels_np.transpose()

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs_np,project_np,labels_o,testfvecs_np,testlabels_o

# Init weights method. (Lifted from Delip Rao: http://deliprao.com/archives/100)
def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    
def main(argv=None):
    # Be verbose?
    verbose = FLAGS.verbose

    # Extract it into numpy arrays.
    train_data,project_data,train_labels,test_data,test_labels = extract_data()

    # Get the shape of the training data.
    train_size,num_features_in = train_data.shape
    num_features_out,train_size = train_labels.shape
    project_size,num_features_p = project_data.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # Get the size of layer one.
    num_hidden = FLAGS.num_hidden
    towritefile = FLAGS.towrite

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features_in])
    y_ = tf.placeholder("float", shape=[None, num_features_out])
    
    # Define and initialize the network.

    num_hidden_1 = num_features_in*10
    w_hidden_1 = init_weights([num_features_in, num_hidden_1],'xavier',xavier_params=(num_features_in, num_hidden_1))
    b_hidden_1 = init_weights([1,num_hidden_1],'zeros')
    hidden_1 = tf.nn.tanh(tf.matmul(x,w_hidden_1) + b_hidden_1)
    #hidden_1 = tf.matmul(x,w_hidden_1) + b_hidden_1
    
    num_hidden_2 = num_features_in*10
    w_hidden_2 = init_weights([num_hidden_1, num_hidden_2],'xavier',xavier_params=(num_hidden_1, num_hidden_2))
    b_hidden_2 = init_weights([1,num_hidden_2],'zeros')
    hidden_2 = tf.nn.tanh(tf.matmul(hidden_1,w_hidden_2) + b_hidden_2)
    #hidden_2 = tf.matmul(hidden_1,w_hidden_2) + b_hidden_2

    num_hidden_3 = num_features_in*10
    w_hidden_3 = init_weights([num_hidden_2, num_hidden_3],'xavier',xavier_params=(num_hidden_2, num_hidden_3))
    b_hidden_3 = init_weights([1,num_hidden_3],'zeros')
    hidden_3 = tf.nn.tanh(tf.matmul(hidden_2,w_hidden_3) + b_hidden_3)
    #hidden_3 = tf.matmul(hidden_2,w_hidden_3) + b_hidden_3

    w_hidden_4 = init_weights([num_hidden_3, num_hidden],'xavier',xavier_params=(num_hidden_3, num_hidden))
    b_hidden_4 = init_weights([1,num_hidden],'zeros')
    hidden_4 = tf.nn.tanh(tf.matmul(hidden_3,w_hidden_4) + b_hidden_4)

    # Initialize the output weights and biases.
    w_out = init_weights(
        [num_hidden,num_features_out],
        'xavier',
        xavier_params=(num_hidden,num_features_out))    
    b_out = init_weights([1,num_features_out],'zeros')
    w_out = w_out / float(math.sqrt(num_hidden))
    b_out = b_out / float(math.sqrt(num_hidden))

    # The output layer.
    y = tf.matmul(hidden_4,w_out)+b_out
    #print(y.shape)
    #print(y_.shape)
    #saver = tf.train.Saver()

    # Optimization.
    myloss = tf.reduce_mean(tf.square(y-y_))
    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-5).minimize(myloss)
    
    # Evaluation.
    predicted_class = y
    correct_prediction = y_
    accuracy = tf.reduce_mean(tf.square(y-y_))
    sumaccuracy = tf.reduce_sum(tf.square(y-y_))

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
    	#tf.global_variables_initializer().run()
	tf.initialize_all_variables().run()
    	if verbose:
    	    print 'Initialized!'
    	    print
    	    print 'Training.'
    	    
    	# Iterate and train.
        lossprev = 0
        idx = 0
    	for step in xrange(num_epochs * train_size // BATCH_SIZE):
    	    #if verbose:
    	        #print step,
    	        
    	    offset = (step * BATCH_SIZE) % train_size
    	    batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            #print(batch_data.shape)
    	    batch_labels = train_labels[:, offset:(offset + BATCH_SIZE)]
            #print(batch_labels.shape)
    	    train_step.run(feed_dict={x: batch_data, y_: batch_labels.transpose()})
            #print(sumaccuracy.eval(feed_dict={x: batch_data, y_: batch_labels.transpose()}))
            #print(predicted_class.eval(feed_dict={x: batch_data, y_: batch_labels.transpose()}).shape)
            #print(correct_prediction.eval(feed_dict={x: batch_data, y_: batch_labels.transpose()}).shape)
    	    if verbose and offset >= train_size-BATCH_SIZE:
	        idx = idx + 1
	        losscurr = accuracy.eval(feed_dict={x: test_data, y_: test_labels.transpose()})
                print(idx,losscurr)
		if lossprev == 0:
		     lossprev = losscurr
                if lossprev > losscurr:
		     lossprev = losscurr
                if lossprev < losscurr:
		     diff = losscurr - lossprev
                     lossprev = losscurr
                     if diff < 0.0001:
                         break
	    #	lossdiff = losscurr - lossprev
	    #	if lossdiff < 0:
	    #	    lossdiff = 0 - lossdiff
	    #	print(lossdiff)
	    #	lossprev = losscurr

	offset = 0
	batch_data = project_data[offset:(offset + BATCH_SIZE), :]
	y_out_np = predicted_class.eval(feed_dict={x: batch_data})
	sumacc = sumaccuracy.eval(feed_dict={x: batch_data, y_:batch_data})
        if project_size > BATCH_SIZE:
	    totalstep = project_size // BATCH_SIZE
            for step in xrange(project_size // BATCH_SIZE):
                step = step + 1;
                print step
	        print totalstep
                offset = (step * BATCH_SIZE) % project_size
                batch_data = project_data[offset:(offset + BATCH_SIZE), :]
                y_out=predicted_class.eval(feed_dict={x: batch_data})
                y_out_np = np.concatenate((y_out_np,y_out), axis = 0)
                sumacc = sumacc + sumaccuracy.eval(feed_dict={x: batch_data, y_:batch_data})
                if offset >= project_size-BATCH_SIZE:
                    break
        sumacc = sumacc / project_size
        print(sumacc)
    	#for step in xrange(project_size // BATCH_SIZE):
	#    step = step + 1;
        #    #print(step)
	#    offset = (step * BATCH_SIZE) % project_size
	#    batch_data = project_data[offset:(offset + BATCH_SIZE), :]
	#    y_out=predicted_class.eval(feed_dict={x: batch_data})
	#    y_out_np = np.concatenate((y_out_np,y_out), axis = 0)
        #    sumacc = sumacc + sumaccuracy.eval(feed_dict={x: batch_data, y_:batch_data})
	#    #y_out.append(predicted_class.eval(feed_dict={x: batch_data}))
	#    if offset >= project_size-BATCH_SIZE:
	#        break
        #sumacc = sumacc / project_size
        #print(sumacc)
        #print(y_out_np.shape)	
	#y_out_np = np.reshape(y_out, (-1,100))
	#print(y_out_np.shape)
    	#print "Accuracy:", accuracy.eval(feed_dict={x: train_data, y_: train_labels[:,:].transpose()})
	#weights_hidden = w_hidden.eval()
	#bias_hidden = b_hidden.eval()
	#weights_out = w_out.eval()
	#bias_out = b_out.eval()
        #y_out = predicted_class.eval(feed_dict={x: project_data}) 
	np.savetxt(towritefile,y_out_np,delimiter=" ")
	#np.savetxt("bias_hidden.txt",bias_hidden,delimiter=" ")
	#np.savetxt("weights_out.txt",weights_out,delimiter=" ")
	#np.savetxt("bias_out.txt",bias_out,delimiter=" ")

if __name__ == '__main__':
    tf.app.run()
