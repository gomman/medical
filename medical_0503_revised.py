# Lab 10 MNIST and Dropout
import tensorflow as tf
import random
import numpy as np
import time

# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

#tf.set_random_seed(777)  # reproducibility
tic = time.time()
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for

#P = np.array([[0,1,1,0,0,1], [1,0,1,0,1,0], [1,0,0,1,0,1], [0,1,0,1,1,0]])
#res_ue_num = np.array([[1,2,3], [1,4,5],[2,4,6], [3,5,6]])
# parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 10
iteration_num = 1000
# input place holders
X = tf.placeholder(tf.float32, [None, 24])
Y = tf.placeholder(tf.float32, [None, 2])

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
W1 = tf.get_variable("W1", shape=[24, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[512, 2],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([2]))
hypothesis = tf.matmul(L4, W5) + b5

# define cost/loss & optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
cost = tf.reduce_mean(tf.square(hypothesis-Y))#+0.0001*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4)+tf.nn.l2_loss(W5)+tf.nn.l2_loss(b1)+tf.nn.l2_loss(b2)+tf.nn.l2_loss(b3)+tf.nn.l2_loss(b4)+tf.nn.l2_loss(b5))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
# initialize
y_temp = np.loadtxt('y_data.txt')
y = np.zeros((len(y_temp), 2))
for j in range(len(y_temp)):
    if y_temp[j] == 0:
        y[j][0] = 1
        y[j][1] = 0
    else:
        y[j][1] = 1
        y[j][0] = 0
x = np.loadtxt('x_data.txt')
accuracy_sum = 0

for n in range(iteration_num):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train my model
    total_batch = int(len(y) / batch_size)
    test_batch_idx = np.random.randint(total_batch)
    test_rows = np.arange(test_batch_idx*batch_size,(test_batch_idx+1)*batch_size)
    x_train = np.delete(x,test_rows,0)
    x_test = x[test_rows]
    y_train = np.delete(y,test_rows,0)
    y_test = y[test_rows]
    for epoch in range(training_epochs):
        avg_cost = 0

        for i in range(total_batch-1):
            batch_x = x_train[batch_size*i:batch_size*(i+1)][:]
            batch_y = y_train[batch_size*i:batch_size*(i+1)][:]
            feed_dict = {X: batch_x, Y: batch_y, keep_prob: 0.7}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        #print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('%03d' % n,'th Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print('Accuracy:', sess.run(accuracy, feed_dict={
    #      X: x[batch_size*(total_batch-1):batch_size*total_batch][:], Y:y[batch_size*(total_batch-1):batch_size*total_batch][:], keep_prob: 1.0}))
    accuracy_sum = accuracy_sum + sess.run(accuracy, feed_dict={X: x_test, Y:y_test, keep_prob: 1.0})
# Get one and predict
print ('avraged accuracy is',accuracy_sum/iteration_num)

# Get one and predict
# r = random.randint(0, y - 1)
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(
#     tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()
