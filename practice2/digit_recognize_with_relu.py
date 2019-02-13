import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])  # input

# add relu layer
weight_relu = tf.Variable(tf.truncated_normal(shape=[784, 784], stddev=0.1))
bias_relu = tf.Variable(tf.truncated_normal(shape=[784], stddev=0.1))
relu_layer = tf.layers.dropout(tf.nn.relu(tf.matmul(x, weight_relu) + bias_relu))

W = tf.Variable(tf.zeros([784, 10]))  # weight

b = tf.Variable(tf.zeros([10]))  # bias

y = tf.nn.softmax(tf.matmul(relu_layer, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

keep_probability = tf.placeholder(tf.float32)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_probability: 0.5})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
