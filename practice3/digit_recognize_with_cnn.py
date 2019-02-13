import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

OUTPUT_SIZE = 7 * 7 * 64


def cnn_layer(input_shape, filter, kernel_size, input_data):
    W = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, input_shape, filter], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[filter]))
    # 1) Layer of conv2d
    conv = tf.nn.conv2d(input_data, W, strides=[1, 1, 1, 1], padding='SAME') + b
    # 2) ReLu
    relu = tf.nn.relu(conv)
    # 3) Pooling
    return tf.layers.max_pooling2d(relu, 2, 2)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])  # input
# x_image = tf.reshape(x, [-1,28,28,1])
x_image = tf.reshape(x, [-1, 28, 28, 1])
# Kernel – 5x5, filters – 32
# 1) Layer of conv2d
# 2) ReLu
# 3) Pooling
pooling_layer_1 = cnn_layer(1, 32, 5, x_image)
# 4) + one more layer* with 64 filters
pooling_layer_2 = cnn_layer(32, 64, 5, pooling_layer_1)
# 5) h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7*7*64])
h_pool_2_flat = tf.reshape(pooling_layer_2, [-1, OUTPUT_SIZE])
# fully connect
W = tf.Variable(tf.truncated_normal([OUTPUT_SIZE, 1024], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[1024]))

fully_connect_layer = tf.add(tf.matmul(h_pool_2_flat, W), b)
fully_connect_layer_relu = tf.nn.relu(fully_connect_layer)
fully_connect = tf.layers.dropout(fully_connect_layer_relu, rate=0.30)
# prediction
W_out = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_out = tf.Variable(tf.constant(0.1, shape=[10]))

y = tf.add(tf.matmul(fully_connect, W_out), b_out)
y = tf.nn.softmax(y)
y_ = tf.placeholder(tf.float32, [None, 10])

keep_probability = tf.placeholder(tf.float32)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_probability: 0.5})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# 6) Use Adam
