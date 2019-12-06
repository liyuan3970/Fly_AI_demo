import argparse
import tensorflow as tf
from flyai.dataset import Dataset

from model import Model

# 数据获取辅助类
from path import MODEL_PATH

dataset = Dataset()
dataset.get_all_processor_data()
model = Model(dataset)

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()
'''
使用tensorflow实现自己的算法

'''
# 定义命名空间
x = tf.placeholder(tf.float32, shape=[None, 200, 200, 3], name='input_x')
y = tf.placeholder(tf.int64, shape=[None], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# 初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


# 初始化偏置
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


#  定义卷积层网络的步长
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化层网络的大小和步长
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def fc(name, x, out_channel):
    shape = x.get_shape().as_list()
    if len(shape) == 4:
        size = shape[-1] * shape[-2] * shape[-3]
    else:
        size = shape[1]
    x_flat = tf.reshape(x, [-1, size])
    with tf.variable_scope(name):
        W = tf.get_variable(name='weights', shape=[size, out_channel], dtype=tf.float32)
        b = tf.get_variable(name='biases', shape=[out_channel], dtype=tf.float32)
        res = tf.matmul(x_flat, W)
        out = tf.nn.relu(tf.nn.bias_add(res, b))

    return out


with tf.name_scope('Conv1'):
    # 初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([3, 3, 3, 32], name='W_conv1')
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32], name='b_conv1')
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('Conv2'):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([3, 3, 32, 64], name='W_conv2')
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('Conv3'):
    # 初始化第三个卷积层的权值和偏置
    with tf.name_scope('W_conv3'):
        W_conv3 = weight_variable([3, 3, 64, 128], name='W_conv3')
    with tf.name_scope('b_conv3'):
        b_conv3 = bias_variable([128], name='b_conv3')
    with tf.name_scope('conv2d_3'):
        conv2d_3 = conv2d(h_pool2, W_conv3) + b_conv3
    with tf.name_scope('relu'):
        h_conv3 = tf.nn.relu(conv2d_3)
    with tf.name_scope('h_pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

fc1 = fc(name='fc1', x=h_pool3, out_channel=128)
fc2 = fc(name='fc2', x=fc1, out_channel=62)
prediction = tf.nn.relu(fc2, name="prediction")
y_conv = tf.nn.relu(fc2, name="y_conv")

with tf.name_scope('loss_fucntion'):
    loss_function = tf.losses.softmax_cross_entropy(tf.one_hot(y, depth=62), prediction)
    tf.summary.scalar('loss_function', loss_function)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_function)

# 求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), y)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

best_accuracy = 0.0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(args.EPOCHS):
        x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)
        sess.run(train_step, feed_dict={x: x_train, y: y_train, keep_prob: 0.7})
        train_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
        loss = sess.run(loss_function, feed_dict={x: x_train, y: y_train, keep_prob: 1.0})
        print(i + 1, loss, train_acc)
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            model.save_model(sess, MODEL_PATH, overwrite=True)
            print("stpe %d, best_accuracy %g" % (i + 1, best_accuracy))
            print(str(i + 1) + "/" + str(args.EPOCHS))