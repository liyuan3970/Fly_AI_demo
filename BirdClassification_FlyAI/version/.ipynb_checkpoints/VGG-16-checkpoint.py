# -*- coding: utf-8 -*

import argparse
import tensorflow as tf
from flyai.dataset import Dataset
import tensorflow.contrib.slim as slim
from model import Model
from path import MODEL_PATH, LOG_PATH,DATA_PATH

# 数据获取辅助类
dataset = Dataset()

# 模型操作辅助类
model = Model(dataset)

'''
使用tensorflow实现自己的算法

'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()
# 定义命名空间
x = tf.placeholder(tf.float32, shape=[None, 80, 80, 3], name='input_x')
y = tf.placeholder(tf.float32, shape=[None, 200], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#learning_rate = 0.001




x_image = tf.reshape(x, [-1, 80, 80, 3])

# 定义卷积操作
def conv_op(input, name, kernel_h, kernel_w, num_out, step_h, step_w, para):
    # num_in是输入的深度，这个参数被用来确定过滤器的输入通道数
    num_in = input.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kernel_h, kernel_w, num_in, num_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input, kernel, (1, step_h, step_w, 1), padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[num_out], dtype=tf.float32),
                             trainable=True, name="b")
        # 计算relu后的激活值
        activation = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        para += [kernel, biases]
        return activation


# 定义全连操作
def fc_op(input, name, num_out, para):
    # num_in为输入单元的数量
    num_in = input.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        weights = tf.get_variable(scope + "w", shape=[num_in, num_out], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[num_out], dtype=tf.float32), name="b")

        # tf.nn.relu_layer()函数会同时完成矩阵乘法以加和偏置项并计算relu激活值
        # 这是分步编程的良好替代
        activation = tf.nn.relu_layer(input, weights, biases)

        para += [weights, biases]
        return activation


# 定义前向传播的计算过程，input参数的大小为224x224x3，也就是输入的模拟图片数据
def inference_op(input, keep_prob):
    parameters = []

    # 第一段卷积，输出大小为112x112x64(省略了第一个batch_size参数)
    conv1_1 = conv_op(input, name="conv1_1", kernel_h=3, kernel_w=3, num_out=4,
                      step_h=1, step_w=1, para=parameters)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kernel_h=3, kernel_w=3, num_out=64,
                      step_h=1, step_w=1, para=parameters)
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool1")
    print(pool1.op.name, ' ', pool1.get_shape().as_list())

    # 第二段卷积，输出大小为56x56x128(省略了第一个batch_size参数)
    conv2_1 = conv_op(pool1, name="conv2_1", kernel_h=3, kernel_w=3, num_out=128,
                      step_h=1, step_w=1, para=parameters)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kernel_h=3, kernel_w=3, num_out=128,
                      step_h=1, step_w=1, para=parameters)
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool2")
    print(pool2.op.name, ' ', pool2.get_shape().as_list())

    # 第三段卷积，输出大小为28x28x256(省略了第一个batch_size参数)
    conv3_1 = conv_op(pool2, name="conv3_1", kernel_h=3, kernel_w=3, num_out=256,
                      step_h=1, step_w=1, para=parameters)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kernel_h=3, kernel_w=3, num_out=256,
                      step_h=1, step_w=1, para=parameters)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kernel_h=3, kernel_w=3, num_out=256,
                      step_h=1, step_w=1, para=parameters)
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool3")
    print(pool2.op.name, ' ', pool2.get_shape().as_list())

    # 第四段卷积，输出大小为14x14x512(省略了第一个batch_size参数)
    conv4_1 = conv_op(pool3, name="conv4_1", kernel_h=3, kernel_w=3, num_out=512,
                      step_h=1, step_w=1, para=parameters)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kernel_h=3, kernel_w=3, num_out=512,
                      step_h=1, step_w=1, para=parameters)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kernel_h=3, kernel_w=3, num_out=512,
                      step_h=1, step_w=1, para=parameters)
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool4")
    print(pool4.op.name, ' ', pool4.get_shape().as_list())

    # 第五段卷积，输出大小为7x7x512(省略了第一个batch_size参数)
    conv5_1 = conv_op(pool4, name="conv5_1", kernel_h=3, kernel_w=3, num_out=512,
                      step_h=1, step_w=1, para=parameters)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kernel_h=3, kernel_w=3, num_out=512,
                      step_h=1, step_w=1, para=parameters)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kernel_h=3, kernel_w=3, num_out=512,
                      step_h=1, step_w=1, para=parameters)
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool5")
    print(pool5.op.name, ' ', pool5.get_shape().as_list())

    # pool5的结果汇总为一个向量的形式
    pool_shape = pool5.get_shape().as_list()
    flattened_shape = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshped = tf.reshape(pool5, [-1, flattened_shape], name="reshped")

    # 第一个全连层
    fc_6 = fc_op(reshped, name="fc6", num_out=4096, para=parameters)
    fc_6_drop = tf.nn.dropout(fc_6, keep_prob, name="fc6_drop")

    # 第二个全连层
    fc_7 = fc_op(fc_6_drop, name="fc7", num_out=4096, para=parameters)
    fc_7_drop = tf.nn.dropout(fc_7, keep_prob, name="fc7_drop")

    # 第三个全连层及softmax层
    fc_8 = fc_op(fc_7_drop, name="fc8", num_out=200, para=parameters)
    softmax = tf.nn.softmax(fc_8)

    # predictions模拟了通过argmax得到预测结果
    predictions = tf.argmax(softmax, 1,name='test')
    print("###########################################################")
    return predictions, softmax, fc_8, parameters
        
        
g = tf.Graph()
predictions, softmax, fc_8, parameters = inference_op(x_image, keep_prob)
prediction = tf.add(softmax, 0, name='y_conv')
print("##################################")
print(prediction.shape)

loss = slim.losses.softmax_cross_entropy(prediction, y)  # 读入标签
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = slim.learning.create_train_op(loss, optimizer)  # 训练以及优化

# 求准确率：
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged = tf.summary.merge_all()
saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
    #x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)
    for i in range(args.EPOCHS):
        x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)
        train_dict = {x: x_train, y: y_train, keep_prob: 0.5}
        sess.run(train_op, feed_dict=train_dict)
        losses, acc_ = sess.run([loss, accuracy], feed_dict=train_dict)
        y_convs = sess.run(prediction, feed_dict=train_dict)
        print("step:{}, loss:{}, acc:{}".format(i + 1, losses, acc_))
        model.save_model(sess, MODEL_PATH, overwrite=True)