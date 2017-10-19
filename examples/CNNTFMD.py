
import gzip
import numpy as np
import pickle

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable( shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积和池化，使用卷积步长为1（stride size）,0边距（padding size） 池化用简单传统的2x2大小的模板做max pooling



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class LogisticRegression(object):

    def __init__(self):
        self.X = tf.placeholder("float", [None, 784])
        self.Y = tf.placeholder("float", [None, 5])

        self.W = tf.Variable(tf.random_normal([28 * 28, 5], stddev=0.01))
        self.b = tf.Variable(tf.zeros([5, ]))
        self.x_image = tf.reshape(self.X, [-1, 28, 28, 1])  # 最后一维代表通道数目，如果是rgb则为3
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])


        # x_image权重向量卷积，加上偏置项，之后应用ReLU函数，之后进行max_polling
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        # 实现第二层卷积

        # 每个5x5的patch会得到64个特征
        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1,self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)

        # 全连接层 图片尺寸变为4x4，加入有1024个神经元的全连接层，把池化层输出张量reshape成向量 乘上权重矩阵，加上偏置，然后进行ReLU

        self.W_fc1 = weight_variable([4 * 4 * 64, 1024])
        self.b_fc1 = bias_variable([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 4 * 4 * 64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        # Dropout， 用来防止过拟合 #加在输出层之前，训练过程中开启dropout，测试过程中关闭
        self.keep_prob = tf.placeholder("float")
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        # 输出层, 添加softmax层
        self.W_fc2 = weight_variable([1024, 5])
        self.b_fc2 = bias_variable([5])

        self.model = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.Y))

        # gradient descent method to minimize error
        self.train = tf.train.GradientDescentOptimizer(0.01).minimize(self.cost)
        # calculate the max pos each row
        self.predict = tf.argmax(self.model, 1)


    def load_datas(self):
        f = gzip.open('cnns.pkl.gz', 'rb')
        train_set, test_set = pickle.load(f, encoding='latin1')
        f.close()
        return train_set, test_set

    def dense_to_one_hot(self, labels_dense, num_classes=5):
        # ont hot copy from https://github.com/nlintz/TensorFlow-Tutorials
        # also can use sklearn preprocessing OneHotEncoder()
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot


    def run(self):
        #train_set, valid_set, test_set = self.load_data()
        train_set, test_set = self.load_datas()
        #train_X, train_Y = train_set
        #test_X, test_Y = test_set
        train_X, train_Y=train_set[0:375], test_set[0:375]
        test_X, test_Y=train_set[375:500], test_set[375:500]
        train_Y = self.dense_to_one_hot(train_Y)
        test_Y = self.dense_to_one_hot(test_Y)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(500):
            for start, end in zip(range(0, len(train_X), 128), range(128, len(train_X), 128)):
                sess.run(self.train, feed_dict={self.X: train_X[start:end], self.Y: train_Y[start:end], self.keep_prob: 0.5})
            print('Iteration:',i,'rate:',
                  np.mean(np.argmax(test_Y, axis=1) ==
                          (sess.run(self.predict, feed_dict={self.X: test_X, self.Y: test_Y, self.keep_prob: 1}))),'%')
        sess.close()


if __name__ == '__main__':
    lr_model = LogisticRegression()
    lr_model.run()