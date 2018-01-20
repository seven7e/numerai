#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import data_helper
import util

tour = 89

summaries_dir = 'tour{}_logs'.format(tour)
model_path =  "tour{}_model/save_net.ckpt".format(tour)
batch_size = 100

#TODO: make it params
n_hidden1 = 150
n_hidden2 = 50
class NNModel(object):
    """ builds the graph for a deep net for classifying digits.
    """

    def __init__(self, n_input, n_out, n_hiddens=[50], thr=0.5):
        self.n_input = n_input
        self.n_out = n_out
        self.n_hiddens = n_hiddens
        self.thr = thr
        assert(n_out == 2)
        self.build_placeholder()
        self.build_nn()
        self.build_object()

    def build_nn(self):
        # Create the model
        last_layer_out_size = self.n_input
        last_layer_out = self.x
        self._hidden_layers = []
        for i, hsize in enumerate(self.n_hiddens):
            s = str(i + 1)
            print('creating hidden layer {} with size {}'.format(s, hsize))
            with tf.name_scope('hidden_layer_' + s):
                W1 = tf.Variable(tf.random_normal([last_layer_out_size, hsize]), name='W' + s)
                b1 = tf.Variable(tf.random_normal([hsize]), name='b' + s)
                tmp = tf.matmul(last_layer_out, W1) + b1
                # h1 = tf.nn.relu(tmp)
                h1 = tf.nn.sigmoid(tmp)
                # h1 = tf.nn.tanh(tmp)
                self._hidden_layers.append((h1, W1, b1))
                last_layer_out = h1
                last_layer_out_size = hsize

        # with tf.name_scope('hidden_layer_2'):
        #     W2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]), name='W2')
        #     b2 = tf.Variable(tf.random_normal([n_hidden2]), name='b2')
        #     tmp = tf.matmul(h1, W2) + b2
        #     h2 = tf.nn.relu(tmp)

        with tf.name_scope('output_layer'):
            W3 = tf.Variable(tf.random_normal([last_layer_out_size, self.n_out]), name='Wout')
            b3 = tf.Variable(tf.random_normal([self.n_out]), name='bout')
            y_out = tf.matmul(last_layer_out, W3) + b3

        self.y_out = y_out
        # self.y_out = tf.reshape(y_out, [-1])
        # self.keep_prob = keep_prob

    def build_placeholder(self):
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, self.n_input], name='x')

        with tf.name_scope('label'):
            self.y = tf.placeholder(tf.float32, [None, self.n_out], name='y')
            # self.y = tf.placeholder(tf.float32, [None], name='y')

        with tf.name_scope('params'):
            self.thr_sym = tf.constant(self.thr, name='threshold')

    def build_object(self):
        with tf.name_scope('loss'):
            self.cross_entropy = tf.reduce_mean(
                    # tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.y, logits=self.y_out))

        with tf.name_scope('accuracy'):
            self.correct_prediction = tf.equal(tf.argmax(self.y_out, 1),
                    tf.argmax(self.y, 1))
            # self.correct_prediction = tf.equal(
            #         tf.to_int32(tf.greater_equal(self.y_out, self.thr)),
            #         self.y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def build_trainer(self):
        with tf.name_scope('train'):
            # opt = tf.train.GradientDescentOptimizer(0.5)
            opt = tf.train.AdadeltaOptimizer()
            # opt = tf.train.AdagradOptimizer()
            # opt = tf.train.AdagradDAOptimizer()
            # opt = tf.train.MomentumOptimizer()
            # opt = tf.train.AdamOptimizer()
            # opt = tf.train.FtrlOptimizer()
            # opt = tf.train.ProximalGradientDescentOptimizer()
            # opt = tf.train.ProximalAdagradOptimizer()
            # opt = tf.train.RMSPropOptimizer(0.5)
            self.train_step = opt.minimize(self.cross_entropy)

    def build_summarier(self, sess):
        tf.summary.scalar('cross entropy', self.cross_entropy)
        tf.summary.scalar('accuracy', self.accuracy)

        self.merged_summary_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
        self.test_writer = tf.summary.FileWriter(summaries_dir + '/test')

def make_feed(nn, x, y, keep_prob=None):
    return {nn.x :x, nn.y :y}  #, nn.keep_prob :keep_prob}

def train_model(nn, X, y, batch_size=10, batch_count=1000):

    # y = tf.one_hot(y, 2)
    # print('one hot y:', y)
    print('trainning data shape: X {}, y {}'.format(X.shape, y.shape))

    X_train, X_vali, y_train, y_vali = train_test_split(
        X, y, test_size=0.25) #, random_state=42)

    print('training/validation: {:d}/{:d}'.format(X_train.shape[0], X_vali.shape[0]))

    batcher = util.Batcher(X_train, y_train)

    nn.build_trainer()

    # save model for latter usage
    saver = tf.train.Saver()

    print('start training on batch size {}'.format(batch_size))
    # start a session
    with tf.Session() as sess:
        nn.build_summarier(sess)

        tf.global_variables_initializer().run()

        # Train
        for i in range(batch_count):
            batch_xs, batch_ys = batcher.next_batch(batch_size)
            # print(batch_xs, batch_ys)
            if i % 100 == 0:
                # if False:

                train_accuracy, train_logloss = sess.run(
                    [nn.accuracy, nn.cross_entropy],
                    feed_dict=make_feed(nn, batch_xs, batch_ys, 0.5))

                eval_accuracy, eval_logloss, summary_str = sess.run(
                    [nn.accuracy, nn.cross_entropy, nn.merged_summary_op],
                    feed_dict=make_feed(nn, X_vali, y_vali, 1.0))
                print('step {}, training cross entropy {:.6f}, accuracy {:.6f};'
                    ' validation cross entropy {:.6f}, accuracy {:.6f}' \
                    .format(i, train_logloss, train_accuracy, eval_logloss, eval_accuracy))
                nn.test_writer.add_summary(summary_str, i)

                if i > 0:
                    save_path = saver.save(sess, model_path)
                    print("Save to path: ", save_path)

            _, summary_str = sess.run([nn.train_step, nn.merged_summary_op],
                    feed_dict=make_feed(nn, batch_xs, batch_ys, 0.5))
            nn.train_writer.add_summary(summary_str, i)

# def test(nn):
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         saver.restore(sess, model_path)
#         print('model is restored from %s' % model_path)

#         x = mnist.test.images
#         y = mnist.test.labels
#         nb_batches = len(y) // batch_size + 1
#         accuracies = []
#         for i in range(nb_batches):
#             batch_xs, batch_ys = mnist.test.next_batch(batch_size)
#             # # Test trained model
#             accuracies.append(sess.run(nn.accuracy, feed_dict=
#                    make_feed(nn, batch_xs, batch_ys, 1.0)))
#         print('testing accuracy: %g' % np.mean(accuracies))

def train(nn):

    nrows = 30
    nrows = None
    training_data = data_helper.load_training_data(tour, nrows=nrows)
    # training_data = data_helper.get_random_data(nrows=5, nfeat=3)
    # training_data = data_helper.get_random_data()

    frac = None
    #frac = 0.1
    #frac = 0.01
    #frac = 0.0001
    if frac is not None:
        training_data = training_data.sample(frac=frac)

    X, y = data_helper.get_Xy(training_data, onehot=True)

    batch_count = 50000
    batch_size = 5000
    train_model(nn, X, y, batch_size=batch_size, batch_count=batch_count)

def main():

    n_input = 50
    n_out = 2
    # n_hiddens = []
    # n_hiddens = [100]
    # n_hiddens = [150, 50]
    # n_hiddens = [300, 100]
    # n_hiddens = [500, 100]
    # n_hiddens = [300, 300, 100]
    n_hiddens = [40, 20, 10]
    # n_hiddens = [1000, 1000, 1000, 500, 100]

    try:
        mode = sys.argv[1]
    except IndexError:
        mode = 'train'

    nn = NNModel(n_input, n_out, n_hiddens)

    if mode == 'train':
        # train(nn)
        train(nn)
    elif mode == 'test':
        test(nn)
    elif mode == 'plot':
        plot(nn)


if __name__ == '__main__':
    main()
