# encoding: utf-8


import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
import os

import gpu_config

tf.logging.set_verbosity(tf.logging.INFO)

LEARNING_RATE = 1E-4
SAVE_DIR = 'simple_cnn_model_1d'
# SAVE_DIR = 'simple_cnn_model_2d'


def calc_rsq(y, yhat):
    # ret = 1 - (y-yhat).var() / y.var()
    ret = 1 - ((y - yhat) ** 2).mean() / y.var()
    return ret


def calc_accu(y, yhat):
    """

    Parameters
    ----------
    y : np.ndarray
        shape of [n_samples, n_classes]
    yhat : np.ndarray
        shape of [n_samples, n_classes]

    Returns
    -------
    accuracy : float

    """
    assert len(y) == len(yhat)

    y = np.argmax(y, axis=1)
    yhat = np.argmax(yhat, axis=1)

    accuracy = np.mean(y == yhat)
    return accuracy



def init_weights(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

#######################################################################
# Various Layers


def cnn_1d(input_, n_channel, p_keep_conv):
    """

    Parameters
    ----------
    input_ : Tensor
        shape of [batch_size, width (>=1), height (=1), n_channel (n_feature)]
    n_channel : int
    p_keep_conv : float Tensor

    Returns
    -------
    output : Tensor
        shape of [batch_size, width (smaller), height (smaller), n_channel (larger)]

    """
    with tf.name_scope("CNN/"):
        with tf.name_scope('CNN/layer-1/'):
            # [filter_height, filter_width, in_channels, out_channels]
            w1 = init_weights([3, 1, n_channel, 32], name='w1')
            l1a = tf.nn.relu(tf.nn.conv2d(input_,
                                          filter=w1, strides=[1, 1, 1, 1], padding='SAME',
                                          name='conv1'),
                             name='l1a')
            l1b = tf.nn.max_pool(l1a,
                                 ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME',
                                 name='l1_')
            l1 = tf.nn.dropout(l1b, p_keep_conv, name='l1')

        with tf.name_scope('CNN/layer-2/'):
            w2 = init_weights([3, 1, 32, 64], name='w2')
            l2a = tf.nn.relu(tf.nn.conv2d(l1,
                                          filter=w2, strides=[1, 1, 1, 1], padding='SAME',
                                          name='conv2'),
                             name='l2a')
            l2b = tf.nn.max_pool(l2a,
                                 ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME',
                                 name='l2_')
            l2 = tf.nn.dropout(l2b, p_keep_conv, name='l2')

        with tf.name_scope('CNN/layer-3/'):
            w3 = init_weights([3, 1, 64, 128], name='w3')
            l3a = tf.nn.relu(tf.nn.conv2d(l2,
                                          filter=w3, strides=[1, 1, 1, 1], padding='SAME',
                                          name='conv3'),
                             name='l3a')
            l3b = tf.nn.max_pool(l3a,
                                 ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME',
                                 name='l3_')

            l3 = tf.nn.dropout(l3b, p_keep_conv, name='l3')
            #l3c = tf.reshape(l3b, [batch_size, -1], name='l3_reshape')

    output = l3
    return output


def cnn_2d(input_, n_channel, p_keep_conv):
    """

    Parameters
    ----------
    input_ : Tensor
        shape of [batch_size, width (>=1), height (=1), n_channel (n_feature)]
    n_channel : int
    p_keep_conv : float Tensor

    Returns
    -------
    output : Tensor
        shape of [batch_size, width (smaller), height (smaller), n_channel (larger)]

    """
    with tf.name_scope("CNN/"):
        with tf.name_scope('CNN/layer-1/'):
            # [filter_height, filter_width, in_channels, out_channels]
            w1 = init_weights([3, 3, n_channel, 32], name='w1')
            l1a = tf.nn.relu(tf.nn.conv2d(input_,
                                          filter=w1, strides=[1, 1, 1, 1], padding='SAME',
                                          name='conv1'),
                             name='l1a')
            l1b = tf.nn.max_pool(l1a,
                                 ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                 name='l1_')
            l1 = tf.nn.dropout(l1b, p_keep_conv, name='l1')

        with tf.name_scope('CNN/layer-2/'):
            w2 = init_weights([3, 3, 32, 64], name='w2')
            l2a = tf.nn.relu(tf.nn.conv2d(l1,
                                          filter=w2, strides=[1, 1, 1, 1], padding='SAME',
                                          name='conv2'),
                             name='l2a')
            l2b = tf.nn.max_pool(l2a,
                                 ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                 name='l2_')
            l2 = tf.nn.dropout(l2b, p_keep_conv, name='l2')

        with tf.name_scope('CNN/layer-3/'):
            w3 = init_weights([3, 3, 64, 128], name='w3')
            l3a = tf.nn.relu(tf.nn.conv2d(l2,
                                          filter=w3, strides=[1, 1, 1, 1], padding='SAME',
                                          name='conv3'),
                             name='l3a')
            l3b = tf.nn.max_pool(l3a,
                                 ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                 name='l3_')

            l3 = tf.nn.dropout(l3b, p_keep_conv, name='l3')
            #l3c = tf.reshape(l3b, [batch_size, -1], name='l3_reshape')

    output = l3
    return output


def full_conn(input_, output_size, p_keep_hidden, act='relu'):
    """

    Parameters
    ----------
    input_ : Tensor
        shape of [batch_size, n]
    output_size
    p_keep_hidden
    act : str
        Name of activation function.

    Returns
    -------
    output : Tensor
        shape of (output_size, )

    """
    with tf.name_scope('FullyConnected'):
        input_size = input_.get_shape()[-1].value
        w4 = init_weights([input_size, output_size], name='w4')

        matmul_res = tf.matmul(input_, w4, name='FC_MatMul')
        if act == 'relu':
            act_func = tf.nn.relu
        elif act == 'softmax':
            act_func = tf.nn.softmax
        elif act == 'sigmoid':
            act_func = tf.nn.sigmoid
        else:
            act_func = tf.nn.relu
        l4 = act_func(matmul_res, name='FC_act')

        l4_drop = tf.nn.dropout(l4, p_keep_hidden, name='FC_dropout')

    output = l4_drop
    return output


#######################################################################
# Loss and Optimization


def calc_loss(y, yhat, kind='cross_entropy'):
    if kind == 'mse':
        loss = tf.reduce_sum(
            tf.losses.mean_squared_error(y, yhat),
            name="MSE"
        )
    elif kind == 'cross_entropy':
        # cross entropy loss
        loss = tf.reduce_mean(-tf.multiply(tf.log(yhat), y), name='cost')
    else:
        raise NotImplemented

    return loss


def opt(loss, optimizer, global_step_var=None):
    minimize_operation = optimizer.minimize(loss, global_step=global_step_var)
    return minimize_operation


#######################################################################
# Train and Predict

def run_train(sess,
              x_ph, y_ph, x_train, y_train, batch_size, n_epoch,
              fetches, extra_feed_dict,
              merged_summary=None,
              writer_dir='',
              global_step_tensor=None,
              saver_dir='saved_model',
              # saver=None
              save_and_test_interval=100,
              test_func=None,
              ):
    """

    Parameters
    ----------
    sess : tf.Session
    x_ph : placeholder
    y_ph : placeholder
    x_train
    y_train
    batch_size : int
    n_epoch : int
    fetches : list
    extra_feed_dict : dict
    merged_summary
    writer_dir : str, default ''
    global_step_tensor : Tensor
    saver_dir : str
    save_and_test_interval : int
        How many steps to save and test current model.
    test_func : callable, default None
        test function to run.

    Returns
    -------

    """

    # write event file (contains summaries)
    if merged_summary is not None and writer_dir:
        fetches.append(merged_summary)
        writer = tf.summary.FileWriter(writer_dir, sess.graph)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)

    ckpt_fp = tf.train.latest_checkpoint(SAVE_DIR)
    if ckpt_fp is not None:
        saver.restore(sess, ckpt_fp)
    else:
        sess.run(tf.global_variables_initializer())

    train_len = len(x_train)
    iter_per_epoch = train_len // batch_size
    epoch_trained = tf.train.global_step(sess, global_step_tensor) // iter_per_epoch
    tf.logging.info("Num. of iterations per epoch: {:d}".format(iter_per_epoch))
    for epoch in range(epoch_trained+1, n_epoch+1):
        for start, end in tqdm(zip(range(0         , train_len, batch_size),
                                   range(batch_size, train_len, batch_size))):
            feeds = {x_ph: x_train[start: end],
                     y_ph: y_train[start: end]}
            feeds.update(extra_feed_dict)

            res = sess.run(fetches=fetches, feed_dict=feeds)

            gs = tf.train.global_step(sess, global_step_tensor)
            if saver is not None and gs % save_and_test_interval == 0:
                save_fp = os.path.join(saver_dir, 'saved_model')
                saver.save(sess, save_fp, global_step=gs)
                tf.logging.info("Global step = {:d}: model saved at {:s}".format(gs, save_fp))

                if test_func is not None:
                    test_func()

            if writer_dir:
                if merged_summary is not None and writer_dir:
                    res_fetch, res_summary = res[: -1], res[-1]
                    writer.add_summary(res_summary,
                                       global_step=gs)
                    writer.flush()


def run_predict(sess,
                x_ph, x_test, batch_size,
                predict_step, extra_feed_dict=None):
    results = []
    test_len = len(x_test)
    start, end = 0, 0
    for start, end in zip(range(0         , test_len, batch_size),
                          range(batch_size, test_len, batch_size)):
        feeds = {x_ph: x_test[start: end]}
        if extra_feed_dict is not None:
            feeds.update(extra_feed_dict)

        y_pred = sess.run(predict_step, feed_dict=feeds)
        results.append(y_pred)

    if (test_len - batch_size) <= end < test_len:
        real_start_idx = end - (test_len - batch_size)
        start, end = test_len - batch_size, test_len

        feeds = {x_ph: x_test[start: end]}
        if extra_feed_dict is not None:
            feeds.update(extra_feed_dict)

        y_pred = sess.run(predict_step, feed_dict=feeds)
        y_pred = y_pred[real_start_idx: ]
        results.append(y_pred)

    y_pred = np.concatenate(results, axis=0)
    return y_pred


def run_score(sess,
              x_ph, x_test, batch_size,
              predict_step, extra_feed_dict=None):
    pass


#######################################################################
# Main functions


def build_model(x_input, batch_size, n_channel, output_size, p_keep_1, p_keep_2):
    """

    Parameters
    ----------
    x_input : Tensor
    batch_size : int
    n_channel : int
    output_size : int
    p_keep_1 : Tensor
    p_keep_2 : Tensor

    Returns
    -------
    y_pred : Tensor

    """
    conv_net_output = cnn_1d(x_input, n_channel=n_channel, p_keep_conv=p_keep_1)
    # conv_net_output = cnn_2d(x_input, n_channel=n_channel, p_keep_conv=p_keep_1)
    conv_o_reshape = tf.reshape(conv_net_output, shape=[batch_size, -1])

    fc1_output = full_conn(conv_o_reshape, output_size=512, act='relu', p_keep_hidden=p_keep_2)
    fc2_output = full_conn(fc1_output, output_size=output_size, act='softmax', p_keep_hidden=p_keep_2)

    y_pred = fc2_output
    return y_pred


def build_and_train(sess_config):
    # data
    from demo_fully_diff_ndf import load_custom_data
    trX, teX, trY, teY, n_class, input_shape, output_shape, is_regression = load_custom_data()
    # hyper-parameters
    batch_size = input_shape[0]
    # width, height = 30, 1
    n_channel = 1
    output_size = 10

    # Create a Graph and set as default
    g = tf.get_default_graph()  # tf.Graph()

    # placeholders
    X = tf.placeholder("float", shape=input_shape, name='X')
    Y = tf.placeholder("float", shape=output_shape, name='Y')
    p_keep1 = tf.placeholder("float", name='p_keep_conv')
    p_keep2 = tf.placeholder("float", name='p_keep_hidden')

    # model
    y_pred = build_model(X,
                         batch_size=batch_size, n_channel=n_channel, output_size=output_size,
                         p_keep_1=p_keep1, p_keep_2=p_keep2)
    g.add_to_collection('predict_op', y_pred)
    loss = calc_loss(Y, y_pred, kind='cross_entropy')
    tf.summary.scalar('loss_cross_entropy', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    g.add_to_collection('global_step', global_step)
    train_step = opt(loss, optimizer, global_step_var=global_step)

    def test_():
        y_test_pred = run_predict(sess,
                                  X,
                                  teX, batch_size=batch_size,
                                  predict_step=y_pred,
                                  extra_feed_dict={p_keep1 : 1.0,
                                                   p_keep2 : 1.0})
        accu = calc_accu(teY, y_test_pred)
        tf.logging.info(accu)

    # try to restore
    sess = tf.Session(graph=g, config=sess_config)
    run_train(sess,
              X, Y,
              trX, trY, batch_size=batch_size, n_epoch=100,
              fetches=[train_step],
              extra_feed_dict={p_keep2: 1.0,
                               p_keep1: 1.0},
              merged_summary=tf.summary.merge_all(),
              writer_dir=SAVE_DIR,
              global_step_tensor=global_step,
              save_and_test_interval=1000,
              # saver=saver,
              saver_dir=SAVE_DIR,
              test_func=test_
              )


def predict_and_calc(sess_config=None):
    # data
    from demo_fully_diff_ndf import load_custom_data
    trX, teX, trY, teY, n_class, input_shape, output_shape, is_regression = load_custom_data()
    # hyper-parameters
    batch_size = input_shape[0]

    # reset default graph to avoid potential errors
    tf.reset_default_graph()

    ckpt_fp = tf.train.latest_checkpoint(SAVE_DIR)
    meta_fp = '{}.meta'.format(ckpt_fp)

    with tf.Session(config=sess_config) as sess:
        new_saver = tf.train.import_meta_graph(meta_fp)
        new_saver.restore(sess, ckpt_fp)

        g = sess.graph

        # restore global step, predict_op and placeholders
        global_step = g.get_collection_ref('global_step')[0]
        predict_op = g.get_collection_ref('predict_op')[0]
        X = g.get_tensor_by_name('X:0')
        p_keep1 = g.get_tensor_by_name('p_keep_conv:0')
        p_keep2 = g.get_tensor_by_name('p_keep_hidden:0')

        tf.logging.info("global step = {}".format(tf.train.global_step(sess, global_step)))

        y_test_pred = run_predict(sess,
                                  X,
                                  teX, batch_size=batch_size,
                                  predict_step=predict_op,
                                  extra_feed_dict={p_keep1 : 1.0,
                                                   p_keep2 : 1.0})

        accu = calc_accu(teY, y_test_pred)
        tf.logging.info(accu)
        # rsq = calc_rsq(teY, y_test_pred)
        # print(rsq)
        y_test_pred = run_predict(sess,
                                  X,
                                  trX, batch_size=batch_size,
                                  predict_step=predict_op,
                                  extra_feed_dict={p_keep1 : 1.0,
                                                   p_keep2 : 1.0})

        accu = calc_accu(trY, y_test_pred)
        tf.logging.info(accu)


if __name__ == "__main__":
    build_and_train(gpu_config.gpuconfig)
    predict_and_calc(gpu_config.gpuconfig)

