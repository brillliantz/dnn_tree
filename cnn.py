# encoding: utf-8


import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
import os

SAVE_DIR = 'simple_cnn_model'


def calc_rsq(y, yhat):
    # ret = 1 - (y-yhat).var() / y.var()
    ret = 1 - ((y - yhat) ** 2).mean() / y.var()
    return ret


def init_weights(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


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


def full_conn(input_, output_size, p_keep_hidden):
    """

    Parameters
    ----------
    input_ : Tensor
        shape of [batch_size, n]
    output_size
    p_keep_hidden

    Returns
    -------
    output : Tensor
        shape of (output_size, )

    """
    input_size = input_.get_shape()[-1].value
    w4 = init_weights([input_size, output_size], name='w4')

    l4 = tf.nn.relu(tf.matmul(input_, w4, name='FC_MatMul'), name='FC_act')

    l4_drop = tf.nn.dropout(l4, p_keep_hidden, name='FC_dropout')

    output = l4_drop
    return output


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
    conv_o_reshape = tf.reshape(conv_net_output, shape=[batch_size, -1])

    fc_output = full_conn(conv_o_reshape, output_size=output_size, p_keep_hidden=p_keep_2)

    y_pred = fc_output
    return y_pred


def build_and_fit():
    # data
    from demo_fully_diff_ndf import load_custom_data
    trX, teX, trY, teY, n_class, input_shape, output_shape, is_regression = load_custom_data()

    # hyper-parameters
    batch_size = input_shape[0]
    # width, height = 30, 1
    n_channel = 1
    output_size = 10

    # placeholders
    X = tf.placeholder("float", shape=input_shape, name='X')
    Y = tf.placeholder("float", shape=output_shape, name='Y')
    p_keep1 = tf.placeholder("float", name='p_keep_conv')
    p_keep2 = tf.placeholder("float", name='p_keep_hidden')

    # model
    y_pred = build_model(X,
                         batch_size=batch_size, n_channel=n_channel, output_size=output_size,
                         p_keep_1=p_keep1, p_keep_2=p_keep2)
    loss = calc_loss(Y, y_pred)
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = opt(loss, optimizer, global_step_var=global_step)

    # session
    sess = tf.Session(config=None)

    ckpt_fn = tf.train.latest_checkpoint(SAVE_DIR)
    if ckpt_fn is not None:
        meta_fn = ckpt_fn + '.meta'
        saver = tf.train.import_meta_graph(meta_fn)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, ckpt_fn)

        print("global step = ", tf.train.global_step(sess, global_step))

        y_test_pred = run_predict(sess,
                                  X,
                                  teX, batch_size=batch_size,
                                  predict_step=y_pred,
                                  extra_feed_dict={p_keep1 : 1.0,
                                                   p_keep2 : 1.0})

        ytrue = np.argmax(teY, axis=1)
        yhat = np.argmax(y_test_pred, axis=1)
        eq = ytrue == yhat
        accu = np.mean(eq)
        print(accu)
        # rsq = calc_rsq(teY, y_test_pred)
        # print(rsq)
    else:
        saver = tf.train.Saver(max_to_keep=4)

        run_train(sess,
                  X, Y,
                  trX, trY, batch_size=batch_size, n_epoch=2,
                  fetches=[train_step],
                  extra_feed_dict={p_keep2 : 0.8,
                                   p_keep1 : 0.8},
                  merged_summary=None,  # tf.summary.merge_all(),
                  writer_dir='simple_cnn_model',
                  global_step_tensor=global_step,
                  saver=saver,
                  saver_dir=SAVE_DIR
                  )


def calc_loss(y, yhat, kind='mse'):
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


def run_train(sess,
              x_ph, y_ph, x_train, y_train, batch_size, n_epoch,
              fetches, extra_feed_dict,
              merged_summary=None,
              writer_dir='',
              global_step_tensor=None,
              saver_dir='saved_model',
              saver=None):
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

    Returns
    -------

    """

    if merged_summary is not None and writer_dir:
        fetches.append(merged_summary)
        writer = tf.summary.FileWriter(writer_dir, sess.graph)

    sess.run(tf.global_variables_initializer())

    train_len = len(x_train)
    tf.logging.info("Num. of iterations per epoch: {:d}".format(train_len // batch_size))
    for epoch in range(1, n_epoch+1):
        for start, end in tqdm(zip(range(0         , train_len, batch_size),
                                   range(batch_size, train_len, batch_size))):
            feeds = {x_ph: x_train[start: end],
                     y_ph: y_train[start: end]}
            feeds.update(extra_feed_dict)

            res = sess.run(fetches=fetches, feed_dict=feeds)

            gs = tf.train.global_step(sess, global_step_tensor)
            if saver is not None and gs % 100 == 0:
                saver.save(sess, os.path.join(saver_dir, 'saved_model'),
                           global_step=global_step_tensor)

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


def predict_and_calc():
    pass

if __name__ == "__main__":
    build_and_fit()
    predict_and_calc()

