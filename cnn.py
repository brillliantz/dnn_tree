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
DROPOUT_KEEP_PROB = 1.0
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


def calc_accu_tf(y, yhat):
    y = tf.argmax(y, axis=1)
    yhat = tf.argmax(yhat, axis=1)

    accuracy = tf.reduce_mean(tf.equal(y, yhat))
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


def calc_loss_tf(y, yhat, kind='cross_entropy'):
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


def calc_rsq_tf(y, yhat):
    residue = tf.subtract(y, yhat, name='residue')
    ss_residue = tf.square(residue)
    y_mean = tf.reduce_mean(y, name='y_mean')
    ss_total = tf.reduce_mean(tf.square(tf.subtract(y, y_mean)))
    res = tf.subtract(tf.constant(1.0), ss_residue / ss_total)
    return res


def opt(loss, optimizer, global_step_var=None):
    minimize_operation = optimizer.minimize(loss, global_step=global_step_var)
    return minimize_operation


#######################################################################
# Train and Predict

def run_train(sess,
              batch_size, n_epoch,
              fetches,
              merged_summary=None,
              writer_dir='',
              global_step_tensor=None,
              save_and_eval_interval=100,
              eval_ops=None,
              saver_dir='saved_model',
              ):
    """

    Parameters
    ----------
    sess : tf.Session
    batch_size : int
    n_epoch : int
    fetches : list
    merged_summary
    writer_dir : str, default ''
    global_step_tensor : Tensor
    saver_dir : str
    save_and_eval_interval : int
        How many steps to save and test current model.
    eval_ops : dict
        {str: tf.Tensor}

    Returns
    -------

    """

    # write event file (contains summaries)
    if merged_summary is not None and writer_dir:
        fetches.append(merged_summary)
        writer = tf.summary.FileWriter(writer_dir, sess.graph)

    if eval_ops is None:
        eval_ops = dict()

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)

    ckpt_fp = tf.train.latest_checkpoint(SAVE_DIR)
    if ckpt_fp is not None:
        saver.restore(sess, ckpt_fp)
    else:
        sess.run(tf.global_variables_initializer())

    # iter_per_epoch = train_len // batch_size
    # epoch_trained = tf.train.global_step(sess, global_step_tensor) // iter_per_epoch
    # tf.logging.info("Num. of iterations per epoch: {:d}".format(iter_per_epoch))
    pbar = tqdm(total=save_and_eval_interval, desc="Training epoch {:d}.".format(1))
    for epoch in range(1, n_epoch+1):
        # tf.logging.info("Start to train epoch {:d}.".format(epoch))
        while True:
            # feed data using iterator until one epoch ends (OutOfRangeError)
            try:
                pbar.update(1)

                res = sess.run(fetches=fetches)
                gs = tf.train.global_step(sess, global_step_tensor)

                if writer_dir:
                    if merged_summary is not None and writer_dir:
                        res_fetch, res_summary = res[: -1], res[-1]
                        writer.add_summary(res_summary,
                                           global_step=gs)
                        writer.flush()

                if saver is not None and gs % save_and_eval_interval == 0:
                    save_fp = os.path.join(saver_dir, 'saved_model')
                    saver.save(sess, save_fp, global_step=gs)
                    tf.logging.info("Global step = {:d}: model saved at {:s}".format(gs, save_fp))

                    for name, eval_op in eval_ops.items():
                        score = sess.run(fetches=eval_op)
                        tf.logging.info("{:s}: {:.5f}".format(name, score))

                    pbar.close()
                    pbar = tqdm(total=save_and_eval_interval, desc="Training epoch {:d}.".format(epoch))

            except tf.errors.OutOfRangeError:
                break


def run_predict_old(sess,
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


def run_predict(sess,
                x_ph, x_test,
                predict_op, extra_feed_dict=None):
    feeds = {x_ph: x_test}
    if extra_feed_dict is not None:
        feeds.update(extra_feed_dict)

    y_pred = sess.run(predict_op, feed_dict=feeds)
    return y_pred


def run_score(sess,
              x_ph, x_test, y_test,
              predict_op,
              score_func,
              extra_feed_dict=None):
    """

    Parameters
    ----------
    sess : tf.Session
    x_ph : tf.placeholder
    x_test : np.ndarray
        of shape [batch, ...]
    y_test : np.ndarray
        of shape [batch, ...]
    predict_op : tf.Tensor
    score_func : callable
        score_func takes (ytrue, yhat) as parameter and returns a float
    extra_feed_dict : dict. Optional, default None

    Returns
    -------
    score : float

    """
    y_pred = run_predict(sess,
                         x_ph=x_ph, x_test=x_test,
                         predict_op=predict_op,
                         extra_feed_dict=extra_feed_dict)
    score = score_func(y_test, y_pred)
    return score


#######################################################################
# Main functions


def build_model(x_input, input_channel, output_size):
    """

    Parameters
    ----------
    x_input : Tensor
    batch_size : int
    input_channel : int
    output_size : int
    p_keep_1 : Tensor
    p_keep_2 : Tensor

    Returns
    -------
    y_pred : Tensor

    """
    # hyper-parameters
    p_keep_1 = tf.Variable(tf.constant(DROPOUT_KEEP_PROB), name='p_keep_conv')
    p_keep_2 = tf.Variable(tf.constant(DROPOUT_KEEP_PROB), name='p_keep_hidden')

    # conv layers
    conv_net_output = cnn_1d(x_input, n_channel=input_channel, p_keep_conv=p_keep_1)
    # conv_net_output = cnn_2d(x_input, n_channel=n_channel, p_keep_conv=p_keep_1)

    # reshape
    shape = conv_net_output.get_shape().as_list()
    shape_without_batch = shape[1:]
    input_for_fc = tf.reshape(conv_net_output, shape=[-1, np.prod(shape_without_batch)])

    # fully connected layers
    fc1_output = full_conn(input_for_fc, output_size=512, act='relu', p_keep_hidden=p_keep_2)
    fc2_output = full_conn(fc1_output, output_size=output_size, act='softmax', p_keep_hidden=p_keep_2)

    y_pred = fc2_output
    return y_pred


def build_and_train(sess_config):
    # data
    from data_vendor import DataMNIST_new
    vendor = DataMNIST_new()
    ds_train, ds_test, input_shape_without_batch, n_classes = vendor.get_data()

    # hyper-parameters
    batch_size = 110
    input_shape = list(input_shape_without_batch).insert(0, None)
    output_shape = [None, n_classes]
    n_channel = 1

    # Dataset Iterator
    ds_train = ds_train.batch(batch_size)
    ds_test = ds_test.repeat().batch(10000)
    itr_train = ds_train.make_initializable_iterator()
    itr_test = ds_test.make_initializable_iterator()
    X, Y = itr_train.get_next()
    x_test, y_test = itr_test.get_next()

    # Create a Graph and set as default
    g = tf.get_default_graph()  # tf.Graph()

    # placeholders
    # X = tf.placeholder("float", shape=input_shape, name='X')
    # Y = tf.placeholder("float", shape=output_shape, name='Y')

    global_step = tf.Variable(0, name='global_step', trainable=False)
    g.add_to_collection('global_step', global_step)

    # model
    y_pred = build_model(X,
                         input_channel=n_channel, output_size=n_classes, )
    y_pred_test = build_model(x_test,
                              input_channel=n_channel, output_size=n_classes, )
    g.add_to_collection('predict_op', y_pred)

    train_loss_op = calc_loss_tf(Y, y_pred, kind='cross_entropy')
    test_loss_op = calc_loss_tf(y_test, y_pred_test, kind='cross_entropy')

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = opt(train_loss_op, optimizer, global_step_var=global_step)

    train_score_op = calc_accu_tf(Y, y_pred)
    test_score_op = calc_accu_tf(y_test, y_pred_test)

    # summary
    tf.summary.scalar('Train_Loss', train_loss_op)
    tf.summary.scalar('Test_Loss', test_loss_op)
    tf.summary.scalar('Train_Rsquared', train_score_op)
    tf.summary.scalar('Test_Rsquared', test_score_op)

    # try to restore
    sess = tf.Session(graph=g, config=sess_config)
    sess.run(itr_train.initializer)
    sess.run(itr_test.initializer)
    run_train(sess,
              batch_size=batch_size, n_epoch=100,
              fetches=[train_op],
              merged_summary=tf.summary.merge_all(),
              writer_dir=SAVE_DIR,
              global_step_tensor=global_step,
              save_and_eval_interval=100,
              eval_ops={'train rsq': train_score_op,
                        'test rsq' : test_score_op},
              saver_dir=SAVE_DIR,
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
                                  X, teX,
                                  predict_op=predict_op,)

        accu = calc_accu(teY, y_test_pred)
        tf.logging.info(accu)
        # rsq = calc_rsq(teY, y_test_pred)
        # print(rsq)
        y_test_pred = run_predict(sess,
                                  X, trX,
                                  predict_op=predict_op,
                                  )

        accu = calc_accu(trY, y_test_pred)
        tf.logging.info(accu)


def dataset_example():
    from sklearn.preprocessing import OneHotEncoder
    with np.load('Data/MNIST/mnist.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        x_train = x_train / 256.0
        x_test = x_test / 256.0
        y_train = y_train.reshape([-1, 1])
        y_test = y_test.reshape([-1, 1])

        encoder = OneHotEncoder(n_values=10)
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.fit_transform(y_test)

        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

    # ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = tf.data.Dataset.from_tensor_slices(x_train)
    ds_train = ds_train.map(lambda x: tf.cast(x, tf.float32))
    print(ds_train)

    batch_size = 500
    ds_train = ds_train.repeat(100)
    ds_train = ds_train.batch(batch_size)
    def _batch_normalization(tensor_in, epsilon=.0001):
        mean, variance = tf.nn.moments(tensor_in, axes=[0])
        print(mean)
        tensor_normalized = (tensor_in - mean) / (variance + epsilon)
        return tensor_normalized
    # ds_train = ds_train.map(_batch_normalization)

    iterator = ds_train.make_initializable_iterator()
    next_elem = iterator.get_next()
    x = next_elem

    # model
    # X = tf.placeholder("float", shape=(None, 28, 28), name='X')
    # Y = tf.placeholder("float", shape=(None, 10), name='Y')

    x1 = tf.reduce_sum(x, axis=1)
    w = tf.Variable(tf.random_normal(shape=[28, 10]))
    y = tf.matmul(x1, w)

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        while True:
            try:
                xv = sess.run(x)[0]
                res = sess.run(y)
                print(xv.min(), xv.max(), xv.shape)
                print(res.shape)
            except tf.errors.OutOfRangeError:
                break


if __name__ == "__main__":
    build_and_train(gpu_config.gpuconfig)
    # predict_and_calc(gpu_config.gpuconfig)
    # dataset_example()

