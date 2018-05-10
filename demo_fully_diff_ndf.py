#!/usr/bin/python
"""
# Fully Differentiable Deep Neural Decision Forest

[![DOI](https://zenodo.org/badge/20267/chrischoy/fully-differentiable-deep-ndf-tf.svg)](https://zenodo.org/badge/latestdoi/20267/chrischoy/fully-differentiable-deep-ndf-tf)

This repository contains a simple modification of the deep-neural decision
forest [Kontschieder et al.] in TensorFlow. The modification allows joint
optimization of the decision nodes and leaf nodes which theoretically should speed up the training
(haven't verified).


## Motivation:

Deep Neural Deicision Forest, ICCV 2015, proposed an interesting way to incorporate a decision forest into a neural network.

The authors proposed incorporating the terminal nodes of a decision forest as static probability distributions and routing probabilities using sigmoid functions. The final loss is defined as the usual cross entropy between ground truth and weighted average of the terminal probabilities (weights being the routing probabilities).

As there are two trainable parameters, the authors used alternating optimization. They first fixed the terminal node probabilities and trained the base network (routing probabilities), then, fixed the network and optimized the terminal nodes. Such alternating optimization is usually slower than joint optimization since variables that are not being optimized slow down the optimization of the other variable.

However, if we parametrize the terminal nodes using a parametric probability distribution, we can jointly train both terminal and decision nodes, and theoretically, can speed up the convergence.

This code is just a proof-of-concept that

1. One can train both decision nodes and leaf nodes $\pi$ jointly using parametric formulation of leaf (terminal) nodes.

2. It is easy to implement such idea in a symbolic math library.


## Formulation

The leaf node probability $p \in \Delta^{n-1}$ can be parametrized using an $n$ dimensional vector $w_{leaf}$ $\exists w_{leaf}$ s.t. $p = softmax(w_{leaf})$. Thus, we can compute the gradient of $L$ w.r.t $w_{leaf}$ as well and can jointly optimize the terminal nodes as well.

## Experiment

I used a simple (3 convolution + 2 fc) network for this experiment. On the MNIST, it reaches 99.1% after 10 epochs.

## Slides

[SDL Reading Group Slides](https://docs.google.com/presentation/d/1Ze7BAiWbMPyF0ax36D-aK00VfaGMGvvgD_XuANQW1gU/edit?usp=sharing)


## Reference

[Kontschieder et al.] Deep Neural Decision Forests, ICCV 2015

The following is the expected output: the number of epoch and corresponding test accuracy.
```
0 0.955829326923
1 0.979166666667
2 0.982572115385
3 0.988080929487
4 0.988181089744
5 0.988481570513
6 0.987980769231
7 0.989583333333
8 0.991185897436
9 0.991586538462
10 0.991987179487
11 0.992888621795
12 0.993088942308
13 0.992988782051
14 0.992988782051
15 0.992588141026
16 0.993289262821
17 0.99358974359
18 0.992387820513
19 0.993790064103
20 0.994090544872
21 0.993289262821
22 0.993489583333
23 0.99358974359
24 0.993990384615
25 0.993689903846
26 0.99469150641
27 0.994491185897
28 0.994090544872
29 0.994090544872
30 0.99469150641
31 0.994090544872
32 0.994791666667
33 0.993790064103
34 0.994190705128
35 0.994591346154
36 0.993990384615
37 0.995092147436
38 0.994391025641
39 0.993389423077
40 0.994991987179
41 0.994991987179
42 0.994991987179
43 0.994491185897
44 0.995192307692
45 0.995192307692
46 0.994791666667
47 0.995092147436
48 0.994991987179
49 0.994290865385
50 0.994591346154
51 0.994791666667
52 0.995092147436
53 0.995492788462
54 0.994591346154
55 0.995092147436
56 0.994190705128
57 0.99469150641
58 0.99469150641
59 0.994090544872
60 0.994290865385
61 0.994891826923
62 0.994791666667
63 0.994491185897
64 0.994591346154
65 0.994290865385
66 0.99469150641
67 0.994391025641
68 0.994791666667
69 0.99469150641
70 0.994791666667
71 0.994591346154
72 0.994891826923
73 0.994791666667
74 0.995192307692
75 0.995392628205
76 0.995392628205
77 0.995292467949
78 0.994791666667
79 0.995092147436
80 0.995392628205
81 0.994891826923
82 0.995092147436
83 0.994891826923
84 0.995092147436
85 0.995092147436
86 0.995292467949
87 0.994891826923
88 0.995693108974
89 0.994391025641
90 0.994591346154
91 0.995592948718
92 0.995292467949
93 0.995192307692
94 0.994791666667
95 0.995192307692
96 0.995092147436
97 0.994591346154
98 0.995292467949
99 0.995392628205
```

## Slides

[SDL Reading Group Slides](https://docs.google.com/presentation/d/1Ze7BAiWbMPyF0ax36D-aK00VfaGMGvvgD_XuANQW1gU/edit?usp=sharing)

## References
[Kontschieder et al.] Deep Neural Decision Forests, ICCV 2015


## License

The MIT License (MIT)

Copyright (c) 2016 Christopher B. Choy (chrischoy@ai.stanford.edu)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Leaf Node: random_uniform -> soft_max -> *mu -> py_x_tree -> py_x_e -> py_x -> cross_entrophy

random_uniform weights: [n_leaf, n_class]                               range [-x, x]
nn.softmax: [n_leaf, n_class]                                           range [0, 1]
add batch dimension: [n_batch, n_leaf, n_class]                         range [0, 1]
leaf probability: [n_batch, n_leaf, n_class]                            range [0, 1]
routing probability: [n_batch, n_leaf, n_class]                         range [0, 1]
prob = routing_prob * leaf_prob = [n_batch, n_leaf, n_class]            range [0, 1]
P(Y | X) = reduce_mean(prob) = [n_batch, n_class]                       range [0, 1]

For each sample in a batch, P(Y | X) are probabilities of each class.

"""

# use limited GPU resources
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import tensorflow as tf
gpuconfig = tf.ConfigProto()
gpuconfig.gpu_options.allow_growth = True

import math
import numpy as np
import pandas as pd
import sklearn

DEPTH   = 3                 # Depth of a tree
N_LEAF  = 2 ** (DEPTH + 1)  # Number of leaf node
# N_LABEL = 10                # Number of classes
N_TREE  = 2                 # Number of trees (ensemble)
N_BATCH = 77               # Number of data points per mini-batch

FC_output_dim = 625


def variable_summaries(var, name=''):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('sum' + name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def define_ndf(upper_model_choice='dnn', regression=False):
    def init_weights(shape, name=None):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    def init_prob_weights(shape, minval=-5, maxval=5, name=None):
        return tf.Variable(tf.random_uniform(shape, minval, maxval), name=name)

    def get_tree_name(n):
        return "TREE-{:d}".format(n)

    def upper_model_dnn(x, n_features, layer_sizes=(50, 50, 50)):
        """Only support 1-d data (no image). """
        def get_stddev(in_dim, out_dim):
            return 1.3 / math.sqrt(float(in_dim) + float(out_dim))

        hidden = []

        with tf.name_scope("CNN/"):
            # Input Layer
            with tf.name_scope("input-layer"):
                weights = tf.Variable(tf.truncated_normal([n_features, layer_sizes[0]],
                                                          stddev=get_stddev(n_features, layer_sizes[0])),
                                      name='weights')
                biases = tf.Variable(tf.zeros([layer_sizes[0]]), name='biases')
                input = tf.matmul(x, weights) + biases

            # Hidden Layers
            for index, num_hidden in enumerate(layer_sizes):
                if index == len(layer_sizes) - 1: break
                with tf.name_scope("hidden{}".format(index + 1)):
                    weights = tf.Variable(tf.truncated_normal([num_hidden, layer_sizes[index + 1]],
                                                              stddev=get_stddev(num_hidden,
                                                                                layer_sizes[index + 1])),
                                          name='weights')
                    biases = tf.Variable(tf.zeros([layer_sizes[index + 1]]), name='biases')
                    inputs = input if index == 0 else hidden[index - 1]
                    hidden.append(tf.nn.relu(tf.matmul(inputs, weights) + biases,
                                             name="hidden{}".format(index + 1)))

        final_layer = hidden[-1]
        return final_layer, layer_sizes[-1]

    def upper_model_cnn(X, w1, w2, w3, p_keep_conv):
        with tf.name_scope("CNN/"):
            with tf.name_scope('CNN/layer-1/'):
                l1a = tf.nn.relu(tf.nn.conv2d(X, w1, [1, 1, 1, 1], 'SAME'), name='l1a')
                l1b = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1], padding='SAME', name='l1_')
                l1 = tf.nn.dropout(l1b, p_keep_conv, name='l1')

            with tf.name_scope('CNN/layer-2/'):
                l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'), name='l2a')
                l2b = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1], padding='SAME', name='l2_')
                l2 = tf.nn.dropout(l2b, p_keep_conv, name='l2')

            with tf.name_scope('CNN/layer-3/'):
                l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'), name='l3a')
                l3b = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1], padding='SAME', name='l3_')

                l3c = tf.reshape(l3b, [N_BATCH, -1], name='l3_reshape')
                l3 = tf.nn.dropout(l3c, p_keep_conv, name='l3')

        final_layer = l3
        return final_layer, 4 * 4 * 128

    def model(upper_layer, w4_e, w_d_e, w_l_e, p_keep_hidden):
        """
        Create a forest and return the neural decision forest outputs:

            decision_p_e: decision node routing probability for all ensemble
                If we number all nodes in the tree sequentially from top to bottom,
                left to right, decision_p contains
                [d(0), d(1), d(2), ..., d(2^n - 2)] where d(1) is the probability
                of going left at the root node, d(2) is that of the left child of
                the root node.

                decision_p_e is the concatenation of all tree decision_p's

            leaf_p_e: terminal node probability distributions for all ensemble. The
                indexing is the same as that of decision_p_e.
            """
        assert(len(w4_e) == len(w_d_e))
        assert(len(w4_e) == len(w_l_e))

        decision_p_e = []
        leaf_p_e = []
        count = 0
        for w4, w_d, w_l in zip(w4_e, w_d_e, w_l_e):
            #with tf.name_scope(get_tree_name(count) + '/' + 'FullyConnected/'):
            with tf.name_scope('FullyConnected-{:d}/'.format(count)):
                # n_loops = N_TREES
                l4 = tf.nn.relu(tf.matmul(upper_layer, w4, name='FC_MatMul'), name='FC_act')
                l4 = tf.nn.dropout(l4, p_keep_hidden, name='FC_dropout')
            with tf.name_scope(get_tree_name(count)+'/'):
                # d_n (x) = \sigma ( f_n (x) )
                decision_p = tf.nn.sigmoid(tf.matmul(l4, w_d), name='DecisionNode_{:d}'.format(count))
                leaf_p = tf.nn.softmax(w_l, name='LeafNode_{:d}'.format(count))

                decision_p_e.append(decision_p)
                leaf_p_e.append(leaf_p)

                count += 1

        return decision_p_e, leaf_p_e

    ##################################################
    # Input Output shape and placeholder
    ##################################################
    # Input X, output Y
    X = tf.placeholder("float", shape=input_shape, name='X')
    Y = tf.placeholder("float", shape=output_shape, name='Y')

    ##################################################
    # Initialize network weights
    ##################################################
    p_keep_conv = tf.placeholder("float", name='p_keep_conv')
    p_keep_hidden = tf.placeholder("float", name='p_keep_hidden')

    with tf.name_scope("CNN"):
        with tf.name_scope("CNN/layer-1/"):
            # [filter_height, filter_width, in_channels, out_channels]
            w1 = init_weights([3, 3, 1, 32], name='w1')
        with tf.name_scope("CNN/layer-2/"):
            w2 = init_weights([3, 3, 32, 64], name='w2')
        with tf.name_scope("CNN/layer-3/"):
            w3 = init_weights([3, 3, 64, 128], name='w3')

    if upper_model_choice == 'cnn':
        final_layer_of_upper_model, final_layer_size = upper_model_cnn(X, w1, w2, w3, p_keep_conv)
    elif upper_model_choice == 'dnn':
        final_layer_of_upper_model, final_layer_size = upper_model_dnn(X, input_shape[1], layer_sizes=[50] * 3)
    else:
        final_layer_of_upper_model, final_layer_size = upper_model_cnn(X, w1, w2, w3, p_keep_conv)


    w4_ensemble = []
    w_d_ensemble = []
    w_l_ensemble = []
    for i in range(N_TREE):
        #with tf.name_scope(get_tree_name(i) + '/' + 'FullyConnected/'):
        with tf.name_scope('FullyConnected-{:d}/'.format(i)):
            # FC layer input dim = output dim of upper network
            # 128 channels, 4 = round(28 / 2 / 2 / 2) (max-pooling)
            w4_ensemble.append(init_weights([final_layer_size, FC_output_dim],
                                            name='w4_ensemble_{:d}'.format(i)))
        with tf.name_scope(get_tree_name(i) + '/'):
            w_d_ensemble.append(init_prob_weights([FC_output_dim, N_LEAF], -1, 1,
                                                  name='w_d_ensemble_{:d}'.format(i)))
        w_l_ensemble.append(init_prob_weights([N_LEAF, n_class], -2, 2,
                                              name='w_l_ensemble_{:d}'.format(i)))

    ##################################################
    # Define a fully differentiable deep-ndf
    ##################################################
    # With the probability decision_p, route a sample to the right branch

    decision_p_e, leaf_p_e = model(final_layer_of_upper_model, w4_ensemble, w_d_ensemble,
                                   w_l_ensemble, p_keep_hidden)

    #with tf.name_scope('vec_decision_probs'):
    flat_decision_p_e = []
    count = 0
    # iterate over each tree
    for decision_p in decision_p_e:
        with tf.name_scope(get_tree_name(count)):
            # Compute the complement of d, which is 1 - d
                # where d is the sigmoid of fully connected output
            decision_p_comp = tf.subtract(tf.ones_like(decision_p), decision_p, name='decision_p_comp')

            # Concatenate both d, 1-d
            decision_p_pack = tf.stack([decision_p, decision_p_comp], name='decision_p_pack')

            # Flatten/vectorize the decision probabilities for efficient indexing
            flat_decision_p = tf.reshape(decision_p_pack, [-1], name='flat_decision_p')
            flat_decision_p_e.append(flat_decision_p)

            count += 1

    # 0 index of each data instance in a mini-batch
    batch_0_indices = \
        tf.tile(tf.expand_dims(tf.range(0, N_BATCH * N_LEAF, N_LEAF), 1),
                [1, N_LEAF], name='batch_0_indices')

    ###############################################################################
    # The routing probability (of each leaf node) computation
    #
    # We will create a routing probability matrix \mu. First, we will initialize
    # \mu using the root node d, 1-d. To efficiently implement this routing, we
    # will create a giant vector (matrix) that contains all d and 1-d from all
    # decision nodes. The matrix version of that is decision_p_pack and vectorized
    # version is flat_decision_p.
    #
    # The suffix `_e` indicates an ensemble. i.e. concatenation of all responsens
    # from trees.
    #
    # For depth = 2 tree, the routing probability for each leaf node can be easily
    # compute by multiplying the following vectors elementwise.
    # \mu =       [d_0,   d_0,   d_0,   d_0, 1-d_0, 1-d_0, 1-d_0, 1-d_0]
    # \mu = \mu * [d_1,   d_1, 1-d_1, 1-d_1,   d_2,   d_2, 1-d_2, 1-d_2]
    # \mu = \mu * [d_3, 1-d_3,   d_4, 1-d_4,   d_5, 1-d_5,   d_6, 1-d_6]
    #
    # Tree indexing
    #      0
    #    1   2
    #   3 4 5 6
    ##############################################################################
    with tf.name_scope('route_prob'):
        tree_scopes = []

        in_repeat = N_LEAF // 2
        out_repeat = N_BATCH

        # Let N_BATCH * N_LEAF be N_D. flat_decision_p[N_D] will return 1-d of the
        # first root node in the first tree.
        batch_complement_indices = \
            np.array([[0] * in_repeat, [N_BATCH * N_LEAF] * in_repeat]
                     * out_repeat).reshape(N_BATCH, N_LEAF)

        # First define the routing probabilities d for root nodes
        mu_e = []

        # iterate over each tree
        for i, flat_decision_p in enumerate(flat_decision_p_e):
            with tf.name_scope(get_tree_name(i)) as scope:
                tree_scopes.append(scope)
                mu = tf.gather(flat_decision_p,
                               tf.add(batch_0_indices, batch_complement_indices), name='mu_{:d}'.format(i))
                mu_e.append(mu)

        # from the second layer to the last layer, we make the decision nodes
        for d in range(1, DEPTH + 1):
            indices = tf.range(2 ** d, 2 ** (d + 1)) - 1
            tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1),
                                              [1, 2 ** (DEPTH - d + 1)]), [1, -1])
            batch_indices = tf.add(batch_0_indices, tf.tile(tile_indices, [N_BATCH, 1]))

            in_repeat = in_repeat // 2
            out_repeat = out_repeat * 2

            # Again define the indices that picks d and 1-d for the node
            batch_complement_indices = \
                np.array([[0] * in_repeat, [N_BATCH * N_LEAF] * in_repeat]
                         * out_repeat).reshape(N_BATCH, N_LEAF)

            mu_e_update = []
            count = 0
            for mu, flat_decision_p in zip(mu_e, flat_decision_p_e):
                with tf.name_scope(tree_scopes[count]):
                    mu = tf.multiply(mu, tf.gather(flat_decision_p,
                                              tf.add(batch_indices, batch_complement_indices)), name='mu_{:d}_new'.format(count))
                    mu_e_update.append(mu)
                    count += 1

            mu_e = mu_e_update

    ##################################################
    # Define p(y|x)
    ##################################################
    with tf.name_scope('P_y_cond_x'):
        py_x_e = []
        count = 0
        for mu, leaf_p in zip(mu_e, leaf_p_e):
            with tf.name_scope(get_tree_name(count)):
                # average all the leaf p
                py_x_tree = tf.reduce_mean(
                    tf.multiply(tf.tile(tf.expand_dims(mu, 2), [1, 1, n_class]),
                                tf.tile(tf.expand_dims(leaf_p, 0), [N_BATCH, 1, 1])),
                    1,
                    name='py_x_tree_{:d}'.format(count))
                py_x_e.append(py_x_tree)
                count += 1

        py_x_e = tf.stack(py_x_e, name='py_x_e')
        py_x = tf.reduce_mean(py_x_e, 0, name='py_x')

    def get_ladder(tick=1.0):
        mid = 0.0
        n_levels = n_class
        if n_levels % 2 == 0:
            lower = mid - tick * (n_levels / 2)
            upper = mid + tick * (n_levels / 2)
        else:
            lower = mid - tick * (n_levels // 2 + 0.5)
            upper = mid + tick * (n_levels // 2 + 0.5)

        return tf.linspace(lower, upper, n_levels)

    ##################################################
    # Define cost and optimization method
    ##################################################

    with tf.name_scope('cost_opt'):
        if regression:
            with tf.name_scope("continuous_output"):
                ladder = get_ladder()
                ladder_batch = tf.tile(tf.expand_dims(ladder, 0), [N_BATCH, 1])
                tmp = tf.multiply(py_x, ladder_batch)
                print(tmp.get_shape())
                y_pred_node = tf.reduce_sum(tmp, axis=1, name='Y_pred')
                print(y_pred_node.get_shape())
                assert(0)

            cost = tf.reduce_sum(
                tf.losses.mean_squared_error(Y, y_pred_node),
                name="MSE"
            )
        else:
            # cross entropy loss
            cost = tf.reduce_mean(-tf.multiply(tf.log(py_x), Y), name='cost') # brz

        tf.summary.scalar('cross_entropy', cost)

        # cost = tf.reduce_mean(tf.nn.cross_entropy_with_logits(py_x, Y))
        train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
        predict_step = tf.argmax(py_x, 1, name='predict')

    # return
    return train_step, predict_step, X, Y, p_keep_conv, p_keep_hidden


def load_custom_data():
    # from data_vendor import DataFutureTick as Vendor
    from data_vendor import DataMNIST as Vendor
    dv = Vendor()
    task_regression = 0

    trX, teX, trY, teY, input_shape_without_batch, n_classes = dv.get_data()

    # common transform
    input_reshape_arg = np.hstack([(-1, ),
                                   input_shape_without_batch]).tolist()
    input_shape = np.hstack([(N_BATCH, ),
                             input_shape_without_batch]).tolist()

    output_shape = [N_BATCH, n_classes]

    # data reshape
    trX = trX.reshape(*input_reshape_arg)
    teX = teX.reshape(*input_reshape_arg)

    return trX, teX, trY, teY, n_classes, input_shape, output_shape, task_regression


def init_and_run():
    print("Start init_and_run.")

    # prepare session and do initialization
    sess = tf.Session(config=gpuconfig)

    # summary writer
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('ndf_logs', sess.graph)

    sess.run(tf.initialize_all_variables())

    # loop run
    import time
    t0 = time.time()
    for i in range(100):
        # One epoch
        print("Need to run {:d} times per epoch.".format(len(trX) // N_BATCH))
        for start, end in zip(range(0, len(trX), N_BATCH), range(N_BATCH, len(trX), N_BATCH)):
            # print("start {} - end {}".format(start, end))
            print('.', end='', flush=True)
            summary_train, _ = sess.run([merged, train_step], feed_dict={X_in          : trX[start:end],
                                                                         Y_in          : trY[start:end],
                                                                         p_keep_conv   : 0.8,
                                                                         p_keep_hidden : 0.5})
            train_writer.add_summary(summary_train, i)
            train_writer.flush()

        # Result on the test set
        results = []
        for start, end in zip(range(0, len(teX), N_BATCH), range(N_BATCH, len(teX), N_BATCH)):
            y_pred = sess.run(predict_step, feed_dict={X_in          : teX[start:end],
                                                       p_keep_conv   : 1.0,
                                                       p_keep_hidden : 1.0})
            results.extend(
                np.argmax(teY[start:end], axis=1) == y_pred
                )
            accu_test = np.mean(results)
        print('\nEpoch: %d, Test Accuracy: %f' % (i + 1, accu_test))
        print(time.time() - t0)


if __name__ == "__main__":
    trX, teX, trY, teY, n_class, input_shape, output_shape, is_regression = load_custom_data()
    train_step, predict_step, X_in, Y_in, p_keep_conv, p_keep_hidden = \
        define_ndf(upper_model_choice='dnn', regression=is_regression)
    init_and_run()
    """
    """
