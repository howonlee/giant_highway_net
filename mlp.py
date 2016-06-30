import tensorflow as tf
import numpy as np
import math
import cPickle
import gzip
import datetime

params = {
        "training_epochs": 25,
        "batch_size": 200,
        "display_step": 1
        }

net_params = {
        "n_input": 3072, # CIFAR-100, iow
        "n_classes": 100
        }

def create_mlp(size, curr_x, keep_prob):
    # make vars
    h1_stddev = math.sqrt(3.0 / (net_params["n_input"] + size))
    h1 = tf.Variable(tf.truncated_normal([net_params["n_input"], size], stddev=h1_stddev))
    b1 = tf.Variable(tf.constant(0.1, shape=[size]))
    h2_stddev = math.sqrt(3.0 / (size + size))
    h2 = tf.Variable(tf.truncated_normal([size, size], stddev=h2_stddev))
    b2 = tf.Variable(tf.constant(0.1, shape=[size]))
    out_stddev = math.sqrt(3.0 / (size + net_params["n_classes"]))
    out = tf.Variable(tf.truncated_normal([size, net_params["n_classes"]], stddev=out_stddev))
    bout = tf.Variable(tf.constant(0.1, shape=[net_params["n_classes"]]))

    # make layers
    layer_1 = tf.add(tf.matmul(curr_x, h1), b1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, h2), b2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.add(tf.matmul(layer_2, out), bout)
    return out_layer

def create_loss(mlp, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(mlp, y))

def create_opt(loss):
    return tf.train.AdamOptimizer().minimize(loss)

def onehotify(batch):
    new_batch = np.zeros((len(batch), net_params["n_classes"]))
    for idx, member in enumerate(batch):
        new_batch[idx, member] = 1.0
    return new_batch

if __name__ == "__main__":
    with open("cifar-100-python/train", "rb") as train_file:
        train_dict = cPickle.load(train_file)
        train = [train_dict['data'] / 255.0, np.array(train_dict['fine_labels'])]
    with open("cifar-100-python/test", "rb") as test_file:
        test_dict = cPickle.load(test_file)
        test = [test_dict['data'] / 255.0, np.array(test_dict['fine_labels'])]
    for size in xrange(500, 2500, 500):
        curr_x = tf.placeholder("float", [None, net_params["n_input"]])
        curr_y = tf.placeholder("float", [None, net_params["n_classes"]])
        keep_prob = tf.placeholder("float")
        curr_mlp = create_mlp(size, curr_x, keep_prob)
        curr_loss = create_loss(curr_mlp, curr_y)
        curr_opt = create_opt(curr_loss)
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            print "size: ", size
            print "================="
            for epoch in xrange(params["training_epochs"]):
                print "epoch: ", epoch, "time: ", datetime.datetime.now()
                avg_cost = 0.0
                total_batch = int(len(train[0])/params["batch_size"])
                # print "total batch", total_batch
                for i in range(total_batch):
                    if i % 20 == 0:
                        print i, " / ", total_batch, datetime.datetime.now()
                    batch_start = i * params["batch_size"]
                    # ugly, deal with it
                    batch_x, batch_y =\
                            train[0][batch_start:batch_start+params["batch_size"]],\
                            onehotify(train[1][batch_start:batch_start+params["batch_size"]])
                    curr_feed_dict = {keep_prob:0.5, curr_x: batch_x, curr_y: batch_y}
                    _, c = sess.run([curr_opt, curr_loss], feed_dict=curr_feed_dict)
                    avg_cost += c / total_batch
            print "opt finished"
            corrects = tf.equal(tf.argmax(curr_mlp, 1), tf.argmax(curr_y, 1))
            acc = tf.reduce_mean(tf.cast(corrects, "float"))
            print "acc: ", acc.eval({keep_prob:1.0, curr_x: test[0], curr_y: onehotify(test[1])})
