import tensorflow as tf
import numpy as np
import cPickle
import gzip
import datetime

params = {
        "learning_rate": 0.005,
        "training_epochs": 7,
        "batch_size": 100,
        "display_step": 1
        }

net_params = {
        "n_input": 784, # MNIST, iow
        "n_classes": 10
        }

def create_mlp(size, curr_x, keep_prob):
    # make vars
    h1 = tf.Variable(tf.truncated_normal([net_params["n_input"], size], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[size]))
    h2 = tf.Variable(tf.truncated_normal([size, size], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[size]))
    out = tf.Variable(tf.truncated_normal([size, net_params["n_classes"]], stddev=0.1))
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
    return tf.train.AdamOptimizer(learning_rate = params["learning_rate"]).minimize(loss)

def onehotify(batch):
    new_batch = np.zeros((len(batch), net_params["n_classes"]))
    for idx, member in enumerate(batch):
        new_batch[idx, member] = 1.0
    return new_batch

if __name__ == "__main__":
    with gzip.open("mnist.pkl.gz") as mnist_file:
        train, valid, test = cPickle.load(mnist_file)
    for size in xrange(50, 2500, 50):
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
                    batch_start = i * params["batch_size"]
                    # ugly, deal with it
                    batch_x, batch_y =\
                            train[0][batch_start:batch_start+params["batch_size"]],\
                            onehotify(train[1][batch_start:batch_start+params["batch_size"]])
                    curr_feed_dict = {keep_prob:0.5, curr_x: batch_x, curr_y: batch_y}
                    _, c = sess.run([curr_opt, curr_loss], feed_dict=curr_feed_dict)
                    avg_cost += c / total_batch
                # if epoch % params["display_step"] == 0:
                #     print "Epoch:", '%04d' % (epoch+1), "cost=", \
                #         "{:.9f}".format(avg_cost)
            print "opt finished"
            corrects = tf.equal(tf.argmax(curr_mlp, 1), tf.argmax(curr_y, 1))
            acc = tf.reduce_mean(tf.cast(corrects, "float"))
            print "acc: ", acc.eval({keep_prob:1.0, curr_x: test[0], curr_y: onehotify(test[1])})
