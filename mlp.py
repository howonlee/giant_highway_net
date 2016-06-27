import tensorflow as tf
import numpy as np
import cPickle
import gzip

params = {
        "learning_rate": 0.001,
        "training_epochs": 15,
        "batch_size": 100,
        "display_step": 1
        }

net_params = {
        "n_input": 784, # MNIST, iow
        "n_classes": 10
        }

def create_mlp(size, curr_x):
    # make vars
    h1 = tf.Variable(tf.random_normal([net_params["n_input"], size]))
    b1 = tf.Variable(tf.random_normal([size]))
    # h2 = tf.Variable(something) ????
    # b2 = tf.Variable(something) ???
    out = tf.Variable(tf.random_normal([size, net_params["n_classes"]]))
    bout = tf.Variable(tf.random_normal([net_params["n_classes"]]))

    # make layers
    layer_1 = tf.add(tf.matmul(curr_x, h1), b1)
    layer_1 = tf.nn.relu(layer_1)
    out_layer = tf.add(tf.matmul(layer_1, out), bout)
    return out_layer

def create_loss(mlp, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(mlp, y))

def create_opt(loss):
    return tf.train.AdamOptimizer(learning_rate = params["learning_rate"]).minimize(loss)

if __name__ == "__main__":
    with gzip.open("mnist.pkl.gz") as mnist_file:
        train, valid, test = cPickle.load(mnist_file)
    for size in xrange(50, 1000, 50):
        curr_x = tf.placeholder("float", [None, net_params["n_input"]])
        curr_y = tf.placeholder("float", [None, net_params["n_classes"]])
        curr_mlp = create_mlp(size, curr_x)
        curr_loss = create_loss(curr_mlp, curr_y)
        curr_opt = create_opt(curr_loss)
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in xrange(params["training_epochs"]):
                avg_cost = 0.0
                total_batch = int(len(train[0])/params["batch_size"])
                print "total batch", total_batch
                # Loop over all batches
                for i in range(total_batch):
                    batch_start = i * params["batch_size"]
                    batch_x, batch_y = train[0][batch_start:batch_start+params["batch_size"]], train[1][batch_start:batch_start+params["batch_size"]]
                    print batch_y # onehotify it
                    curr_feed_dict = {curr_x: batch_x, curr_y: batch_y}
                    _, c = sess.run([curr_opt, curr_loss], feed_dict=curr_feed_dict)
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % params["display_step"] == 0:
                    print "Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost)
        print "opt finished"
