import tensorflow as tf
import numpy as np

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

def create_mlp(size):
    # make vars
    x = tf.placeholder("float", [None, net_params["n_input"]])
    y = tf.placeholder("float", [None, net_params["n_classes"]])
    h1 = tf.Variable(tf.random_normal([net_params["n_input"], size]))
    b1 = tf.Variable(tf.random_normal([size]))
    # h2 = tf.Variable(something) ????
    # b2 = tf.Variable(something) ???
    out = tf.Variable(tf.random_normal([size, net_params["n_classes"]]))
    bout = tf.Variable(tf.random_normal([net_params["n_classes"]]))

    # make layers
    layer_1 = tf.add(tf.matmul(x, h1), b1)
    layer_1 = tf.nn.relu(layer_1)
    out_layer = tf.add(tf.matmul(layer_1, out), bout)
    return out_layer, y

def create_loss(mlp, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(mlp, y))

def create_opt(loss):
    return tf.train.AdamOptimizer(learning_rate = params["learning_rate"]).minimize(loss)

if __name__ == "__main__":
    for size in xrange(50, 1000, 50):
        curr_mlp = create_mlp(size)
        curr_loss = create_loss(curr_mlp)
        curr_opt = create_opt(curr_loss)
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in xrange(params["training_epochs"]):
                avg_cost = 0.0
                #######################
                #######################
                #######################
                #######################
                total_batch = int(mnist.train.num_examples/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    #### batch_x, batch_y = mnist.train.next_batch(batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                  y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % params["display_step"] == 0:
                    print "Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost)
    print "Optimization Finished!"

                stuff
            print "opt finished"
