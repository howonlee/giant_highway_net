import tensorflow as tf

def create_mlp(size):
    # h1 = tf.Variable(something)
    # h2 = tf.Variable(something)
    # out = tf.Variable(something)

    # b1 = tf.Variable(something)
    # b2 = tf.Variable(something)
    # bout = tf.Variable(something)
    return size

def create_loss(mlp):
    pass

def create_opt(loss):
    pass

if __name__ == "__main__":
    for size in xrange(50, 1000, 50):
        curr_mlp = create_mlp(size)
        curr_loss = create_loss(curr_mlp)
        curr_opt = create_opt(curr_loss)
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            print "woobs"
