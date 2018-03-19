import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import reduce


def load_data():
    with np.load("data/notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        return trainData, trainTarget, validData, validTarget, testData, testTarget


def make_chunks(data, chunk_size):
    while data.any():
        chunk, data = data[:chunk_size], data[chunk_size:]
        yield chunk


def simple_nn_init(hidden_activations, num_hidden, data_size=28 * 28,
                   initializer=tf.contrib.layers.xavier_initializer()):
    """
    This function also init the weight matrix and the biases
    :param hidden_activations: output of previous layer (x_{l - 1})
    :param num_hidden: number of hidden units
    :param data_size: size of data (length of pic array)
    :param initializer: initializer of weight matrix
    :return: weighted sum of the inputs(s_l)
    """
    w = tf.get_variable("weight_matrix_1", shape=[data_size, num_hidden],
                        initializer=initializer,
                        dtype=tf.float32)
    bias = tf.Variable(0, dtype=tf.float32)
    signal = tf.add(tf.matmul(hidden_activations, w), bias, name="s_1")
    return signal, w


def simple_nn_layer(signal, output_size, input_size=1000,
                    initializer=tf.contrib.layers.xavier_initializer()):
    x_1 = tf.nn.relu(signal, "x_1")
    w = tf.get_variable("weight_matrix_2", shape=[input_size, output_size],
                        initializer=initializer,
                        dtype=tf.float32)
    bias = tf.Variable(0, dtype=tf.float32)
    pred_y = tf.add(tf.matmul(x_1, w), bias, name="pred_y")
    return pred_y, w


def compare_learning_rate(learning_rate, wd_coeff, x, y,
                          trainData, trainTarget, optimizer, loss):
    plt.figure()
    plt.title("convergence comparison")
    plt.xlabel("epoch")
    plt.ylabel("training loss")

    for LEARNING_RATE in [0.005, 0.001, 0.0001]:
        ep_range = range(50)
        batch_size = 3000
        base_dict = {
            wd_coeff: 0.0003,
            learning_rate: LEARNING_RATE
        }
        train_loss = []
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for _ in ep_range:
                for (chunk_x, chunk_y) in zip(make_chunks(trainData, batch_size),
                                              make_chunks(trainTarget, batch_size)):
                    sess.run(optimizer, feed_dict={**base_dict, x: chunk_x, y: chunk_y})
                train_loss.append(sess.run(
                    loss, feed_dict={**base_dict, x: trainData, y: trainTarget}))
        plt.plot(ep_range, train_loss, label="learning rate " + str(LEARNING_RATE))

    plt.grid()
    plt.legend()
    plt.show()


def main():
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()
    class_num = max(trainTarget) - min(trainTarget) + 1

    x_size = np.shape(trainData)[1:]
    y_size = np.shape(trainTarget)[1:]
    h_size = 1000

    x_flat_size = reduce((lambda a, b: a * b), x_size)

    x_shape = [None]
    x_shape.extend(x_size)
    y_shape = [None]
    y_shape.extend(y_size)

    x = tf.placeholder(tf.float32, shape=x_shape, name="input_x")
    y = tf.placeholder(tf.int32, shape=y_shape, name="input_y")
    wd_coeff = tf.placeholder(tf.float32, name="weight_decay_coefficient")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    _x = tf.reshape(x, [-1, x_flat_size])
    _y = tf.one_hot(y, depth=class_num, dtype=tf.float32)

    signal, w1 = simple_nn_init(_x, h_size, x_flat_size)
    linear_pred_y, w2 = simple_nn_layer(signal, class_num, h_size)
    pred_y = tf.nn.softmax(linear_pred_y)

    ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=linear_pred_y),
                        name="cross_entropy_loss")
    wd = tf.multiply(wd_coeff / 2, tf.reduce_sum(tf.square(w1)) + tf.reduce_sum(tf.square(w2)),
                     name="weight_decay_loss")
    loss = ce + wd

    accuracy = tf.reduce_mean(tf.to_float(tf.equal(
        tf.argmax(pred_y, axis=1),
        tf.argmax(_y, axis=1)
    )))

    error = 1.0 - accuracy

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # 1.1.2
    compare_learning_rate(learning_rate, wd_coeff, x, y,
                          trainData, trainTarget, optimizer, loss)


if __name__ == '__main__':
    main()
