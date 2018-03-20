import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import reduce

CATEGORIES = ['train', 'test', 'validation']
MODEL_DIR = './model'


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
        return {
                   'train': trainData,
                   'test': testData,
                   'validation': validData
               }, {
                   'train': trainTarget,
                   'test': testTarget,
                   'validation': validTarget
               }


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


def train_model(tf_ctx, data, target, learning_rate=0.005,
                epoch=100, batch_size=3000, name="default_model"):
    """
    general method to train a tensorflow model
    :param tf_ctx: tensorflow context, i.e. a dictionary which stores
    value needed
    :param data: data dict(train, test, validation)
    :param target: target dict(train, test, validation)
    :param learning_rate: learning rate
    :param epoch: epoch value
    :param batch_size: batch size to use
    :param name: name of the model, used for saving files
    :return:
    """

    optimizer = tf_ctx['optimizer']
    loss = tf_ctx['loss']
    error = tf_ctx['error']
    x = tf_ctx['x']
    y = tf_ctx['y']
    base_dict = {
        tf_ctx['wd_coeff']: 0.0003,
        tf_ctx['learning_rate']: learning_rate
    }
    losses = {
        'train': [],
        'test': [],
        'validation': []
    }
    errors = {
        'train': [],
        'test': [],
        'validation': []
    }
    print("start training " + name + " with context:", tf_ctx)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        save_freq = epoch // 4  # save every epoch / 4 epoch
        saver = tf.train.Saver()

        for i in range(epoch):
            for (chunk_x, chunk_y) in zip(make_chunks(data['train'], batch_size),
                                          make_chunks(target['train'], batch_size)):
                sess.run(optimizer, feed_dict={**base_dict, x: chunk_x, y: chunk_y})
            for category in CATEGORIES:
                losses[category].append(
                    sess.run(loss, feed_dict={**base_dict, x: data[category], y: target[category]}))
                errors[category].append(
                    sess.run(error, feed_dict={**base_dict, x: data[category], y: target[category]}))
            percentage = (100 * i / epoch)
            print("training in progress", percentage, "%")
            if i % save_freq == save_freq - 1:
                saved_path = saver.save(sess, MODEL_DIR + "/" + name + str(percentage) + ".ckpt")
                print(name + " saved to " + saved_path)

    print("training complete")
    return losses, errors


def compare_learning_rate(tf_ctx, data, target, epoch, batch_size):
    name = "convergence comparison"
    plt.figure(name)
    plt.title(name)
    plt.xlabel("epoch")
    plt.ylabel("training loss")

    for LEARNING_RATE in [0.005, 0.001, 0.0001]:
        ep_range = range(epoch)
        losses, _ = train_model(tf_ctx, data, target, LEARNING_RATE, epoch, batch_size,
                                name="simple_nn_compare_rate")
        plt.plot(ep_range, losses['train'], label="learning rate " + str(LEARNING_RATE))

    plt.grid()
    plt.legend()
    plt.show()


def simple_nn_training(tf_ctx, data, target, epoch, batch_size):
    losses, errors = train_model(tf_ctx, data, target,
                                 learning_rate=0.005, epoch=epoch, batch_size=batch_size,
                                 name="simple_nn")
    ep_range = range(epoch)
    title_1 = "single hidden layer network losses"
    plt.figure(title_1)
    plt.title(title_1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    for category in CATEGORIES:
        plt.plot(ep_range, losses[category], label=category)
        print("Best {} loss: {}".format(category, min(losses[category])))
    plt.grid()
    plt.legend()
    plt.show()

    title_2 = "single hidden layer network errors"
    plt.figure(title_2)
    plt.title(title_2)
    plt.xlabel("epoch")
    plt.ylabel("error")
    for category in CATEGORIES:
        plt.plot(ep_range, errors[category], label=category)
        print("Best {} error: {}".format(category, min(errors[category])))
    plt.grid()
    plt.legend()
    plt.show()


def create_tf_ctx(x_size, y_size, h_size, class_num):
    tf.reset_default_graph()
    input_size = reduce((lambda a, b: a * b), x_size)
    output_size = class_num

    x_shape = [None]
    x_shape.extend(x_size)
    y_shape = [None]
    y_shape.extend(y_size)

    x = tf.placeholder(tf.float32, shape=x_shape, name="input_x")
    y = tf.placeholder(tf.int32, shape=y_shape, name="input_y")
    wd_coeff = tf.placeholder(tf.float32, name="weight_decay_coefficient")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    _x = tf.reshape(x, [-1, input_size])
    _y = tf.one_hot(y, depth=output_size, dtype=tf.float32)

    signal, w1 = simple_nn_init(_x, h_size, input_size)
    linear_pred_y, w2 = simple_nn_layer(signal, output_size, h_size)
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

    return {
        'x': x,
        'y': y,
        'loss': loss,
        'error': error,
        'optimizer': optimizer,
        'wd_coeff': wd_coeff,
        'learning_rate': learning_rate
    }


def main():
    data, target = load_data()
    trainData = data['train']
    trainTarget = target['train']
    class_num = max(trainTarget) - min(trainTarget) + 1

    x_size = np.shape(trainData)[1:]
    y_size = np.shape(trainTarget)[1:]

    # 1.1.2
    # tf_ctx_1_1_2 = create_tf_ctx(x_size, y_size, h_size=1000, class_num=class_num)
    # compare_learning_rate(tf_ctx_1_1_2, data, target, epoch=50, batch_size=3000)
    # simple_nn_training(tf_ctx_1_1_2, data, target, epoch=50, batch_size=5000)

    # 1.2.1
    for h_size in [100, 500, 1000]:
        tf_ctx_1_2_1 = create_tf_ctx(x_size, y_size, h_size, class_num)
        simple_nn_training(tf_ctx_1_2_1, data, target, epoch=50, batch_size=3000)


if __name__ == '__main__':
    main()
