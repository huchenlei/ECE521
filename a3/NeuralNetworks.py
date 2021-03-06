import random
from functools import reduce

import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

CATEGORIES = ['train', 'test', 'validation']
MODEL_DIR = './model'
SEED = 521


def model_paths(name):
    return [MODEL_DIR + "/" + name + checkpoint + ".ckpt" for checkpoint in ["25", "50", "75", "100"]]


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
                    initializer=tf.contrib.layers.xavier_initializer(),
                    layer_id=2, keep_prob=1.0):
    x_1 = tf.nn.relu(signal, "x_1")
    w = tf.get_variable("weight_matrix_" + str(layer_id), shape=[input_size, output_size],
                        initializer=initializer,
                        dtype=tf.float32)
    bias = tf.Variable(0, dtype=tf.float32)
    # drop out with 0.5 probability
    _x_1 = tf.nn.dropout(x_1, keep_prob=keep_prob, seed=SEED)
    pred_y = tf.add(tf.matmul(_x_1, w), bias, name="pred_y")
    return pred_y, w


def save_model(sess, name):
    saver = tf.train.Saver()
    return saver.save(sess, MODEL_DIR + "/" + name + ".ckpt")


def train_model(tf_ctx, data, target, learning_rate=0.005,
                epoch=100, batch_size=3000, wd_coeff=0.0003, name="default_model",
                dropout=False, overfit_threshold=0.03):
    """
    general method to train a tensorflow model
    :param overfit_threshold: the error bound to judge whether the model gets overfit
    :param wd_coeff: weight decay coefficient
    :param tf_ctx: tensorflow context, i.e. a dictionary which stores
    value needed
    :param data: data dict(train, test, validation)
    :param target: target dict(train, test, validation)
    :param learning_rate: learning rate
    :param epoch: epoch value
    :param batch_size: batch size to use
    :param name: name of the model, used for saving files
    :param dropout: whether to apply 0.5 prob dropout
    :return:
    """

    optimizer = tf_ctx['optimizer']
    loss = tf_ctx['loss']
    error = tf_ctx['error']
    x = tf_ctx['x']
    y = tf_ctx['y']
    keep_prob = tf_ctx['keep_prob']

    base_dict = {
        tf_ctx['wd_coeff']: wd_coeff,
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

        _keep_prob = 0.5 if dropout else 1.0
        prev_percentage = 0

        for i in range(epoch):
            for (chunk_x, chunk_y) in zip(make_chunks(data['train'], batch_size),
                                          make_chunks(target['train'], batch_size)):
                sess.run(optimizer, feed_dict={**base_dict, x: chunk_x, y: chunk_y,
                                               keep_prob: _keep_prob})
            for category in CATEGORIES:
                losses[category].append(
                    sess.run(loss, feed_dict={**base_dict,
                                              x: data[category], y: target[category],
                                              keep_prob: 1.0}))
                errors[category].append(
                    sess.run(error, feed_dict={**base_dict,
                                               x: data[category], y: target[category],
                                               keep_prob: 1.0}))

            percentage = int(100 * i / epoch)
            if prev_percentage != percentage:
                print("training in progress", percentage, "%")
                print("validation error: %f training error: %f" % (errors['validation'][-1], errors['train'][-1]))
                prev_percentage = percentage

            # Save model at checkpoints
            if i % save_freq == save_freq - 1:
                saved_path = save_model(sess, name + str(25 * ((i + 1) // save_freq)))
                # print(name + " saved to " + saved_path)

            if errors['train'][-1] < 0.01 \
                    and (errors['test'][-1] - errors['train'][-1] > overfit_threshold or errors['validation'][-1] -
                         errors['train'][-1] > overfit_threshold or errors['test'][-1] < overfit_threshold or
                         errors['validation'][-1] < overfit_threshold):
                print("Over-fit detected")
                save_model(sess, name + "_final")
                break

    print("training complete")
    return losses, errors


def compare_learning_rate(tf_ctx, data, target, epoch, batch_size):
    name = "convergence comparison"
    plt.figure(name)
    plt.title(name)
    plt.xlabel("epoch")
    plt.ylabel("training loss")

    for LEARNING_RATE in [0.005, 0.001, 0.0001]:
        losses, errors = train_model(tf_ctx, data, target, LEARNING_RATE, epoch, batch_size,
                                     name="simple_nn_compare_rate")
        ep_range = range(len(losses['train']))
        plt.plot(ep_range, losses['train'], label="learning rate " + str(LEARNING_RATE))

    plt.grid()
    plt.legend()
    plt.show()


def simple_nn_training(tf_ctx, data, target, epoch, batch_size, dropout=False, name="simple_nn",
                       learning_rate=0.005, wd_coeff=0.0003, overfit_threshold=0.03):
    losses, errors = train_model(tf_ctx, data, target,
                                 learning_rate=learning_rate, epoch=epoch, batch_size=batch_size,
                                 wd_coeff=wd_coeff, name=name, dropout=dropout,
                                 overfit_threshold=overfit_threshold)
    ep_range = range(len(losses['train']))
    title_1 = "Losses: " + name
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

    title_2 = "Errors: " + name
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

    return losses, errors


def create_tf_ctx(x_size, y_size, class_num, layer_sizes):
    assert len(layer_sizes) > 0

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
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    _x = tf.reshape(x, [-1, input_size])
    _y = tf.one_hot(y, depth=output_size, dtype=tf.float32)

    signal, w1 = simple_nn_init(_x, layer_sizes[0], input_size)
    signals = [signal]
    weights = [w1]
    for i in range(1, len(layer_sizes)):
        s, w = simple_nn_layer(signals[i - 1], layer_sizes[i], layer_sizes[i - 1],
                               layer_id=i + 1, keep_prob=keep_prob)
        signals.append(s)
        weights.append(w)

    linear_pred_y, w2 = simple_nn_layer(signals[-1], output_size, layer_sizes[-1],
                                        layer_id=len(layer_sizes) + 1, keep_prob=keep_prob)
    weights.append(w2)

    pred_y = tf.nn.softmax(linear_pred_y)

    ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=linear_pred_y),
                        name="cross_entropy_loss")

    total_w_square = tf.add_n([tf.reduce_mean(tf.square(w)) for w in weights])
    wd = tf.multiply(wd_coeff / 2, total_w_square, name="weight_decay_loss")
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
        'learning_rate': learning_rate,
        'keep_prob': keep_prob
    }


def visualize_model(name, shape=None):
    if shape is None:
        shape = [28 * 28, 1000]
    tf.reset_default_graph()
    w = tf.get_variable("weight_matrix_1", shape=shape)
    saver = tf.train.Saver()
    for i, path in enumerate(model_paths(name=name)):
        with tf.Session() as sess:
            saver.restore(sess, path)
            w_vals = w.eval().T
            show_model(w_vals, name + "-" + str(i))


def show_model(w_vals, name, shape=None):
    """
    Use PIL to show w_vals as image
    :param shape: shape of image
    :param w_vals: weight matrix in shape [1000, 28*28]
    :param name: name of picture
    :return: void
    """
    if shape is None:
        shape = [28, 28]
    grid_weight = 2
    # Every row has 10 images
    row_count = 25
    output_img = Image.new('L', (row_count * (shape[0] + grid_weight),
                                 (len(w_vals) // row_count) * (shape[1] + grid_weight)))
    x_offset = 0
    y_offset = 0

    for i, w_val in enumerate(w_vals):
        scale = max(w_val) - min(w_val)
        img = Image.fromarray(
            np.uint8((w_val + abs(min(w_val))).reshape(shape[0], shape[1]) * int(255)))
        output_img.paste(img, (x_offset, y_offset))
        x_offset += shape[0] + grid_weight
        if i % row_count == row_count - 1:
            x_offset = 0
            y_offset += shape[1] + grid_weight
    # output_img.show(title=name)
    output_img.save("./pic/" + name + ".bmp")


def random_train(x_size, y_size, class_num, data, target, name="random"):
    layer_size = random.randint(1, 5)
    layer_sizes = [random.randint(100, 501) for _ in range(layer_size)]
    dropout = random.choice([True, False])
    wd_coeff = math.exp(random.uniform(-9, -6))
    learning_rate = math.exp(random.uniform(-7.5, -4.5))

    info = "Training with {} layers, dropout={} wd_coeff={} learning_rate={}".format(layer_sizes, dropout, wd_coeff,
                                                                                     learning_rate)
    print(info)

    tf_ctx_random = create_tf_ctx(x_size, y_size, class_num, layer_sizes=layer_sizes)
    _, errors = simple_nn_training(tf_ctx_random, data, target, epoch=1000, batch_size=3000, name=name,
                                   dropout=dropout, wd_coeff=wd_coeff, learning_rate=learning_rate,
                                   overfit_threshold=0.05)
    return min(errors['test']), info


def ideal_train(x_size, y_size, class_num, data, target, name="ideal"):
    layer_sizes = [488, 488]
    dropout = True
    wd_coeff = 0.00926529
    learning_rate = 0.00106265

    info = "Training with {} layers, dropout={} wd_coeff={} learning_rate={}".format(layer_sizes, dropout, wd_coeff,
                                                                                     learning_rate)
    print(info)

    tf_ctx_random = create_tf_ctx(x_size, y_size, class_num, layer_sizes=layer_sizes)
    _, errors = simple_nn_training(tf_ctx_random, data, target, epoch=1000, batch_size=3000, name=name,
                                   dropout=dropout, wd_coeff=wd_coeff, learning_rate=learning_rate,
                                   overfit_threshold=0.05)
    return min(errors['test']), info


def main():
    data, target = load_data()
    trainData = data['train']
    trainTarget = target['train']
    class_num = max(trainTarget) - min(trainTarget) + 1

    x_size = np.shape(trainData)[1:]
    y_size = np.shape(trainTarget)[1:]

    # 1.1.2
    tf_ctx_1_1_2 = create_tf_ctx(x_size, y_size, layer_sizes=[1000], class_num=class_num)
    compare_learning_rate(tf_ctx_1_1_2, data, target, epoch=50, batch_size=3000)
    simple_nn_training(tf_ctx_1_1_2, data, target, epoch=150, batch_size=5000, name="1.1.2")

    # 1.2.1
    for h_size in [100, 500, 1000]:
        tf_ctx_1_2_1 = create_tf_ctx(x_size, y_size, class_num, layer_sizes=[h_size])
        simple_nn_training(tf_ctx_1_2_1, data, target, epoch=150, batch_size=3000, name="1.2.1-" + str(h_size))

    # 1.2.2
    tf_ctx_1_2_2 = create_tf_ctx(x_size, y_size, class_num, layer_sizes=[500, 500])
    simple_nn_training(tf_ctx_1_2_2, data, target, epoch=150, batch_size=3000, name="1.2.2")

    # 1.3.1
    tf_ctx_1_3_1 = create_tf_ctx(x_size, y_size, class_num, layer_sizes=[1000])
    simple_nn_training(tf_ctx_1_3_1, data, target, epoch=150, batch_size=3000, dropout=True, name="1.3.1")

    # 1.3.2
    visualize_model("1.1.2")
    visualize_model("1.3.1")

    # 1.4.1
    for i in range(5):
        random_train(x_size, y_size, class_num, data, target, name="random" + str(i))

    # 1.4.2
    best_err = 1.0
    best_model = ""
    for i in range(100):
        err, info = random_train(x_size, y_size, class_num, data, target, name="random" + str(i))
        if err < best_err:
            best_err = err
            best_model = info

    print(str(best_err) + ": " + best_model)

    ideal_train(x_size, y_size, class_num, data, target)


if __name__ == '__main__':
    main()
