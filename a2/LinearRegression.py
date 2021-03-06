import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

import time


def load_data():
    with np.load("./data/notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
        return trainData, trainTarget, validData, validTarget, testData, testTarget


def make_chunks(data, chunk_size):
    while data.any():
        chunk, data = data[:chunk_size], data[chunk_size:]
        yield chunk


def train_model(optimizer, iter_num, batch_size, xs, ys, rate, l, vxs, vys):
    """ @warning: need to close session """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    losses = []
    for ep_i in range(int(math.ceil(iter_num / (len(xs) / batch_size)))):
        for (chunk_x, chunk_y) in zip(make_chunks(xs, batch_size),
                                      make_chunks(ys, batch_size)):
            sess.run(optimizer, feed_dict={raw_x: chunk_x,
                                           raw_y: chunk_y,
                                           learning_rate: rate,
                                           lamb: l})
        losses.append(sess.run(loss, feed_dict={
            raw_x: vxs,
            raw_y: vys,
            lamb: l
        }))
    return sess, losses


trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()

raw_x = tf.placeholder(tf.float32, [None, 28, 28], name="input_x")
_x = tf.reshape(raw_x, [-1, 28 * 28])
x = tf.concat([tf.ones([tf.shape(_x)[0], 1]), _x], 1)

raw_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
y = raw_y

learning_rate = tf.placeholder(tf.float32, name='learning_rate')
lamb = tf.placeholder(tf.float32, name='lambda')

w = tf.Variable(tf.zeros([28 * 28 + 1, 1], dtype=tf.float32))

pred_y = tf.matmul(x, w)
mse = tf.reduce_mean(tf.square(pred_y - y)) / 2
wd = tf.multiply(lamb / 2, tf.reduce_sum(tf.square(w)))  # weight decay loss

loss = mse + wd
accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(tf.greater(pred_y, 0.5)), y)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer_adam = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

ITER_NUM = 20000  # total iteration


def loss_wrapper(sess, lamb_):
    result = sess.run(loss, feed_dict={
        raw_x: validData,
        raw_y: validTarget,
        lamb: lamb_
    })
    sess.close()
    return result


def run_diff_rates():
    rates = [0.005, 0.001, 0.0001]
    rate_losses = []
    plt.figure(0)
    plt.title("Q1")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    for rate in rates:
        sess, losses = train_model(optimizer, ITER_NUM, 500, trainData, trainTarget, rate=rate, l=0, vxs=validData,
                                   vys=validTarget)
        rate_loss = loss_wrapper(sess, lamb_=0)
        rate_losses.append(rate_loss)
        print("training rate: %f\t loss: %f" % (rate, rate_loss))

        plt.plot(range(len(losses)), losses, label="rate=" + str(rate))

    plt.grid()
    plt.legend()
    return rates[rate_losses.index(min(rate_losses))]


def run_diff_batches(best_rate):
    batches = [500, 1500, 3500]
    batch_losses = []
    batch_times = []
    for batch in batches:
        start = time.time()
        batch_losses.append(
            loss_wrapper(
                train_model(optimizer, ITER_NUM, batch, trainData, trainTarget, rate=best_rate, l=0, vxs=validData,
                            vys=validTarget)[0], lamb_=0)

        )
        end = time.time()
        batch_times.append(end - start)

    for (batch, batch_loss, batch_time) in zip(batches, batch_losses, batch_times):
        print("batch size: %08d\t loss: %f\ttime: %d s" % (batch, batch_loss, batch_time))


def run_diff_lambs():
    lambs = [0, 0.001, 0.1, 1]
    for l in lambs:
        sess, _ = train_model(optimizer, ITER_NUM, 500, trainData, trainTarget, rate=0.005, l=l, vxs=validData,
                              vys=validTarget)
        test_result = sess.run(accuracy, feed_dict={
            raw_x: testData,
            raw_y: testTarget,
            lamb: l
        })
        valid_result = sess.run(accuracy, feed_dict={
            raw_x: validData,
            raw_y: validTarget,
            lamb: l
        })
        sess.close()
        print("lambda: %f\t test accuracy: %f\t validation accuracy: %f" % (l, test_result, valid_result))


def run_normal_equation():
    train_raw_x = tf.placeholder(tf.float32, [None, 28, 28], name="train_raw_x")
    _train_x = tf.reshape(train_raw_x, [-1, 28 * 28])
    train_x = tf.concat([tf.ones([tf.shape(_train_x)[0], 1]), _train_x], 1)
    train_y = tf.placeholder(tf.float32, [None, 1], name="train_y")

    w_normal = \
        tf.matmul(
            tf.matmul(tf.matrix_inverse(
                tf.matmul(train_x, train_x, transpose_a=True)),
                train_x, transpose_b=True), train_y)

    pred_y_normal = tf.matmul(x, w_normal)
    mse_normal = tf.reduce_mean(tf.square(pred_y_normal - y)) / 2
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(tf.greater(pred_y_normal, 0.5)), y)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start = time.time()
        result = sess.run(mse_normal, feed_dict={
            raw_x: trainData,
            raw_y: trainTarget,
            train_raw_x: trainData,
            train_y: trainTarget
        })
        end = time.time()
        print("normal equation training loss: %f" % result)
        print("normal equation takes %f s" % (end - start))

        test_acc = sess.run(accuracy, feed_dict={raw_x: testData, raw_y: testTarget, train_raw_x: trainData,
                                                 train_y: trainTarget})
        train_acc = sess.run(accuracy, feed_dict={raw_x: trainData, raw_y: trainTarget, train_raw_x: trainData,
                                                  train_y: trainTarget})
        valid_acc = sess.run(accuracy, feed_dict={raw_x: validData, raw_y: validTarget, train_raw_x: trainData,
                                                  train_y: trainTarget})

        print("normal equation training/test/validation accuracy: %f\t%f\t%f" % (train_acc, test_acc, valid_acc))


def run_sgd_equation():
    start = time.time()
    sess, _ = train_model(optimizer, ITER_NUM, 500, trainData, trainTarget, rate=0.005, l=0, vxs=validData,
                          vys=validTarget)
    end = time.time()
    print("SGD equation takes %f s" % (end - start))
    train_result = sess.run(loss, feed_dict={
        raw_x: trainData,
        raw_y: trainTarget,
        lamb: 0
    })
    sess.close()
    print("SGD training loss: %f" % train_result)


def run_linear():
    """ for comparison with logistic regression """
    iter_num = 5000
    batch_size = 500
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    losses = []
    accs = []
    for ep_i in range(int(math.ceil(iter_num / (len(trainData) / batch_size)))):
        for (chunk_x, chunk_y) in zip(make_chunks(trainData, batch_size),
                                      make_chunks(trainTarget, batch_size)):
            sess.run(optimizer_adam, feed_dict={raw_x: chunk_x,
                                                raw_y: chunk_y,
                                                learning_rate: 0.001,
                                                lamb: 0})
        train_dict = {
            raw_x: trainData,
            raw_y: trainTarget,
            lamb: 0
        }
        losses.append(sess.run(loss, feed_dict=train_dict))
        accs.append(sess.run(accuracy, feed_dict=train_dict))

    sess.close()
    return losses, accs


if __name__ == "__main__":
    print("Q1")
    best_rate = run_diff_rates()
    print("Best learning rate is %f" % best_rate)

    print("Q2")
    run_diff_batches(best_rate)

    print("Q3")
    run_diff_lambs()

    print("Q4")
    run_normal_equation()
    run_sgd_equation()

    plt.show()
