import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math


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
    for ep_i in range(math.ceil(iter_num / batch_size)):
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
x = tf.reshape(raw_x, [-1, 28 * 28])
raw_y = tf.placeholder(tf.float32, [None, 1], name="input_y")

learning_rate = tf.placeholder(tf.float32, name='learning_rate')
lamb = tf.placeholder(tf.float32, name='lambda')

w = tf.Variable(tf.zeros([28 * 28, 1], dtype=tf.float32))
b = tf.Variable(tf.zeros(1, dtype=tf.float32))

pred_y = tf.add(tf.matmul(x, w), b)
mse = tf.reduce_mean(tf.square(pred_y - raw_y)) / 2

# TODO add wd to loss

loss = mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

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
    print("Q1:")
    rates = [0.005, 0.001, 0.0001]
    rate_losses = []
    plt.figure(0)
    plt.title("Q1")
    for rate in rates:
        sess, losses = train_model(optimizer, ITER_NUM, 500, trainData, trainTarget, rate=rate, l=0, vxs=validData,
                                   vys=validTarget)
        rate_loss = loss_wrapper(sess, lamb_=0)
        rate_losses.append(rate_loss)
        print("training rate: %f\t loss: %f" % (rate, rate_loss))

        plt.plot(range(len(losses)), losses)

    plt.show()
    return rates[rate_losses.index(min(rate_losses))]


best_rate = run_diff_rates()


def run_diff_batches():
    batches = [500, 1500, 3500]
    batch_losses = [
        loss_wrapper(train_model(optimizer, ITER_NUM, batch, trainData, trainTarget, rate=best_rate, l=0, vxs=validData,
                                 vys=validTarget)[0], lamb_=0)
        for batch in batches
        ]

    print("Q2:")
    for (batch, batch_losses) in zip(batches, batch_losses):
        print("batch size: %08d\t loss: %f" % (batch, batch_losses))

run_diff_batches()