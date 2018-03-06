import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

from a2.LinearRegression import run_linear


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


trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()

raw_x = tf.placeholder(tf.float32, [None, 28, 28], name="input_x")
_x = tf.reshape(raw_x, [-1, 28 * 28])
x = tf.concat([tf.ones([tf.shape(_x)[0], 1]), _x], 1)

raw_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
y = raw_y

learning_rate = tf.placeholder(tf.float32, name='learning_rate')
lamb = tf.placeholder(tf.float32, name='lambda')

w = tf.Variable(tf.zeros([28 * 28 + 1, 1], dtype=tf.float32))

linear_pred_y = tf.matmul(x, w)
pred_y = tf.nn.sigmoid(linear_pred_y)
ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=linear_pred_y), name="cross_entropy_loss")
wd = tf.multiply(lamb / 2, tf.reduce_sum(tf.square(w)), name="weight_decay_loss")

loss = ce + wd
# accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.round(pred_y), y)))
accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(tf.greater(pred_y, 0.5)), y)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer_adam = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

ITER_NUM = 5000  # total iteration
BATCH_SIZE = 500
LEARNING_RATE = 0.001
LAMBDA = 0.01


def train_model(optimizer, l=LAMBDA):
    train_ces = []
    valid_ces = []
    train_accs = []
    valid_accs = []
    test_accs = []
    train_dict = {
        raw_x: trainData,
        raw_y: trainTarget,
        lamb: l
    }
    valid_dict = {
        raw_x: validData,
        raw_y: validTarget,
        lamb: l
    }
    test_dict = {
        raw_x: testData,
        raw_y: testTarget,
        lamb: l
    }
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ep_range = range(int(math.ceil(ITER_NUM / (len(trainData) / BATCH_SIZE))))
    for _ in ep_range:
        for (chunk_x, chunk_y) in zip(make_chunks(trainData, BATCH_SIZE),
                                      make_chunks(trainTarget, BATCH_SIZE)):
            sess.run(optimizer, feed_dict={raw_x: chunk_x,
                                           raw_y: chunk_y,
                                           learning_rate: LEARNING_RATE,
                                           lamb: l})
        train_ces.append(sess.run(ce, feed_dict=train_dict))
        valid_ces.append(sess.run(ce, feed_dict=valid_dict))
        train_accs.append(sess.run(accuracy, feed_dict=train_dict))
        valid_accs.append(sess.run(accuracy, feed_dict=valid_dict))
        test_accs.append(sess.run(accuracy, feed_dict=test_dict))

    sess.close()
    return train_ces, valid_ces, train_accs, valid_accs, test_accs


def run_logistic():
    train_ces, valid_ces, train_accs, valid_accs, test_accs = train_model(optimizer=optimizer)
    ep_range = range(len(train_ces))

    print("Best test classification accuracy: %f" % max(test_accs))

    plt.figure(0)
    plt.title("Q1")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(ep_range, train_ces, label="train cross-entropy loss")
    plt.plot(ep_range, valid_ces, label="validation cross-entropy loss")
    plt.grid()
    plt.legend()

    plt.figure(1)
    plt.title("Q1")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(ep_range, train_accs, label="train accuracy")
    plt.plot(ep_range, valid_accs, label="validation accuracy")
    plt.grid()
    plt.legend()


def run_adam():
    train_ces_adam, _, _, _, _ = train_model(optimizer=optimizer_adam)
    train_ces, _, _, _, _ = train_model(optimizer=optimizer)

    ep_range = range(len(train_ces))
    plt.figure(2)
    plt.title("Q2")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(ep_range, train_ces, label="SGD cross-entropy loss")
    plt.plot(ep_range, train_ces_adam, label="Adam cross-entropy loss")
    plt.grid()
    plt.legend()


def run_adam_zl():
    """ run adam with zero lambda """
    _, _, train_accs, valid_accs, test_accs = train_model(optimizer=optimizer_adam, l=0)
    print("logistic regression training/test/validation accuracy: %f\t%f\t%f" % (
        train_accs[-1], valid_accs[-1], test_accs[-1]))


def compare_linear_logistic():
    train_ces, _, _, _, _ = train_model(optimizer=optimizer_adam)
    train_mses = run_linear()

    ep_range = range(len(train_ces))
    plt.figure(3)
    plt.title("Q3")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(ep_range, train_ces, label="logistic cross entropy loss")
    plt.plot(ep_range, train_mses, label="linear mean square error loss")
    plt.grid()
    plt.legend()

    my_pred_y = tf.placeholder(tf.float32, name="my_pred_y")
    my_y = tf.placeholder(tf.float32, name="my_y")

    my_mse = tf.reduce_mean(tf.square(my_pred_y - my_y)) / 2
    # take my_y as 0
    my_ce = tf.reduce_mean(tf.negative(tf.log(1 - my_pred_y)))

    Y = np.linspace(0, 1, num=100)
    M = []
    C = []
    with tf.Session() as sess:
        for val in Y:
            M.append(sess.run(my_mse, feed_dict={my_pred_y: val, my_y: 0}))
            C.append(sess.run(my_ce, feed_dict={my_pred_y: val, my_y: 0}))

    plt.figure(4)
    plt.title("Q3")
    plt.xlabel("predicted y")
    plt.ylabel("loss")
    plt.plot(Y, M, label="mean square error")
    plt.plot(Y, C, label="cross-entropy")
    plt.grid()
    plt.legend()

if __name__ == "__main__":
    print("Q1")
    run_logistic()
    print("Q2")
    run_adam()
    print("Q3")
    run_adam_zl()
    compare_linear_logistic()

    plt.show()
