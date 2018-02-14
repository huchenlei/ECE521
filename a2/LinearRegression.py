import numpy as np
import tensorflow as tf


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


trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()

raw_x = tf.placeholder(tf.float32, [None, 28, 28], name="input_x")
x = tf.reshape(raw_x, [-1, 28 * 28])
y = tf.placeholder(tf.float32, [None, 1], name="input_y")

learning_rate = tf.placeholder(tf.float32, name='learning_rate')
lamb = tf.placeholder(tf.float32, name='lambda')

w = tf.Variable(np.zeros((28 * 28, 1)), dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

pred_y = tf.add(tf.matmul(x, w), b)
mse = tf.reduce_mean(tf.reduce_sum((pred_y - y) ** 2, 1)) / 2
wd = tf.multiply(lamb, w ** 2) / 2

loss = mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

for rate in [0.005, 0.001, 0.0001]:
    sess.run(init)
    sess.run(train, feed_dict={
        raw_x: trainData,
        y: trainTarget,
        learning_rate: rate,
        lamb: 0
    })

    print(sess.run(loss, feed_dict={
        raw_x: validData,
        y: validTarget,
        learning_rate: rate,
        lamb: 0
    }))

