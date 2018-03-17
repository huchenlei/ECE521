import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math


def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path) / 255
    data = np.reshape(data, [-1, 32 * 32])
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8 * len(rnd_idx))
    validBatch = int(0.1 * len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch], :], \
                                     data[rnd_idx[trBatch + 1:trBatch + validBatch], :], \
                                     data[rnd_idx[trBatch + validBatch + 1:-1], :]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
                                           target[rnd_idx[trBatch + 1:trBatch + validBatch], task], \
                                           target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def make_chunks(data, chunk_size):
    while data.any():
        chunk, data = data[:chunk_size], data[chunk_size:]
        yield chunk


trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("../a1/data/data.npy",
                                                                                         "../a1/data/target.npy", 0)
CLASS_NUM = max(trainTarget) - min(trainTarget) + 1
assert CLASS_NUM == 6

raw_x = tf.placeholder(tf.float32, [None, 32 * 32], name="input_x")
# _x = tf.reshape(raw_x, [-1, 32 * 32])
_x = raw_x
x = tf.concat([tf.ones([tf.shape(_x)[0], 1]), _x], 1)

raw_y = tf.placeholder(tf.int32, [None], name="input_y")
# parse raw_y to valid probability distribution
y = tf.one_hot(raw_y, depth=CLASS_NUM, dtype=tf.float32)

learning_rate = tf.placeholder(tf.float32, name='learning_rate')
lamb = tf.placeholder(tf.float32, name='lambda')

w = tf.Variable(tf.zeros([32 * 32 + 1, CLASS_NUM], dtype=tf.float32))
linear_pred_y = tf.matmul(x, w)
pred_y = tf.nn.softmax(linear_pred_y)

ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=linear_pred_y), name="cross_entropy_loss")
wd = tf.multiply(lamb / 2, tf.reduce_sum(tf.square(w)), name="weight_decay_loss")

loss = ce + wd
accuracy = tf.reduce_mean(tf.to_float(tf.equal(
    tf.argmax(pred_y, axis=1),
    tf.argmax(y, axis=1)
)))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

for LEARNING_RATE in [0.005, 0.001, 0.0001]:
    with tf.Session() as sess:
        train_accs = []
        train_ces = []
        valid_accs = []
        valid_ces = []
        test_accs = []

        ITER_NUM = 18000
        BATCH_SIZE = 300
        # LEARNING_RATE = 0.001  # TODO adjust this val to get better result
        LAMBDA = 0.01

        train_dict = {raw_x: trainData, raw_y: trainTarget, lamb: LAMBDA}
        valid_dict = {raw_x: validData, raw_y: validTarget, lamb: LAMBDA}
        test_dict = {raw_x: testData, raw_y: testTarget, lamb: LAMBDA}

        sess.run(tf.global_variables_initializer())
        ep_range = range(int(math.ceil(ITER_NUM / (len(trainData) / BATCH_SIZE))))
        for _ in ep_range:
            for (chunk_x, chunk_y) in zip(make_chunks(trainData, BATCH_SIZE),
                                          make_chunks(trainTarget, BATCH_SIZE)):
                sess.run(optimizer, feed_dict={raw_x: chunk_x,
                                               raw_y: chunk_y,
                                               learning_rate: LEARNING_RATE,
                                               lamb: LAMBDA})

            train_accs.append(sess.run(accuracy, feed_dict=train_dict))
            train_ces.append(sess.run(loss, feed_dict=train_dict))
            valid_accs.append(sess.run(accuracy, feed_dict=valid_dict))
            valid_ces.append(sess.run(loss, feed_dict=valid_dict))
            test_accs.append(sess.run(accuracy, feed_dict=test_dict))

        plt.figure(0)
        plt.title("Multi-class classification accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.plot(ep_range, train_accs, label="train" + str(LEARNING_RATE))
        plt.plot(ep_range, valid_accs, label="validation" + str(LEARNING_RATE))
        plt.grid()
        plt.legend()

        plt.figure(1)
        plt.title("Multi-class classification cross-entropy")
        plt.xlabel("epoch")
        plt.ylabel("cross-entropy")
        plt.plot(ep_range, train_ces, label="train" + str(LEARNING_RATE))
        plt.plot(ep_range, valid_ces, label="validation" + str(LEARNING_RATE))
        plt.grid()
        plt.legend()

        print("For learning rate %f validation accuracy/loss is %f %f testing accuracy is %f" % (
            LEARNING_RATE, max(valid_accs), min(valid_ces), max(test_accs)))  # 89% worse than binary classification

plt.show()
