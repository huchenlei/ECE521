import tensorflow as tf
import numpy as np
from PIL import Image


# Team Info
# Chenlei Hu 1002030651
# Xin Li     1002371408
# Jiahui Cai 1002061911


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


def distance_func(X, Z):
    r1 = tf.reduce_sum(X * X, 1, keep_dims=True)  # col vec
    r2 = tf.reduce_sum(Z * Z, 1)  # row vec

    D = r1 - 2 * tf.matmul(X, Z, transpose_b=True) + r2
    return D


def pick_majority(labels):
    y, idx, count = tf.unique_with_counts(labels)
    _, indices = tf.nn.top_k(count, 1)  # select max count
    return y[indices[0]]


def knn_predict(dv, train_labels, k):
    top_k_val, top_k_indices = tf.nn.top_k(tf.negative(dv), k)
    top_labels = tf.gather(train_labels, top_k_indices)

    return tf.map_fn(pick_majority, top_labels), top_k_indices


# Load data
trainData, validData, testData, trainTarget, validTarget, testTarget = \
    data_segmentation("./data/data.npy", "./data/target.npy",
                      1)  # change this value to run face(0)/gender(1) classification

train_x = tf.placeholder(tf.float32, [None, 32 * 32], name="input_x")
train_y = tf.placeholder(tf.int32, [None], name="input_y")
new_x = tf.placeholder(tf.float32, [None, 32 * 32], name='new_x')
new_y = tf.placeholder(tf.int32, [None], name='new_y')
k = tf.placeholder(tf.int32, name='k')

pred_y, top_k_y = knn_predict(distance_func(new_x, train_x), train_y, k)

error = tf.reduce_mean(tf.to_float(tf.equal(pred_y, new_y)))

sess = tf.InteractiveSession()

kv = [1, 5, 10, 25, 50, 100, 200]
for kc in kv:
    e_valid = sess.run(error, feed_dict={
        k: kc,
        train_x: trainData,
        train_y: trainTarget,
        new_x: validData,
        new_y: validTarget
    })
    e_test = sess.run(error, feed_dict={
        k: kc,
        train_x: trainData,
        train_y: trainTarget,
        new_x: testData,
        new_y: testTarget
    })
    print("k: %d\t validation accuracy: %f\t test accuracy: %f" % (kc, e_valid, e_test))

error_index = 2  # How many error case to display
error_seen = 0

for i, td in enumerate(testData):
    feed_dict = {
        k: 10,
        train_x: trainData,
        train_y: trainTarget,
        new_x: [td]
    }
    y = sess.run(pred_y, feed_dict=feed_dict)
    top_y_indices = sess.run(top_k_y, feed_dict=feed_dict)

    if y[0] != testTarget[i]:
        error_seen += 1
        print(i, y[0], testTarget[i])

        # Show the input data
        img_arr = np.uint8(td.reshape(32, 32) * 255)
        im = Image.fromarray(img_arr, 'L')
        im.show(title="new_x")

        # show the top 10 nearest
        imgs = Image.new('L', (32 * 10, 32))
        x_offset = 0
        for j, index in enumerate(top_y_indices[0]):
            print("%d: face %d" % (j, trainTarget[index]))

            img = Image.fromarray(np.uint8(trainData[index].reshape(32, 32) * 255))
            imgs.paste(img, (x_offset, 0))
            x_offset += 32

        imgs.show()

        if error_seen == error_index:
            break
