import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Team Info
# Chenlei Hu 1002030651
# Xin Li     1002371408
# Jiahui Cai 1002061911

np.random.seed(521)
Data = np.linspace(1.0, 10.0, num=100)[:, np.newaxis]
Target = np.sin(Data) + 0.1 * np.power(Data, 2) \
         + 0.5 * np.random.randn(100, 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]


def distance_func(X, Z):
    r1 = tf.reduce_sum(X * X, 1, keep_dims=True)  # col vec
    r2 = tf.reduce_sum(Z * Z, 1)  # row vec

    D = r1 - 2 * tf.matmul(X, Z, transpose_b=True) + r2
    return D


def responsibility(dv, k):
    row_dv = tf.shape(dv)[0]
    top_k_val, top_k_indices = tf.nn.top_k(tf.negative(dv), k)

    # Convert top_k_indices to the format fitting the need in sparse_to_dense
    top_k_indices = tf.expand_dims(top_k_indices, 2)
    # broadcast col vector to a certain shape
    index_matrix = tf.reshape(tf.range(row_dv), [-1, 1, 1]) + tf.zeros(tf.shape(top_k_indices), dtype=tf.int32)

    index_pair_matrix = tf.reshape(tf.concat([index_matrix, top_k_indices], axis=2), [-1, 2])
    return tf.sparse_to_dense(
        sparse_indices=index_pair_matrix,
        output_shape=tf.shape(dv),
        default_value=0,
        sparse_values=tf.to_float(1 / k),
        validate_indices=False
    )


def knn_predict(r, train_y):
    return tf.matmul(r, train_y)


train_x = tf.placeholder(tf.float32, [None, 1], name="input_x")
train_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
new_x = tf.placeholder(tf.float32, [None, 1], name='new_x')
new_y = tf.placeholder(tf.float32, [None, 1], name='new_y')
k = tf.placeholder(tf.int32, name='k')

pred_y = knn_predict(responsibility(distance_func(new_x, train_x), k), train_y)

MSE = tf.reduce_mean(tf.reduce_sum((pred_y - new_y) ** 2, 1)) / 2

X = np.linspace(0.0, 11.0, num=1000)[:, np.newaxis]

sess = tf.InteractiveSession()

ks = [1, 3, 5, 50]

for i, kc in enumerate(ks):
    validation_mse = sess.run(MSE, feed_dict={
        train_x: trainData,
        train_y: trainTarget,
        new_x: validData,
        new_y: validTarget,
        k: kc
    })

    test_mse = sess.run(MSE, feed_dict={
        train_x: trainData,
        train_y: trainTarget,
        new_x: testData,
        new_y: testTarget,
        k: kc
    })

    train_mse = sess.run(MSE, feed_dict={
        train_x: trainData,
        train_y: trainTarget,
        new_x: trainData,
        new_y: trainTarget,
        k: kc
    })

    print("k = %d\nvalidation MSE: %f, test MSE %f, train MSE %f" % (kc, validation_mse, test_mse, train_mse))

    y = sess.run(pred_y, feed_dict={
        train_x: trainData,
        train_y: trainTarget,
        new_x: X,
        k: kc
    })

    plt.figure(i)
    plt.plot(trainData, trainTarget, 'ro')
    plt.plot(X, y, 'b')
    plt.title("knn regression, k = %d" % kc)
    plt.show()

