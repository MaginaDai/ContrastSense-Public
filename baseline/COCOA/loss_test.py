import tensorflow as tf
import numpy as np

temperature = 0.5
lambd = 3.9e-3
scale_loss= 1/32

def loss(ytrue, ypred):
        batch_size, dim_size = ypred.shape[1], ypred.shape[0]
        # Positive Pairs
        pos_error = []
        for i in range(batch_size):
            sim = tf.linalg.matmul(ypred[:, i, :], ypred[:, i, :], transpose_b=True)
            sim = tf.subtract(tf.ones([dim_size, dim_size], dtype=tf.dtypes.float32), sim)
            sim = tf.exp(sim/temperature)
            pos_error.append(tf.reduce_mean(sim))
        # Negative pairs
        neg_error = 0
        for i in range(dim_size):
            sim = tf.cast(tf.linalg.matmul(ypred[i], ypred[i], transpose_b=True), dtype=tf.dtypes.float32)
            sim = tf.exp(sim /temperature)
            # sim = tf.add(sim, tf.ones([batch_size, batch_size]))
            tri_mask = np.ones(batch_size ** 2, dtype=np.bool).reshape(batch_size, batch_size)
            tri_mask[np.diag_indices(batch_size)] = False
            off_diag_sim = tf.reshape(tf.boolean_mask(sim, tri_mask), [batch_size, batch_size - 1])
            neg_error += (tf.reduce_mean(off_diag_sim, axis=-1))

        error = tf.multiply(tf.reduce_sum(pos_error), scale_loss) + lambd * tf.reduce_sum(neg_error)

        return error


if __name__ == '__main__':
    ytrue = 0
    ypred = t = tf.constant(
	[
	[[1, 2], [3, 4]],
	[[5, 6], [7, 8]],
	[[9, 10],[11, 12]]
	], tf.float32
	)*1e-2
    error = loss(ytrue, ypred)
    print(error)