import tensorflow as tf
import numpy as np
import utils

def glorot(shape, name=None):
  init_range = np.sqrt(6.0/(shape[0]+shape[1]))
  initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
  return tf.Variable(initial, name=name)

def zeros(shape, name=None):
  initial = tf.zeros(shape, dtype=tf.float32)
  return tf.Variable(initial, name=name)

def sparse_dropout(x, dropout, noise_shape):
  random_tensor = 1.0 - dropout
  random_tensor += tf.random_uniform(noise_shape)
  dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
  pre_out = tf.sparse_retain(x, dropout_mask)
  return pre_out * (1./(1.0-dropout))

class GCN(object):
  def __init__(self, args, sess, name="gcn"):
    self.input_size = args.input_size
    self.output_size = args.output_size
    self.num_supports = args.num_supports
    self.features_size = args.features_size
    self.hidden_size = args.hidden_size
    self.num_labels = args.output_size
    self.sess = sess
    self.max_grad_norm = args.max_grad_norm
    self.dropout = args.dropout
    self.learning_rate = tf.Variable(float(args.learning_rate), trainable=False, name="learning_rate")
    self.lr_decay_op = self.learning_rate.assign(tf.multiply(self.learning_rate, args.lr_decay))
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    with tf.name_scope("data"):
      self.support = [tf.sparse_placeholder(tf.float32) for _ in range(self.num_supports)]
      self.features = tf.sparse_placeholder(tf.float32, shape=self.features_size)
      self.labels = tf.placeholder(tf.float32, [None, self.num_labels])
      self.labels_mask = tf.placeholder(tf.int32)
      self.num_features_nonzero = tf.placeholder(tf.int32) 

    with tf.name_scope("gcn"):
      outputs = self.graph_convolution(self.features, self.input_size, self.hidden_size, sparse_inputs=True)
      outputs = self.graph_convolution(outputs, self.hidden_size, self.output_size, act=lambda x:x)
      self.outputs = outputs

    with tf.name_scope("loss"):
      self.loss = utils.masked_softmax_cross_entropy(self.outputs, self.labels, self.labels_mask)
      tf.summary.scalar("loss", self.loss)

    with tf.name_scope("accuracy"):
      self.accuracy = utils.masked_accuracy(self.outputs, self.labels, self.labels_mask)

    with tf.name_scope("train"):
      self.train_op = self.optimizer.minimize(self.loss)

    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)
    self.summary = tf.summary.merge_all()
    self.saver = tf.train.Saver(tf.global_variables())

  def graph_convolution(self, inputs, in_size, out_size, act=tf.nn.relu, sparse_inputs=False):
    x = inputs
    x = self.dropout_func(x, self.dropout, sparse=sparse_inputs)
    supports = []
    variables = {}
    for i in range(self.num_supports):
      variables["weights_" + str(i)] = glorot([in_size, out_size], "weights_i" + str(i))
    variables['bias'] = zeros([out_size], name='bias')
    for i in range(self.num_supports):
      x = self.dot(x, variables["weights_" + str(i)], sparse=sparse_inputs)
      x = self.dot(self.support[i], x, sparse=True)
      supports.append(x)
    output = tf.add_n(supports)
    output += variables['bias']
    return act(output)

  def dot(self, x, y, sparse=False):
    if sparse:
      return tf.sparse_tensor_dense_matmul(x, y)
    else:
      return tf.matmul(x, y)

  def dropout_func(self, x, dropout, sparse=False):
    if sparse:
      return sparse_dropout(x, dropout, self.num_features_nonzero)
    else:
      return tf.nn.dropout(x, keep_prob=1.0-dropout)

  def predict(self):
    return tf.nn.softmax(self.outputs)

  def train(self, feed_dict):
    return self.sess.run([self.train_op, self.loss, self.accuracy, self.summary], feed_dict=feed_dict)

  def trainable_vars(self, scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
