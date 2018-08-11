
import tensorflow as tf
import numpy as np
import utils
import config
import time
import os

from model import GCN
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
  #
  save_dir = args.save_dir
  log_dir = args.log_dir
  train_dir = args.data_dir

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  config_proto = utils.get_config_proto()
  sess = tf.Session(config=config_proto)
  model = GCN(args, sess, name="gcn")
  summary_writer = tf.summary.FileWriter(log_dir)

  adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = utils.load_data(args.data_type)
  features = utils.preprocess_features(features)
  support = [preprocess_adj(adj)]
  args.num_supports = 1
  args.input_size, args.features_size = features[2][1], features[2]
  args.output_size = y_train.shape[1]

  for epoch in range(1, args.nb_epoch + 1):
    print "Epoch %d start with learning rate" % (epoch, model.learning_rate.eval(sess))
    print "- " * 50
    epoch_start_time = time.time()

    feed_dict = construct_feed_dict(model, features, support, y_train, train_mask)
    _, loss, accuracy, summaries = model.train(x_batch, feed_dict)

    if epoch % args.summary_step == 0:
      summary_writer.add_summary(summaries, epoch)

    if epoch % args.print_step == 0:
      print "epoch %d, loss %f, time %.2fs" % (epoch, loss, time.time()-step_start_time)

    if args.anneal and epoch >= args.anneal_start:
      sess.run(model.lr_decay_op)

  model.saver.save(sess, os.path.join(save_dir, "model.ckpt"))
  print "Model stored...."

if __name__ == "__main__":
  args = config.get_args()
  main(args)