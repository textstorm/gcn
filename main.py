
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
  log_dir = os.path.args.log_dir
  train_dir = args.train_dir

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  if not os.path.exists(img_dir):
    os.makedirs(img_dir)

  config_proto = utils.get_config_proto()
  sess = tf.Session(config=config_proto)
  model = GCN(args, sess, name="catvae")
  summary_writer = tf.summary.FileWriter(log_dir)


  for epoch in range(1, args.nb_epoch + 1):
    print "Epoch %d start with learning rate %f temperature %f" % \
        (epoch, model.learning_rate.eval(sess), model.tau.eval(sess))
    print "- " * 50
    epoch_start_time = time.time()
    step_start_time = epoch_start_time
    for i in range(1, total_batch + 1):
      x_batch = celeba.next_batch()

      _, loss, rec_loss, kl, global_step, summaries = model.train(x_batch)
      sess.run(model.tau_decay_op)

      if global_step % args.summary_step == 0:
        summary_writer.add_summary(summaries, global_step)

      if global_step % args.print_step == 0:
        print "epoch %d, step %d, loss %f, rec_loss %f, kl_loss %f, time %.2fs" \
            % (epoch, global_step, loss, rec_loss, kl, time.time()-step_start_time)
        
        step_start_time = time.time()

    if args.anneal and epoch >= args.anneal_start:
      sess.run(model.lr_decay_op)

    if epoch % args.save_epoch == 0:
      x_axis = args.batch_size * args.cat_size
      samples = np.zeros((x_axis, args.cat_range))
      samples[range(x_axis), np.random.choice(args.cat_range, x_axis)] = 1
      samples = np.reshape(samples, [args.batch_size, args.cat_size, args.cat_range])

      x_batch = celeba.next_batch()
      x_recon = model.reconstruct(x_batch)
      x_generate = model.generate(samples, args.batch_size)

      utils.save_images(x_batch, [10, 10], os.path.join(img_dir, "rawImage%s.jpg" % epoch))
      utils.save_images(x_recon, [10, 10], os.path.join(img_dir, "reconstruct%s.jpg" % epoch))
      utils.save_images(x_generate, [10, 10], os.path.join(img_dir, "generate%s.jpg" % epoch))

  model.saver.save(sess, os.path.join(save_dir, "model.ckpt"))
  print "Model stored...."

if __name__ == "__main__":
  args = config.get_args()
  main(args)