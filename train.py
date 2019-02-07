import tensorflow as tf
import data_reader
import model_builder
import os
from functools import partial

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 1, '')
flags.DEFINE_string("logs_dir", "train_logs/5_my_contextual_attention-soft_attention_google", "")
flags.DEFINE_string("start_checkpoint", None, "")
flags.DEFINE_integer("log_steps_n", 5000, "")  # 250


def main(argv):
    params = {'batch_size': FLAGS.batch_size,
              'image_size': data_reader.image_size,
              'float_type': tf.float32,
              'gradient_scale': 1.0,  # Used during fp16 training.
              'leaky_relu_alpha': 0.2,
              'learning_rate': 0.0001,
              'num_steps_start_lr_decay': 500000,
              'max_iter': 1000000,
              'quantize': False}

    train_input_fn = partial(data_reader.train_input_fn, params=params, batch_size=FLAGS.batch_size)
    eval_input_fn = partial(data_reader.eval_input_fn, params=params, batch_size=FLAGS.batch_size)

    sess_config = tf.ConfigProto()
    # sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    # train_distribute = tf.contrib.distribute.MirroredStrategy()
    train_distribute = None

    config = tf.estimator.RunConfig(
        model_dir=None,
        tf_random_seed=None,
        save_summary_steps=FLAGS.log_steps_n,
        save_checkpoints_secs=60 * 60 * 2,
        session_config=sess_config,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=FLAGS.log_steps_n,
        train_distribute=train_distribute
    )

    estimator = tf.estimator.Estimator(model_fn=model_builder.model_fn,
                                       model_dir=os.path.join(FLAGS.logs_dir, "checkpoints"),
                                       params=params,
                                       config=config)

    psnr = 0.0
    while True:
        print("Training...")
        estimator.train(train_input_fn)

        print("Evaluating...")
        eval_result = estimator.evaluate(eval_input_fn)
        print("Finished evaluating!")
        print('psnr: {}'.format(eval_result['psnr']))
        print('ssim: {}'.format(eval_result['ssim']))

        if eval_result['psnr'] > psnr:
            psnr = eval_result['psnr']
            estimator.export_savedmodel(
                export_dir_base=os.path.join(FLAGS.logs_dir, 'SavedModels'),
                serving_input_receiver_fn=lambda: model_builder.serving_input_receiver_fn(params['float_type']),
                strip_default_attrs=True)
            print('Saved new SavedModel with:')
            print('psnr: {}'.format(eval_result['psnr']))
            print('ssim: {}'.format(eval_result['ssim']))

        print()


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run(main)
