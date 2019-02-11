import tensorflow as tf
import data_reader
from conv2d_spectral_norm import conv2d_spectral_norm
from coord_conv import AddCoords

edges_discriminator_name = 'edges_discriminator'
images_discriminator_name = 'image_discriminator'

def float2int(float_image):
    return tf.cast(tf.clip_by_value((float_image + 1) * 127.0, 0, 255), dtype=tf.uint8)
    # return tf.cast(tf.clip_by_value(float_image, 0.0, 255.0), dtype=tf.uint8)


def int2float(int_image):
    return tf.cast(int_image, tf.float32) * (2.0 / 255.0) - 1.0
    # return tf.cast(int_image, tf.float32)


def print_tensor(tensor, comment):
    return tensor
    # return tf.Print(tensor, [tensor], comment, summarize=6)


def activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


def gradients_with_loss_scaling(loss, variables, loss_scale):
    """Gradient calculation with loss scaling to improve numerical stability
    when training with float16.
    """
    grad_scaled = []
    for grad in tf.gradients(loss * loss_scale, variables):
        grad = grad / loss_scale
        grad = tf.Print(grad, [tf.reduce_max(grad)], 'grad')
        grad_scaled.append(grad)
    return grad_scaled
    # return [grad / loss_scale for grad in tf.gradients(loss * loss_scale, variables)]


def serving_input_receiver_fn(float_type):
    """Build the serving inputs."""
    # The outer dimension (None) allows us to batch up inputs for
    # efficiency. However, it also means that if we want a prediction
    # for a single instance, we'll need to wrap it in an outer list.
    inputs = {"i_in": tf.placeholder(shape=[1, data_reader.image_size[0],
                                            data_reader.image_size[1],
                                            3],
                                     dtype=float_type),
              "mask": tf.placeholder(shape=[1, data_reader.image_size[0],
                                            data_reader.image_size[1],
                                            1],
                                     dtype=float_type),
              "edges": tf.placeholder(shape=[1, data_reader.image_size[0],
                                             data_reader.image_size[1],
                                             1],
                                      dtype=float_type)
              }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def model_fn(features, labels, mode, params):
    i_in = features['i_in']
    mask = features['mask']
    edges = features['edges']

    i_out, i_edges = build_generator(params, i_in, mask, edges, mode)

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {'predict_output': tf.estimator.export.PredictOutput({"i_out": i_out})}
        predictions = {'i_out': i_out}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    # Compute loss.
    i_gt = features['i_gt']

    edges_l1_loss = tf.reduce_mean(tf.abs(i_edges - i_gt))
    images_l1_loss = tf.reduce_mean(tf.abs(i_out - i_gt))

    discriminator_edges_data = build_discriminator(edges_discriminator_name, edges)
    discriminator_edges_z = build_discriminator(edges_discriminator_name, i_edges)

    loss_discriminator_edges, loss_generator_edges_adversarial = hinge_gan_loss(discriminator_edges_data, discriminator_edges_z)

    # Discriminator of real images
    tf.print(tf.shape(i_gt))
    tf.print(tf.shape(mask))
    tf.print(tf.shape(edges))
    discriminator_data = build_discriminator(images_discriminator_name, tf.concat([i_gt, mask, edges], axis=3))
    # Discriminator of generated images
    discriminator_z = build_discriminator(images_discriminator_name, tf.concat([i_out, mask, edges], axis=3))

    loss_discriminator_images, loss_generator_adversarial = hinge_gan_loss(discriminator_data, discriminator_z)
    # loss_discriminator_images, loss_generator_adversarial = ra_hinge_gan_loss(discriminator_data, discriminator_z)

    loss_generator = edges_l1_loss + images_l1_loss + loss_generator_adversarial + loss_generator_edges_adversarial

    loss = loss_generator

    tf.summary.scalar('loss_discriminator_edges', loss_discriminator_edges)
    tf.summary.scalar('loss_discriminator_images', loss_discriminator_images)
    tf.summary.scalar('loss_generator_edges_l1', edges_l1_loss)
    tf.summary.scalar('loss_generator_images_l1', images_l1_loss)
    tf.summary.scalar('loss_generator_adversarial', loss_generator_adversarial)

    tf.print(tf.shape(edges))

    summary_image = tf.summary.image('image', tf.concat(
        [float2int(i_gt),
         float2int(tf.image.grayscale_to_rgb(edges)),
         float2int(i_in),
         float2int(tf.image.grayscale_to_rgb(i_edges)),
         float2int(i_out)], axis=1))

    if mode == tf.estimator.ModeKeys.EVAL:
        i_out = (i_out + 1) / 2.
        i_gt = (i_gt + 1) / 2.
        # Compute evaluation metrics.
        psnr = tf.image.psnr(i_out, i_gt, 1.0)
        mean_pnsr = tf.metrics.mean(psnr)

        ssim = tf.image.ssim(i_out, i_gt, 1.0)
        mean_ssim = tf.metrics.mean(ssim)

        mean_edges_l1_loss = tf.metrics.mean(edges_l1_loss)
        mean_images_l1_loss = tf.metrics.mean(images_l1_loss)
        mean_generator_adversarial_loss = tf.metrics.mean(loss_generator_adversarial)
        mean_disriminator_edges_loss = tf.metrics.mean(loss_discriminator_edges)
        mean_disriminator_images_loss = tf.metrics.mean(loss_discriminator_images)

        metrics = {'psnr': mean_pnsr,
                   'ssim': mean_ssim,
                   'loss_discriminator_edges': mean_disriminator_edges_loss,
                   'loss_discriminator_images': mean_disriminator_images_loss,
                   'loss_generator_edges_l1': mean_edges_l1_loss,
                   'loss_generator_images_l1': mean_images_l1_loss,
                   'loss_generator_adversarial': mean_generator_adversarial_loss}

        tf.summary.scalar('psnr', mean_pnsr[0])
        tf.summary.scalar('ssim', mean_ssim[0])

        if params['quantize']:
            tf.contrib.quantize.create_eval_graph()

        # summary_image_hook = tf.train.SummarySaverHook(save_steps=save_steps,
        #                                          output_dir='summary',
        #                                          summary_op=tf.summary.merge([summary_image]))

        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    if params['quantize']:
        tf.contrib.quantize.create_training_graph(quant_delay=200000)

    global_step = tf.train.get_global_step()

    with tf.name_scope('learning_rate_scheduler'):
        lr_multiplier = tf.maximum(
            1.0 - tf.cast(
                tf.maximum(tf.constant(0, dtype=tf.int64),
                           global_step - params['num_steps_start_lr_decay']),
                dtype=tf.float32) /
            tf.cast(params['max_iter'] - params['num_steps_start_lr_decay'] + 1, dtype=tf.float32),
            0)
        learning_rate = params['learning_rate'] * lr_multiplier
    tf.summary.scalar('learning_rate', learning_rate)

    g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)
    d_optimizer = g_optimizer

    g_vars = tf.trainable_variables('generator/')
    d_vars = tf.trainable_variables(edges_discriminator_name + '/')
    d_vars += tf.trainable_variables(images_discriminator_name + '/')

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        if params['float_type'] == tf.float16:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            grads = gradients_with_loss_scaling(loss, variables, params['gradient_scale'])
            train_op = optimizer.apply_gradients(zip(grads, variables), global_step=global_step)
        else:
            train_op_d = d_optimizer.minimize(loss_discriminator_images, var_list=d_vars,
                                              global_step=global_step)
            train_op_g = g_optimizer.minimize(loss_generator, var_list=g_vars,
                                              global_step=global_step)

    class TrainOpSelectorHook(tf.train.SessionRunHook):

        def begin(self):
            self.step = 0

        def before_run(self, run_context):
            print(self.step)
            if self.step % 2 == 0:
                # Train D
                print('D')
                return tf.train.SessionRunArgs({'global_step': global_step, 'train_op': train_op_d})
            else:
                # Train G
                print('G')
                return tf.train.SessionRunArgs({'global_step': global_step, 'train_op': train_op_g})

        def after_run(self, run_context, run_values):
            self.step = run_values.results['global_step']

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=tf.no_op(),
                                      training_hooks=[TrainOpSelectorHook()])


def get_initializer():
    """He normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    Arguments:
        seed: A Python integer. Used to seed the random generator.
    Returns:
        An initializer.
    References:
        He et al., http://arxiv.org/abs/1502.01852
    """
    return tf.initializers.variance_scaling(scale=2., mode='fan_in', distribution='normal')


def hinge_gan_loss(discriminator_data, discriminator_z):
    loss_discriminator_data = tf.reduce_mean(tf.nn.relu(1 - discriminator_data))
    loss_discriminator_z = tf.reduce_mean(tf.nn.relu(1 + discriminator_z))
    loss_discriminator = (loss_discriminator_data + loss_discriminator_z)

    loss_generator_adversarial = -tf.reduce_mean(discriminator_z)
    return loss_discriminator, loss_generator_adversarial


def ra_hinge_gan_loss(discriminator_data, discriminator_z):
    # This is relativistic average hinge gan loss as in https://arxiv.org/pdf/1807.00734.pdf C.7 eq. 33, 34
    d_r = discriminator_data - tf.reduce_mean(discriminator_z, axis=0)
    d_z = discriminator_z - tf.reduce_mean(discriminator_data, axis=0)

    loss_discriminator_data = tf.reduce_mean(tf.nn.relu(1 - d_r))
    loss_discriminator_z = tf.reduce_mean(tf.nn.relu(1 + d_z))
    loss_discriminator = (loss_discriminator_data + loss_discriminator_z)  # / 2

    loss_generator_adversarial = (tf.reduce_mean(tf.nn.relu(1 - d_z)) + tf.reduce_mean(tf.nn.relu(1 + d_r)))  # / 2

    return loss_discriminator, loss_generator_adversarial


def normalize(input_tensor, training):
    renorm_params = {'rmax': 1.5,
                     'rmin': 0.0,
                     'dmax': 0.5}
    # input_tensor = tf.cast(input_tensor, tf.float32)
    # normalized = tf.layers.batch_normalization(input_tensor, training=training, trainable=training, fused=False)
    # normalized = tf.cast(normalized, tf.float16)
    # return normalized
    # return input_tensor
    # return tf.layers.batch_normalization(input_tensor, training=training, trainable=training, fused=True, renorm=True,
    #                                      renorm_clipping=renorm_params)
    return tf.layers.batch_normalization(input_tensor, training=training, trainable=training, fused=True, renorm=True)
    # return tf.layers.batch_normalization(input_tensor, training=training, trainable=training, fused=True)
    # return group_normalization(input_tensor, 16)


def group_normalization(input_tensor, num_groups, gamma=1.0, beta=0.0, epsilon=1e-5):
    channels_int = input_tensor.get_shape().as_list()[3]
    while channels_int % num_groups != 0 and num_groups != 0:
        num_groups -= 1

    batch, height, width, channels = input_tensor.shape
    input_tensor = tf.reshape(input_tensor, shape=(batch, height, width, channels // num_groups, num_groups))
    mean, var = tf.nn.moments(input_tensor, [1, 2, 3], keep_dims=True)
    input_tensor = (input_tensor - mean) / tf.sqrt(var + epsilon)
    input_tensor = tf.reshape(input_tensor, [batch, height, width, channels])
    return input_tensor * gamma + beta


def gated_convolution(input_tensor, kernel_size, depth, stride, dilation_rate, activation=None):
    # assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    # if padding == 'SYMMETRIC' or padding == 'REFELECT':
    #     p = int(rate * (ksize - 1) / 2)
    #     x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode=padding)
    #     padding = 'VALID'

    # TODO: add mirror padding as in https://arxiv.org/pdf/1801.07892.pdf
    # TODO: in https://arxiv.org/pdf/1801.07892.pdf was used ELU instead of ReLU
    padding = 'SAME'
    # net = tf.layers.separable_conv2d(net, depth, kernel_size, strides=stride, padding=padding, use_bias=False)
    # input_tensor_shape = tf.shape(input_tensor)
    # input_tensor = AddCoords(x_dim=input_tensor_shape[2],
    #                          y_dim=input_tensor_shape[1],
    #                          with_r=True)(input_tensor)

    gating = tf.contrib.layers.conv2d(
        input_tensor,
        depth,
        kernel_size,
        stride=stride,
        padding=padding,
        rate=dilation_rate,
        activation_fn=tf.nn.sigmoid
    )
    features = tf.contrib.layers.conv2d(
        input_tensor,
        depth,
        kernel_size,
        stride=stride,
        padding=padding,
        rate=dilation_rate,
        activation_fn=activation,
        weights_initializer=tf.initializers.variance_scaling()
    )

    features = features * gating
    return features


def regular_convolution(input_tensor, kernel_size, depth, stride, dilation_rate, activation=None):
    # assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    # if padding == 'SYMMETRIC' or padding == 'REFELECT':
    #     p = int(rate * (ksize - 1) / 2)
    #     x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode=padding)
    #     padding = 'VALID'

    # TODO: add mirror padding as in https://arxiv.org/pdf/1801.07892.pdf
    # TODO: in https://arxiv.org/pdf/1801.07892.pdf was used ELU instead of ReLU
    padding = 'SAME'
    # net = tf.layers.separable_conv2d(net, depth, kernel_size, strides=stride, padding=padding, use_bias=False)

    # input_tensor_shape = tf.shape(input_tensor)
    # input_tensor = AddCoords(x_dim=input_tensor_shape[2],
    #                          y_dim=input_tensor_shape[1],
    #                          with_r=True)(input_tensor)

    features = tf.contrib.layers.conv2d(
        input_tensor,
        depth,
        kernel_size,
        stride=stride,
        padding=padding,
        rate=dilation_rate,
        activation_fn=activation,
        weights_initializer=tf.initializers.variance_scaling()
    )
    return features


# Taken from https://github.com/taki0112/Spectral_Normalization-Tensorflow
def spectral_norm(w, iteration=1):
    # Placing l2_norm here since we will not need it anywhere else
    def l2_norm(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def sn_conv2d(inputs,
              filters,
              kernel_size,
              strides=1,
              padding='SAME',
              dilation_rate=1,
              use_bias=True,
              activation=None,
              kernel_initializer=None,
              name=None):
    with tf.variable_scope(name):
        w = tf.get_variable("kernel",
                            shape=[kernel_size, kernel_size, inputs.get_shape()[-1], filters],
                            initializer=kernel_initializer)

        conv = tf.nn.conv2d(input=inputs,
                            filter=spectral_norm(w),
                            padding=padding,
                            strides=[1, strides, strides, 1],
                            dilations=[1, dilation_rate, dilation_rate, 1])
        if use_bias:
            b = tf.get_variable("bias",
                                [filters],
                                initializer=tf.zeros_initializer)
            conv += b

        if activation is not None:
            conv = activation(conv)
    return conv


def upsample_x2(input_tensor):
    scale = 2
    shape = tf.shape(input_tensor)
    return tf.image.resize_nearest_neighbor(input_tensor, [shape[1] * scale, shape[2] * scale])


def resize(x, scale=2., to_shape=None, align_corners=True, dynamic=False,
           func=tf.image.resize_bilinear, name='resize'):
    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1] * scale, tf.int32),
                  tf.cast(xs[2] * scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1] * scale), int(xs[2] * scale)]
    with tf.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]],
                     align_corners=align_corners)
    return x


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1] + [3])
        img = img / 127.5 - 1.
        return img


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def highlight_flow_tf(flow, name='flow_to_image'):
    """Tensorflow ops for highlight flow.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(highlight_flow, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1] + [3])
        img = img / 127.5 - 1.
        return img


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.
    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Args:
        f: Input feature to match (foreground).
        b: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.
    Returns:
        tf.Tensor: output
    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2 * rate
    raw_w = tf.extract_image_patches(
        b, [1, kernel, kernel, 1], [1, rate * stride, rate * stride, 1], [1, 1, 1, 1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1. / rate, func=tf.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1] / rate), int(raw_int_bs[2] / rate)],
               func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        # mask = resize(mask, scale=1. / rate, func=tf.image.resize_nearest_neighbor
        mask = resize(mask, to_shape=[int(raw_int_bs[1] / rate), int(raw_int_bs[2] / rate)],
                      func=tf.image.resize_nearest_neighbor)  # https://github.com/tensor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    # m = tf.reshape(m, [int_fs, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0, 1, 2], keepdims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0, 1, 2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1, 1, 1, 1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1] * fs[2], bs[1] * bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1] * fs[2], bs[1] * bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1] * bs[2]])

        # softmax to match
        yi = yi * mm  # mask
        yi = tf.nn.softmax(yi * scale, 3)
        yi *= mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0),
                                    strides=[1, rate, rate, 1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.image.resize_nearest_neighbor)
    return y, flow


def non_local_block_2d(input_tensor, mask, pairwise_func='gaussian', spatial_subsampling_factor=2,
                       embedding_depth_subsampling_factor=2):
    # pairwise_func: 'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
    input_tensor_shape = tf.shape(input_tensor)
    input_tensor_depth = input_tensor.get_shape().as_list()[3]
    batch_size = input_tensor_shape[0]

    g = tf.layers.conv2d(input_tensor,
                         filters=input_tensor_depth // embedding_depth_subsampling_factor,
                         kernel_size=1,
                         use_bias=False)
    g = tf.reshape(g, [batch_size, -1, input_tensor_depth // embedding_depth_subsampling_factor])

    if pairwise_func == 'gaussian':
        x_i = tf.reshape(input_tensor, [batch_size, -1, input_tensor_depth])
        x_j = x_i
        f = tf.exp(tf.matmul(x_i, x_j, transpose_b=True))
        c = tf.reduce_sum(f, axis=[2])

    elif pairwise_func == 'embedded_gaussian':
        theta = tf.layers.conv2d(input_tensor,
                                 filters=input_tensor_depth // embedding_depth_subsampling_factor,
                                 kernel_size=1,
                                 use_bias=False)
        theta = tf.reshape(theta,
                           shape=[batch_size, -1, input_tensor_depth // embedding_depth_subsampling_factor])

        phi = tf.layers.conv2d(input_tensor,
                               filters=input_tensor_depth // embedding_depth_subsampling_factor,
                               kernel_size=1,
                               use_bias=False)
        phi = tf.reshape(phi, [batch_size, -1, input_tensor_depth // embedding_depth_subsampling_factor])
        phi = tf.transpose(phi, perm=[0, 2, 1])
        f = tf.matmul(theta, phi)

        # f = tf.exp(f)
        # c = tf.reduce_sum(f, axis=[2], keepdims=True)
        # f /= c
        f = tf.nn.softmax(f, axis=2)
        # f = tf.Print(f, [tf.reduce_max(f)], 'max f')

    elif pairwise_func == 'dot_product':
        theta = tf.layers.conv2d(input_tensor,
                                 filters=input_tensor_depth // embedding_depth_subsampling_factor,
                                 kernel_size=1,
                                 use_bias=False)
        theta = tf.reshape(theta,
                           shape=[batch_size, -1, input_tensor_depth // embedding_depth_subsampling_factor])

        phi = tf.layers.conv2d(input_tensor,
                               filters=input_tensor_depth // embedding_depth_subsampling_factor,
                               kernel_size=1,
                               use_bias=False)
        phi = tf.reshape(phi, [batch_size, -1, input_tensor_depth // embedding_depth_subsampling_factor])
        phi = tf.transpose(phi, perm=[0, 2, 1])
        f = tf.matmul(theta, phi)
        c = input_tensor_shape[1] * input_tensor_shape[2]
        c = tf.reshape(c, shape=[batch_size])
    elif pairwise_func == 'concatenation':
        # TODO: Finish implementation
        theta = tf.layers.conv2d(input_tensor,
                                 filters=input_tensor_depth // embedding_depth_subsampling_factor,
                                 kernel_size=1,
                                 use_bias=False)
        theta = tf.reshape(theta,
                           shape=[batch_size, -1, input_tensor_depth // embedding_depth_subsampling_factor])

        phi = tf.layers.conv2d(input_tensor,
                               filters=input_tensor_depth // embedding_depth_subsampling_factor,
                               kernel_size=1,
                               use_bias=False)
        phi = tf.reshape(phi, [batch_size, -1, input_tensor_depth // embedding_depth_subsampling_factor])
        concatenated = tf.concat([theta, phi], axis=2)
        # This is probaly wrong move coz we should have (w_f)T - transposed weight vector. Look for details in
        # Relation Networks for visual reasoning paper.
        f = tf.layers.conv1d(concatenated, 1, 1, use_bias=False, activation=tf.nn.relu)
        raise NotImplementedError('Concatenation is not implemented yet. Use other type of `pairwise_func`')
    else:
        raise ValueError('Wrong value of argument `pairwise_func`')

    y = tf.matmul(f, g)
    # y /= c
    # y = tf.Print(y, [tf.reduce_max(g)], 'max g')
    # y = tf.Print(y, [tf.reduce_max(y)], 'max normalized y')
    y = tf.reshape(y, shape=[batch_size,
                             input_tensor_shape[1],
                             input_tensor_shape[2],
                             input_tensor_depth // embedding_depth_subsampling_factor])

    y = tf.layers.conv2d(y,
                         filters=input_tensor_depth,
                         kernel_size=1,
                         use_bias=False,
                         kernel_initializer=tf.initializers.zeros)
    # for i in range(0, 20):
    #     one_channel_attention = f[:, :, i]
    #     one_channel_attention = tf.reshape(one_channel_attention,
    #                                        shape=[batch_size, input_tensor_shape[1], input_tensor_shape[2], 1])
    #     one_channel_attention -= tf.reduce_min(one_channel_attention, axis=[1, 2, 3])
    #     one_channel_attention /= tf.reduce_max(one_channel_attention, axis=[1, 2, 3])
    #     tf.summary.image('attention_channel_{}'.format(i), one_channel_attention)
    #
    # # y = tf.Print(y, [tf.shape(f)], 'shape of f')
    #
    # activation_summary(y)
    # activation_summary(tf.subtract(y, input_tensor, name='y_and_input_tensor_difference'))
    return y + input_tensor


# def custom_context_attenion(input_tensor, mask):
#     mask = tf.image.resize_images(mask, tf.shape(input_tensor)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     input_tensor_shape = tf.shape(input_tensor)
#     input_tensor_depth = input_tensor.get_shape().as_list()[3]
#     batch_size = input_tensor_shape[0]
#
#     input_tensor = tf.reshape(input_tensor, [batch_size, -1, input_tensor_depth])
#     mask = tf.reshape(mask, [batch_size, -1, 1])
#     mask = tf.transpose(mask, perm=[0, 2, 1])
#     f = tf.matmul(input_tensor, input_tensor, transpose_b=True)
#     f *= mask
#
#     # TODO: Необходимо транспонировать маску, т.к. она зауляет строки
#     # TODO: Заменить hard attention на softmax(f*10)
#     # TODO: Add 1-eye mat as mask to prevent self-attention
#     argmax = tf.argmax(f, axis=-1, output_type=tf.int32)
#     one_hot_mask = tf.one_hot(indices=argmax,
#                               depth=input_tensor_shape[1] * input_tensor_shape[2],
#                               axis=-1,
#                               dtype=input_tensor.dtype)
#     y = tf.matmul(one_hot_mask, input_tensor)
#     y = tf.reshape(y, shape=[batch_size,
#                              input_tensor_shape[1],
#                              input_tensor_shape[2],
#                              input_tensor_depth])
#     return y

def custom_context_attenion(input_tensor, mask):
    mask = tf.image.resize_images(mask, tf.shape(input_tensor)[1:3], method=tf.image.ResizeMethod.BILINEAR)
    input_tensor_shape = tf.shape(input_tensor)
    input_tensor_depth = input_tensor.get_shape().as_list()[3]
    batch_size = input_tensor_shape[0]

    inv_identity_mask = 1 - tf.eye(input_tensor.shape[1].value, input_tensor.shape[2].value,
                                   [input_tensor.shape[0].value])
    inv_identity_mask = tf.expand_dims(inv_identity_mask, axis=-1)
    mask *= inv_identity_mask
    input_tensor = tf.reshape(input_tensor, [batch_size, -1, input_tensor_depth])
    mask = tf.reshape(mask, [batch_size, -1, 1])
    f = tf.matmul(input_tensor, input_tensor, transpose_b=True)
    scores = tf.nn.softmax(f * mask * 10) * mask
    tf.summary.image('attention_scores', tf.expand_dims(scores, axis=3))
    tf.summary.image('attention_scores', tf.expand_dims(scores, axis=3))
    y = tf.matmul(scores, input_tensor)
    y = tf.reshape(y, shape=[batch_size,
                             input_tensor_shape[1],
                             input_tensor_shape[2],
                             input_tensor_depth])
    return y


def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


# def custom_context_attenion2(input_tensor, depth):
#     batch_size, height, width, num_channels = x.get_shape().as_list()
#     f = conv(input_tensor, ch // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv')  # [bs, h, w, c']
#     f = max_pooling(f)
#
#     g = conv(input_tensor, ch // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')  # [bs, h, w, c']
#
#     h = conv(input_tensor, ch // 2, kernel=1, stride=1, sn=self.sn, scope='h_conv')  # [bs, h, w, c]
#     h = max_pooling(h)
#
#     # N = h * w
#     s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
#
#     beta = tf.nn.softmax(s)  # attention map
#
#     o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
#     gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
#
#     o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
#     o = conv(o, ch, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
#     x = gamma * o + input_tensor
#
#     return x

def custom_context_attenion2(input_tensor, depth):
    with tf.variable_scope(None, default_name='Attention'):
        batch_size, height, width, num_channels = input_tensor.get_shape().as_list()
        f = tf.layers.conv2d(input_tensor, depth, kernel_size=1, strides=1, name='f_conv')  # [bs, h, w, c']
        # f = max_pooling(f)

        g = tf.layers.conv2d(input_tensor, depth, kernel_size=1, strides=1, name='g_conv')  # [bs, h, w, c']

        h = tf.layers.conv2d(input_tensor, depth, kernel_size=1, strides=1, name='h_conv')  # [bs, h, w, c]
        # h = max_pooling(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        o = tf.reshape(o, shape=[batch_size, height, width, num_channels])  # [bs, h, w, C]
        o = tf.layers.conv2d(o, depth, kernel_size=1, strides=1, name='attn_conv')
        gamma = tf.layers.conv2d(input_tensor, depth, kernel_size=1, strides=1, name='gamma_conv')  # [bs, h, w, c']
        gamma = tf.nn.sigmoid(gamma)
        tf.summary.scalar('max_gamma', tf.reduce_max(gamma))
        tf.summary.scalar('mean_gamma', tf.reduce_mean(gamma))
        x = o * gamma + input_tensor * (1 - gamma)
    return x


def custom_context_attenion3(input_tensor, depth):
    with tf.variable_scope(None, default_name='Attention'):
        batch_size, height, width, num_channels = input_tensor.get_shape().as_list()
        f = tf.layers.conv2d(input_tensor, depth, kernel_size=1, strides=1, name='f_conv')  # [bs, h, w, c']
        # f = max_pooling(f)

        g = tf.layers.conv2d(input_tensor, depth, kernel_size=1, strides=1, name='g_conv')  # [bs, h, w, c']

        h = tf.layers.conv2d(input_tensor, depth, kernel_size=1, strides=1, name='h_conv')  # [bs, h, w, c]
        # h = max_pooling(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        o = tf.reshape(o, shape=[batch_size, height, width, num_channels])  # [bs, h, w, C]
        o = tf.layers.conv2d(o, depth, kernel_size=1, strides=1, name='attn_conv')
        gamma = tf.layers.conv2d(input_tensor, depth, kernel_size=1, strides=1, name='gamma_conv')  # [bs, h, w, c']
        gamma = tf.nn.sigmoid(gamma)
        tf.summary.scalar('max_gamma', tf.reduce_max(gamma))
        tf.summary.scalar('mean_gamma', tf.reduce_mean(gamma))
        x = o * gamma + input_tensor * (1 - gamma)
    return x

def build_coarse_net(input_tensor):
    with tf.variable_scope('coarse_net'):
        # relu = tf.nn.relu
        # lrelu = tf.nn.leaky_relu

        # Looks like ELU allow to get rid of batch normalization because of normalizing properties
        # of this activation function
        relu = tf.nn.elu
        lrelu = relu
        cnum = int(32 * 0.75)

        # Net arch from deepfill v1
        # x = gen_conv(x, cnum, 5, 1, name='conv1')
        # x = gen_conv(x, 2 * cnum, 3, 2, name='conv2_downsample')
        # x = gen_conv(x, 2 * cnum, 3, 1, name='conv3')
        # x = gen_conv(x, 4 * cnum, 3, 2, name='conv4_downsample')
        # x = gen_conv(x, 4 * cnum, 3, 1, name='conv5')
        # x = gen_conv(x, 4 * cnum, 3, 1, name='conv6')
        # mask_s = resize_mask_like(mask, x)
        # x = gen_conv(x, 4 * cnum, 3, rate=2, name='conv7_atrous')
        # x = gen_conv(x, 4 * cnum, 3, rate=4, name='conv8_atrous')
        # x = gen_conv(x, 4 * cnum, 3, rate=8, name='conv9_atrous')
        # x = gen_conv(x, 4 * cnum, 3, rate=16, name='conv10_atrous')
        # x = gen_conv(x, 4 * cnum, 3, 1, name='conv11')
        # x = gen_conv(x, 4 * cnum, 3, 1, name='conv12')
        # x = gen_deconv(x, 2 * cnum, name='conv13_upsample')
        # x = gen_conv(x, 2 * cnum, 3, 1, name='conv14')
        # x = gen_deconv(x, cnum, name='conv15_upsample')
        # x = gen_conv(x, cnum // 2, 3, 1, name='conv16')
        # x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')
        # x = tf.clip_by_value(x, -1., 1.)
        # x_stage1 = x

        # Args are: input_tensor, kernel_size, depth, stride, dilation_rate, activation
        net = gated_convolution(input_tensor, 5, cnum, 1, 1, relu)

        net = gated_convolution(net, 3, 2 * cnum, 2, 1, relu)
        net = gated_convolution(net, 3, 2 * cnum, 1, 1, relu)

        net = gated_convolution(net, 3, 4 * cnum, 2, 1, relu)
        net = gated_convolution(net, 3, 4 * cnum, 1, 1, relu)
        net = gated_convolution(net, 3, 4 * cnum, 1, 1, relu)

        net = gated_convolution(net, 3, 4 * cnum, 1, 2, relu)
        net = gated_convolution(net, 3, 4 * cnum, 1, 4, relu)
        net = gated_convolution(net, 3, 4 * cnum, 1, 8, relu)
        net = gated_convolution(net, 3, 4 * cnum, 1, 16, relu)

        net = gated_convolution(net, 3, 4 * cnum, 1, 1, relu)
        net = gated_convolution(net, 3, 4 * cnum, 1, 1, relu)

        net = upsample_x2(net)
        net = gated_convolution(net, 3, 2 * cnum, 1, 1, lrelu)
        net = gated_convolution(net, 3, 2 * cnum, 1, 1, lrelu)

        net = upsample_x2(net)
        net = gated_convolution(net, 3, cnum, 1, 1, lrelu)
        net = gated_convolution(net, 3, cnum // 2, 1, 1, lrelu)

        net = regular_convolution(net, 3, 1, 1, 1)
        net = tf.clip_by_value(net, -1., 1.)
    return net


def build_refinement_net(input_tensor, mask):
    # relu = tf.nn.relu
    # lrelu = tf.nn.leaky_relu
    relu = tf.nn.elu
    lrelu = relu
    cnum = int(32 * 0.75)

    with tf.variable_scope('refinement_net'):
        # x = gen_conv(xnow, cnum, 5, 1, name='xconv1')
        # x = gen_conv(x, cnum, 3, 2, name='xconv2_downsample')
        # x = gen_conv(x, 2 * cnum, 3, 1, name='xconv3')
        # x = gen_conv(x, 2 * cnum, 3, 2, name='xconv4_downsample')
        # x = gen_conv(x, 4 * cnum, 3, 1, name='xconv5')
        # x = gen_conv(x, 4 * cnum, 3, 1, name='xconv6')
        # x = gen_conv(x, 4 * cnum, 3, rate=2, name='xconv7_atrous')
        # x = gen_conv(x, 4 * cnum, 3, rate=4, name='xconv8_atrous')
        # x = gen_conv(x, 4 * cnum, 3, rate=8, name='xconv9_atrous')
        # x = gen_conv(x, 4 * cnum, 3, rate=16, name='xconv10_atrous')
        # x_hallu = x
        # # attention branch
        # x = gen_conv(xnow, cnum, 5, 1, name='pmconv1')
        # x = gen_conv(x, cnum, 3, 2, name='pmconv2_downsample')
        # x = gen_conv(x, 2 * cnum, 3, 1, name='pmconv3')
        # x = gen_conv(x, 4 * cnum, 3, 2, name='pmconv4_downsample')
        # x = gen_conv(x, 4 * cnum, 3, 1, name='pmconv5')
        # x = gen_conv(x, 4 * cnum, 3, 1, name='pmconv6',
        #              activation=tf.nn.relu)
        # x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
        # x = gen_conv(x, 4 * cnum, 3, 1, name='pmconv9')
        # x = gen_conv(x, 4 * cnum, 3, 1, name='pmconv10')
        # pm = x
        # x = tf.concat([x_hallu, pm], axis=3)
        #
        # x = gen_conv(x, 4 * cnum, 3, 1, name='allconv11')
        # x = gen_conv(x, 4 * cnum, 3, 1, name='allconv12')
        # x = gen_deconv(x, 2 * cnum, name='allconv13_upsample')
        # x = gen_conv(x, 2 * cnum, 3, 1, name='allconv14')
        # x = gen_deconv(x, cnum, name='allconv15_upsample')
        # x = gen_conv(x, cnum // 2, 3, 1, name='allconv16')
        # x = gen_conv(x, 3, 3, 1, activation=None, name='allconv17')
        # x_stage2 = tf.clip_by_value(x, -1., 1.)

        # Args are: input_tensor, kernel_size, depth, stride, dilation_rate, activation
        net = gated_convolution(input_tensor, 5, cnum, 1, 1, relu)

        net = gated_convolution(net, 3, 2 * cnum, 2, 1, relu)
        net = gated_convolution(net, 3, 2 * cnum, 1, 1, relu)

        net = gated_convolution(net, 3, 4 * cnum, 2, 1, relu)
        net = gated_convolution(net, 3, 4 * cnum, 1, 1, relu)
        net = gated_convolution(net, 3, 4 * cnum, 1, 1, relu)

        net = gated_convolution(net, 3, 4 * cnum, 1, 2, relu)
        net = gated_convolution(net, 3, 4 * cnum, 1, 4, relu)
        net = gated_convolution(net, 3, 4 * cnum, 1, 8, relu)
        net = gated_convolution(net, 3, 4 * cnum, 1, 16, relu)
        net_hallu = net

        # Insert attention here
        # net = gated_convolution(input_tensor, 5, cnum, 1, 1, relu)
        #
        # net = gated_convolution(net, 3, 2 * cnum, 2, 1, relu)
        # net = gated_convolution(net, 3, 2 * cnum, 1, 1, relu)
        #
        # net = gated_convolution(net, 3, 4 * cnum, 2, 1, relu)
        # net = gated_convolution(net, 3, 4 * cnum, 1, 1, relu)
        # net = gated_convolution(net, 3, 4 * cnum, 1, 1, tf.nn.relu)
        # # net, offset_flow = contextual_attention(net, net, mask, 3, 1, rate=2)
        # # net, offset_flow = contextual_attention(net, net, mask, 3, stride=1, rate=1,
        # #                                         fuse_k=3, softmax_scale=1.0, fuse=True)
        # net = custom_context_attenion(net, mask)
        # # net = custom_context_attenion2(net, 4 * cnum)
        #
        # net = gated_convolution(net, 3, 4 * cnum, 1, 1, relu)
        # net = gated_convolution(net, 3, 4 * cnum, 1, 1, relu)
        # pm = net
        #
        # # Concatenating regular conv and attention branches
        # net = tf.concat([net_hallu, pm], axis=3)

        net = gated_convolution(net, 3, 4 * cnum, 1, 1, relu)
        net = gated_convolution(net, 3, 4 * cnum, 1, 1, relu)

        net = upsample_x2(net)
        net = gated_convolution(net, 3, 2 * cnum, 1, 1, lrelu)
        net = gated_convolution(net, 3, 2 * cnum, 1, 1, lrelu)

        net = upsample_x2(net)
        net = gated_convolution(net, 3, cnum, 1, 1, lrelu)
        net = gated_convolution(net, 3, cnum // 2, 1, 1, lrelu)

        net = regular_convolution(net, 3, 3, 1, 1)
        net = tf.clip_by_value(net, -1., 1.)
        return net


def build_generator(params, images, masks, edges, mode):
    # Input size in https://arxiv.org/pdf/1801.07892.pdf is 256x256
    # training = True if mode == tf.estimator.ModeKeys.TRAIN else False

    with tf.variable_scope('generator'):
        images_gray = tf.image.rgb_to_grayscale(images)
        images_masks_edges = tf.concat([images_gray, masks, edges * masks], axis=3)
        edges_result = build_coarse_net(images_masks_edges)

        # refinement_input_images = edges_result * (1 - masks) + images * masks
        refinement_input = tf.concat([images, masks, edges_result], axis=3)
        refined_result = build_refinement_net(refinement_input, masks)

        return refined_result, edges_result


def build_discriminator(name, input_tensor):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # Change to leaky relu
        activation = tf.nn.leaky_relu
        cnum = int(64 * 1.0)

        net = conv2d_spectral_norm(input_tensor, cnum, 5, 1, 'SAME', activation=activation, name='conv1')
        net = conv2d_spectral_norm(net, cnum * 2, 5, 2, 'SAME', activation=activation, name='conv2')
        net = conv2d_spectral_norm(net, cnum * 4, 5, 2, 'SAME', activation=activation, name='conv3')
        net = conv2d_spectral_norm(net, cnum * 4, 5, 2, 'SAME', activation=activation, name='conv4')
        net = conv2d_spectral_norm(net, cnum * 4, 5, 2, 'SAME', activation=activation, name='conv5')
        net = conv2d_spectral_norm(net, cnum * 4, 5, 2, 'SAME', activation=activation, name='conv6')
        return net
