from functools import partial
import tensorflow as tf
import os
import model_builder

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("image_size", "512x512", "")

tf.flags.DEFINE_string("train_images_tfrecords_path", "data/tfrecords/celebA/train", "")
tf.flags.DEFINE_string("train_masks_tfrecords_path", "data/tfrecords/masks/train", "")

tf.flags.DEFINE_string("eval_images_tfrecords_path", "data/tfrecords/celebA/eval", "")
tf.flags.DEFINE_string("eval_masks_tfrecords_path", "data/tfrecords/masks/eval", "")

image_size = [int(dim) for dim in FLAGS.image_size.split("x")]


def resize_image_keep_aspect_ratio(image, max_height, max_width, use_min_ratio):
    def compute_new_dims(height, width, max_height, max_width, use_min_ratio):
        # If use_min_ratio is set to true than image will be resized to max of smaller dim
        height_float = tf.cast(height, tf.float32)
        width_float = tf.cast(width, tf.float32)
        max_height_float = tf.cast(max_height, tf.float32)
        max_width_float = tf.cast(max_width, tf.float32)

        height_ratio = height_float / max_height_float
        widht_ratio = width_float / max_width_float

        if use_min_ratio:
            ratio = tf.minimum(height_ratio, widht_ratio)
        else:
            ratio = tf.maximum(height_ratio, widht_ratio)

        new_height = tf.cast(tf.floor(height_float / ratio), tf.int32)
        new_width = tf.cast(tf.floor(width_float / ratio), tf.int32)

        return (new_height, new_width)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    new_height_and_width = compute_new_dims(height, width, max_height, max_width, use_min_ratio=use_min_ratio)

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bicubic(image, tf.stack(new_height_and_width))
    image = tf.squeeze(image, [0])
    return image


def random_flip_left_right(tensors):
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    result_tensors = []
    for tensor in tensors:
        result_tensors.append(tf.reverse(tensor, mirror))

    return result_tensors


def parse_tf_records(image_example_proto, mask_example_proto, float_type):
    features = {"image": tf.FixedLenFeature((), tf.string, ""),
                "edges": tf.FixedLenFeature((), tf.string, "")}

    image_parsed = tf.parse_single_example(image_example_proto, features)
    mask_parsed = tf.parse_single_example(mask_example_proto, features)

    image_gt = tf.image.decode_image(image_parsed["image"], channels=3)
    image_gt = resize_image_keep_aspect_ratio(image_gt, image_size[0], image_size[1], use_min_ratio=True)

    edges = tf.image.decode_image(image_parsed["edges"], channels=1)
    edges = resize_image_keep_aspect_ratio(edges, image_size[0], image_size[1], use_min_ratio=True)

    image_and_edges = tf.concat([image_gt, edges], axis=2)
    image_and_edges = tf.random_crop(image_and_edges, [image_size[0], image_size[1], 4])
    image_gt, edges = tf.split(image_and_edges, [3, 1], axis=2)
    image_gt, edges = random_flip_left_right([image_gt, edges])

    image_gt = tf.cast(image_gt, float_type)
    image_gt = model_builder.int2float(image_gt)
    image_gt.set_shape((None, None, 3))

    edges = tf.cast(edges, float_type)
    edges = tf.cast(tf.greater(edges, 127), dtype=float_type)
    edges.set_shape((None, None, 1))

    mask = tf.image.decode_image(mask_parsed["image"], channels=1)
    mask = resize_image_keep_aspect_ratio(mask, image_size[0], image_size[1], use_min_ratio=True)
    mask = tf.random_crop(mask, [image_size[0], image_size[1], 1])
    mask = tf.image.random_flip_left_right(mask)
    mask = tf.cast(tf.greater(mask, 127), dtype=float_type)
    mask.set_shape((None, None, 1))

    image_in = image_gt * mask
    features = {'i_in': image_in, 'i_gt': image_gt, 'mask': mask, 'edges': edges}
    return features


def train_input_fn(params, batch_size):
    return input_fn(params, FLAGS.train_images_tfrecords_path, FLAGS.train_masks_tfrecords_path, batch_size,
                    shuffle=True)


def eval_input_fn(params, batch_size):
    return input_fn(params, FLAGS.eval_images_tfrecords_path, FLAGS.eval_masks_tfrecords_path, batch_size,
                    shuffle=False)


def input_fn(params, images_path, masks_path, batch_size, shuffle):
    image_filenames_list = tf.data.Dataset.list_files(os.path.join(images_path, "*.tfrecord"), shuffle=shuffle)
    mask_filenames_list = tf.data.Dataset.list_files(os.path.join(masks_path, "*.tfrecord"), shuffle=shuffle)
    images_dataset = tf.data.TFRecordDataset(image_filenames_list, num_parallel_reads=4)
    masks_dataset = tf.data.TFRecordDataset(mask_filenames_list, num_parallel_reads=4)

    if shuffle:
        images_dataset = images_dataset.shuffle(buffer_size=10)
        masks_dataset = masks_dataset.shuffle(buffer_size=10)

    dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))

    if shuffle:
        dataset.shuffle(buffer_size=512)
    partial_parse_tf_records = partial(parse_tf_records, float_type=params['float_type'])
    dataset = dataset.apply(tf.contrib.data.map_and_batch(partial_parse_tf_records,
                                                          batch_size,
                                                          num_parallel_batches=os.cpu_count(),
                                                          drop_remainder=True))
    dataset = dataset.prefetch(buffer_size=5)
    # iterator = dataset.make_one_shot_iterator()
    # dataset = iterator.get_next()
    return dataset
