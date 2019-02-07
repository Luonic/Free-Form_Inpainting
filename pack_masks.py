import tensorflow as tf
import cv2
import numpy as np
import os
import uuid
from tqdm import tqdm
import glob

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("masks_dir", "data/masks/images", "")
tf.flags.DEFINE_string("tfrecords_dir", "data/tfrecords/masks/", "")

def resize(image, min_side_max):
    shape = image.shape

    if shape[0] > shape[1]:
        # Portrait image
        new_width = min_side_max
        new_height = int(shape[0] * (float(min_side_max) / shape[1]))
    else:
        new_height= min_side_max
        new_width = int(shape[1] * (float(min_side_max) / shape[0]))

    image = cv2.resize(image, (new_width, new_height))

def serialize_file(file_path):
    basename = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, "rb") as img_f:
        
        image_buffer = img_f.read()
        image = np.frombuffer(image_buffer, dtype=np.uint8)

        example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                    }))
        serialized = example.SerializeToString()
        return serialized

def dir_to_tfrecord(masks_path, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    filenames = glob.glob(os.path.join(masks_path, '*.png'))
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for i, path in enumerate(tqdm(filenames)):
        if i % 10000 == 0:
            tf_rec_filename = str(uuid.uuid1()) + ".tfrecord"
            writer = tf.python_io.TFRecordWriter(os.path.join(dst_dir, tf_rec_filename), options=options)

        serialized_file = serialize_file(path)
        writer.write(serialized_file)


if __name__ == '__main__':
    dir_to_tfrecord(os.path.join(FLAGS.masks_dir, 'train'), os.path.join(FLAGS.tfrecords_dir, "train"))
    dir_to_tfrecord(os.path.join(FLAGS.masks_dir, 'eval'), os.path.join(FLAGS.tfrecords_dir, "eval"))
    dir_to_tfrecord(os.path.join(FLAGS.masks_dir, 'test'), os.path.join(FLAGS.tfrecords_dir, "test"))