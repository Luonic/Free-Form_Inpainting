import tensorflow as tf
import cv2
import numpy as np
import os
import uuid
from tqdm import tqdm

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("images_dir", "data/images/celebA/images", "")
tf.flags.DEFINE_string("edges_dir", "data/edges/celebA", "")
tf.flags.DEFINE_string("partition_file", "data/images/celebA/list_eval_partition.txt", "Text file containing lines like <filename> <int> "
                                                            "where int is 0 for train, 1 for validation and 2 for test")
tf.flags.DEFINE_string("tfrecords_dir", "data/tfrecords/celebA/", "")

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
    with open(file_path, "rb") as img_f, \
            open(os.path.join(FLAGS.edges_dir, basename + '.png'), "rb") as edges_f:
        
        image_buffer = img_f.read()
        image = np.frombuffer(image_buffer, dtype=np.uint8)

        edges_buffer = edges_f.read()
        edges = np.frombuffer(edges_buffer, dtype=np.uint8)

        example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                        'edges': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[edges.tobytes()])),
                    }))
        serialized = example.SerializeToString()
        return serialized

def paths_to_tfrecord(paths, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for i, path in enumerate(tqdm(paths)):
        if i % 10000 == 0:
            tf_rec_filename = str(uuid.uuid1()) + ".tfrecord"
            writer = tf.python_io.TFRecordWriter(os.path.join(dst_dir, tf_rec_filename), options=options)

        serialized_file = serialize_file(path)
        writer.write(serialized_file)


if __name__ == '__main__':
    with open(FLAGS.partition_file, "r") as partition_f:
        partition_rows = partition_f.readlines()
        partition_rows = [row.strip().split() for row in partition_rows]

    train_list, eval_list, test_list = [], [], []
    for row in partition_rows:
        file_path = os.path.join(FLAGS.images_dir, row[0])
        partition_id = int(row[1])
        if partition_id == 0:
            train_list.append(file_path)
        elif partition_id == 1:
            eval_list.append(file_path)
        elif partition_id == 2:
            test_list.append(file_path)
        else:
            raise Exception("Wrong partition id {} in partitons file for file {}".format(row[1], row[0]))

    paths_to_tfrecord(train_list, os.path.join(FLAGS.tfrecords_dir, "train"))
    paths_to_tfrecord(eval_list, os.path.join(FLAGS.tfrecords_dir, "eval"))
    paths_to_tfrecord(test_list, os.path.join(FLAGS.tfrecords_dir, "test"))