import os
import cv2
import numpy as np
import tensorflow as tf
import data_reader
import random
import uuid
from tqdm import tqdm
import math

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('masks_path', 'data/masks', '')
# maxVertex, maxLength, maxBrushWidth, maxAngle
tf.flags.DEFINE_integer('max_vertex', 6, '')
tf.flags.DEFINE_integer('max_length', 200, '')
tf.flags.DEFINE_integer('max_brush_width', 40, '')
tf.flags.DEFINE_float('max_angle', math.pi, '')
n_masks_train = 200
n_masks_eval = 100
n_masks_test = 100

image_size = tuple(data_reader.image_size)

mask_fill_range = (0.1, 0.3)
min_size = min(data_reader.image_size)
circle_radius_range = (int(0.05 * min_size), int(0.1 * min_size))


def generate_mask():
    mask = np.zeros(image_size, dtype=np.uint8)

    num_vertex = int(np.random.uniform(1, FLAGS.max_vertex))
    start_x = int(np.random.uniform(0, data_reader.image_size[1]))
    start_y = int(np.random.uniform(0, data_reader.image_size[0]))

    for step in range(0, num_vertex):
        angle = np.random.uniform(0, FLAGS.max_angle)

        if step % 2 == 0:
            # Reverse mode
            angle = 2 * math.pi - angle

        length = np.random.uniform(10, FLAGS.max_length)
        brush_width = int(np.random.uniform(FLAGS.max_brush_width / 4, FLAGS.max_brush_width))

        end_x = int(start_x + length * math.sin(angle))
        end_y = int(start_y + length * math.cos(angle))

        cv2.line(mask, (start_y, start_x), (end_y, end_x), 255, brush_width)

        start_x = end_x
        start_y = end_y

        cv2.circle(mask, (447, 63), brush_width // 2, 255, -1)

    if np.random.uniform(0.0, 1.0) > 0.5:
        mask = np.fliplr(mask)

    if np.random.uniform(0.0, 1.0) > 0.5:
        mask = np.flipud(mask)

    mask = 255 - mask
    return mask

def generate_masks(count, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for i in tqdm(range(0, count)):

        mask = generate_mask()

        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

        filename = str(uuid.uuid4()) + '.png'
        dst_path = os.path.join(dst_dir, filename)
        cv2.imwrite(dst_path, mask)

generate_masks(n_masks_train, os.path.join(FLAGS.masks_path, 'train'))
generate_masks(n_masks_eval, os.path.join(FLAGS.masks_path, 'eval'))
# generate_masks(n_masks_test, os.path.join(FLAGS.masks_path, 'test'))


# My old method
# for i in tqdm(range(0, n_masks)):
#     mask = np.zeros(tuple(data_reader.image_size), dtype=np.float32)
#     mask_fill_treshold = np.random.uniform(mask_fill_range[0], mask_fill_range[1])
#
#     for i in range(0, 100):
#         color = float(random.randint(0, 2))
#         center = (random.randint(0, data_reader.image_size[1]), random.randint(0, data_reader.image_size[0]))
#         radius = random.randint(circle_radius_range[0], circle_radius_range[1])
#         mask = cv2.circle(mask, center, radius, color=color, thickness=-1, lineType=cv2.LINE_AA)
#         # mask = cv2.normalize(mask, None, 0.0, 1.0, cv2.NORM_MINMAX)
#         blur_kernel_size = int(min_size / 20)
#         mask = cv2.blur(mask, (blur_kernel_size, blur_kernel_size), borderType=cv2.BORDER_REFLECT101)
#         mask = mask - (np.mean(mask) - 0.5)
#
#     mask_binary = mask.copy()
#     cv2.threshold(mask, mask_fill_treshold, 1, type=cv2.THRESH_BINARY, dst=mask_binary)
#
#     mask_binary = (mask_binary * 255)
#
#     # cv2.imshow('mask', mask_binary)
#     # cv2.waitKey(1)
#
#     filename = str(uuid.uuid4()) + '.jpg'
#     dst_path = os.path.join(FLAGS.masks_path, filename)
#     cv2.imwrite(dst_path, mask_binary)
