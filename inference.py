import numpy as np
import cv2
import tensorflow as tf
import model_builder

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("saved_model_dir", "train_logs/19_0.5-g_0.5-d_original_context_attention/SavedModels/1534124518", "")

sess_config = tf.ConfigProto()
# sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
# sess_config = None
with tf.Session(graph=tf.Graph(), config=sess_config) as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAGS.saved_model_dir)
    sess.run(tf.tables_initializer())
    signature = meta_graph_def.signature_def
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    i_in_tensor_name = signature[signature_key].inputs["i_in"].name
    mask_tensor_name = signature[signature_key].inputs["mask"].name
    edges_tensor_name = signature[signature_key].inputs["edges"].name
    i_out_tensor_name = signature[signature_key].outputs["i_out"].name

    i_in_tensor = sess.graph.get_tensor_by_name(i_in_tensor_name)
    mask_tensor = sess.graph.get_tensor_by_name(mask_tensor_name)
    edges_tensor = sess.graph.get_tensor_by_name(edges_tensor_name)
    i_out_tensor = sess.graph.get_tensor_by_name(i_out_tensor_name)

#
#     while (True):
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#
#         height = frame.shape[0]
#         width = frame.shape[1]
#
#         min_dim = min(height, width)
#
#         frame_square = frame[height // 2 - min_dim // 2:height // 2 + min_dim // 2,
#                        width // 2 - min_dim // 2:width // 2 + min_dim // 2]
#
#         input_size = 512
#         frame = cv2.resize(frame_square, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
#         orig_frame = frame.copy()
#         # Our operations on the frame come here
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         frame = frame.astype(np.float32) * (2.0 / 255) - 1.0
#
#         frame = np.expand_dims(frame, axis=0)
#
#         classes_res = sess.run(classes, feed_dict={image_tensor: frame})
#
#         # Display the resulting frame
#         class_id = 15 #15
#         pred = classes_res[0] == class_id # ==
#         mask = np.expand_dims(pred, axis=2).astype(np.uint8)
#         orig_frame *= mask
#         green_img = np.zeros_like(orig_frame, dtype=np.uint8)
#         green_img[:, :, 1] = 255
#         orig_frame += green_img * (1 - mask)
#         # pred = (pred * 255).astype(np.uint8)
#         # pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
#         # preview = np.concatenate((orig_frame, pred), axis=1)
#         preview = cv2.resize(orig_frame, (1024, 1024), cv2.INTER_LINEAR)
#         cv2.imshow('preview', preview)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

    drawing = False  # true if mouse is pressed
    mode = False  # if True, draw rectangle. Press 'm' to toggle to curve
    ix, iy = -1, -1


    # mouse callback function
    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing, mode, img, img_tmp, mask, edges

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv2.circle(edges, (x, y), 1, (0, 0, 0), -1)
                else:
                    cv2.circle(mask, (x, y), 3, (1, 1, 1), -1)

                img_tmp = img * (1 - mask) + 255 * mask
                img_tmp = img_tmp * (1 - (1 - edges) * mask)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # if mode == True:
            #     cv2.circle(edges, (x, y), 5, (0, 0, 0), -1)
            # else:
            #     cv2.circle(mask, (x, y), 5, (1, 1, 1), -1)


    def inpaint():
        global img_tmp
        img_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tmp = np.expand_dims((img_tmp * (2 / 255) - 1.0) * (1 - mask), axis=0)

        mask_tmp = np.expand_dims(np.expand_dims(1 - mask[:, :, 0], axis=0), axis=3)
        edges_tmp = np.expand_dims(np.expand_dims(1 - edges[:, :, 0], axis=0), axis=3)
        # cv2.imshow('mask', mask * 255)
        # cv2.imshow('edges', edges * 255)

        i_out_res = sess.run(i_out_tensor,
                             feed_dict={i_in_tensor: img_tmp, mask_tensor: mask_tmp, edges_tensor: edges_tmp})
        # i_out_res = (img_tmp)
        # print(i_out_res)
        img_tmp = (i_out_res[0] + 1) * (255 / 2)
        img_tmp = img_tmp.astype(np.uint8)
        img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2BGR)





    # img = cv2.imread('data/images/I7udR5eelBQ.jpg')
    img = cv2.imread('data/images/bjBnleBtoEI.jpg')
    height = img.shape[0]
    width = img.shape[1]

    min_dim = min(height, width)

    img = img[height // 2 - min_dim // 2:height // 2 + min_dim // 2,
                   width // 2 - min_dim // 2:width // 2 + min_dim // 2]

    input_size = 256
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    img_tmp = img.copy()

    mask = np.zeros_like(img)
    edges = np.ones_like(img)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while (1):
        # print(type(img_tmp))
        # img_tmp = cv2.resize(img_tmp, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('image', img_tmp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == ord('p'):
            print('Inpainting...')
            inpaint()
        elif k == 27:
            break

    cv2.destroyAllWindows()
