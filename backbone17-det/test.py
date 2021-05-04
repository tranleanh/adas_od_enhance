import os
import cv2
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.yolo_custom import YOLO_CUSTOM
from core.config_custom import cfg

ckpt_file = 'path/to/model/file/'
input_dir = 'path/to/val/images/'
detection_result_dir = 'path/to/output/folder/'

classes = utils.read_class_names(cfg.YOLO.CLASSES)
num_classes = 1

class YoloTest(object):
    def __init__(self):
        self.input_size = cfg.TEST.INPUT_SIZE
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.weight_file = ckpt_file

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')
        model = YOLO_CUSTOM(self.input_data, self.trainable)
        self.pred_bbox = model.pred_bbox

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):
        org_h, org_w, _ = image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_bbox = self.sess.run(self.pred_bbox, feed_dict={self.input_data: image_data, self.trainable: False})
        pred_bbox = np.reshape(pred_bbox, (-1, 5 + num_classes))

        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, 0.2)
        bboxes = utils.nms(bboxes, 0.5)

        f = open(os.path.join(detection_result_dir, fname.replace('.jpg', '.txt')), "w")
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            cls_ind = int(bbox[5])
            cls = classes[int(cls_ind)]
            f.write('{} {:.6f} {} {} {} {}\n'.format(cls.replace(' ', '_'), score, x1, y1, x2, y2))



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    test = YoloTest()

    if not os.path.isdir(detection_result_dir):
        os.makedirs(detection_result_dir)
    filenames = os.listdir(input_dir)
    for fname in filenames:
        img = cv2.imread(os.path.join(input_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test.predict(img)