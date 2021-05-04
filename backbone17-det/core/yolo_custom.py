import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config_custom import cfg


class YOLO_CUSTOM(object):
    """Implement tensoflow yolo_custom here"""
    def __init__(self, input_data, trainable):

        self.trainable        = trainable
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class        = len(self.classes)
        self.stride           = cfg.YOLO.STRIDE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.anchors          = utils.get_custom_anchors(cfg.YOLO.ANCHORS, self.anchor_per_scale)
        self.iou_loss_thresh  = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method  = cfg.YOLO.UPSAMPLE_METHOD

        self.conv_bbox = self.__build_nework(input_data)

        with tf.variable_scope('pred_bbox'):
            self.pred_bbox = self.decode(self.conv_bbox, self.anchors, self.stride)

    def __build_nework(self, input_data):

        # Backbone17-Det #
        input_data = common.convolutional(input_data, (3, 3, 3, 32), self.trainable, 'conv1', downsample=True)
        input_shrt = common.convolutional(input_data, (3, 3, 32, 64), self.trainable, 'conv2', downsample=True)
        input_data = common.convolutional(input_shrt, (3, 3, 64, 128), self.trainable, 'conv3')
        input_data = common.convolutional(input_data, (1, 1, 128, 64), self.trainable, 'conv4')
        input_data = input_data + input_shrt              # out2 + out4
        input_data = common.squeeze_excitation_block(input_data, 64, 4, 'squeeze_excitation0')
        
        input_shrt = common.convolutional(input_data, (3, 3, 64, 128), self.trainable, 'conv5', downsample=True)
        input_data = common.convolutional(input_shrt, (3, 3, 128, 256), self.trainable, 'conv6')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv7')
        input_data = input_data + input_shrt              # out5 + out7
        input_data = common.squeeze_excitation_block(input_data, 128, 4, 'squeeze_excitation1')

        input_shrt = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv8', downsample=True)
        input_data = common.convolutional(input_shrt, (3, 3, 256, 512), self.trainable, 'conv9')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv10')
        input_data = input_data + input_shrt              # out8 + out10
        input_shrt = common.squeeze_excitation_block(input_data, 256, 4, 'squeeze_excitation2')

        input_data = common.convolutional(input_shrt, (3, 3, 256, 512), self.trainable, 'conv11')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv12')
        input_data = input_data + input_shrt              # se2 + out12
        input_data = common.squeeze_excitation_block(input_data, 256, 4, 'squeeze_excitation3')

        input_shrt = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv13', downsample=True)
        input_data = common.convolutional(input_shrt, (3, 3, 512, 1024), self.trainable, 'conv14')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv15')
        input_data = input_data + input_shrt              # out13 + out15
        input_data = common.squeeze_excitation_block(input_data, 512, 4, 'squeeze_excitation4')

        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv16')
        input_data = common.convolutional(input_data, (3, 3, 1024, 1024), self.trainable, 'conv17')
        conv_bbox = common.convolutional(input_data, (1, 1, 1024, 3 * (self.num_class + 5)),
                                         self.trainable, 'conv_bbox', activate=False, bn=False)
        return conv_bbox


    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < self.iou_loss_thresh, tf.float32 )

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss



    def compute_loss(self, label_bbox, true_bbox):

        with tf.name_scope('box_loss'):
            loss_bbox = self.loss_layer(self.conv_bbox, self.pred_bbox, label_bbox, true_bbox,
                                        anchors=self.anchors, stride=self.stride)

        with tf.name_scope('giou_loss'):
            giou_loss = loss_bbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_bbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_bbox[2]

        return giou_loss, conf_loss, prob_loss


