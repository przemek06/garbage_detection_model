import tensorflow as tf
from tensorflow.keras import layers, Model
from selective_search import get_rois
from normalize_data import standardize_image

class FastRCNN(Model):
    def __init__(self, classes):
        super(FastRCNN, self).__init__()
        self.classes = classes
        self.num_classes = len(classes)
        self._bg_weight = 0.3
        self._fg_weight = 1.0
        self._build_model()
        
        self.cls_loss_metric = tf.keras.metrics.Mean(name='cls_loss')
        self.bbox_loss_metric = tf.keras.metrics.Mean(name='bbox_loss')
        self.cls_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy')
    
    def _build_model(self):
        self.backbone = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet',input_shape=(None, None, 3))
        self.conv_reduce = layers.Conv2D(256, (1, 1), activation='relu')       
        self.roi_pooling = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(256, activation='relu', kernel_regularizer='l2')
        self.fc2 = layers.Dense(128, activation='relu', kernel_regularizer='l2')
        self.cls_output = layers.Dense(self.num_classes, activation='softmax', name='cls_output')
        self.bbox_output = layers.Dense(self.num_classes * 4, name='bbox_output')
            
    def call(self, inputs, training=False):
        image, rois = inputs
        
        feature_maps = self.backbone(image, training=training)
        feature_maps = self.conv_reduce(feature_maps, training=training)
        
        batch_size = tf.shape(image)[0]
        num_rois = tf.shape(rois)[1]
        img_height = tf.cast(tf.shape(image)[1], tf.float32)
        img_width = tf.cast(tf.shape(image)[2], tf.float32)
        
        rois_scaled = rois * [img_width, img_height, img_width, img_height]
        downscale = img_height / tf.cast(tf.shape(feature_maps)[1], tf.float32)
        rois_feature = rois_scaled / downscale
        
        feature_width = tf.cast(tf.shape(feature_maps)[1], tf.float32)
        feature_height = tf.cast(tf.shape(feature_maps)[2], tf.float32)  
        ymin = rois_feature[..., 1] / feature_height
        xmin = rois_feature[..., 0] / feature_width
        ymax = rois_feature[..., 3] / feature_height
        xmax = rois_feature[..., 2] / feature_width
        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        boxes_flat = tf.reshape(boxes, [-1, 4])
        
        box_indices = tf.repeat(tf.range(batch_size), num_rois)
        
        cropped = tf.image.crop_and_resize(
            feature_maps,
            boxes_flat,
            box_indices,
            crop_size=(14, 14),
            method='bilinear'  
        )
        
        pooled = self.roi_pooling(cropped)
        
        flattened = self.flatten(pooled)
        fc1 = self.fc1(flattened)
        fc2 = self.fc2(fc1)
        
        cls_out = self.cls_output(fc2)
        bbox_out = self.bbox_output(fc2)
        
        cls_out = tf.reshape(cls_out, (batch_size, num_rois, self.num_classes))
        bbox_out = tf.reshape(bbox_out, (batch_size, num_rois, self.num_classes * 4))
        
        return (cls_out, bbox_out)
    
    def test_step(self, data):
        x, y = data
        images, rois = x
        true_cls, true_bbox, valid_mask = y

        pred_cls, pred_bbox = self((images, rois), training=False)

        cls_loss = self._weighted_cls_loss(true_cls, pred_cls)

        true_cls_flat = tf.reshape(true_cls, [-1])
        true_bbox_flat = tf.reshape(true_bbox, [-1, 4])
        pred_bbox_flat = tf.reshape(pred_bbox, [-1, self.num_classes, 4])

        indices = tf.stack([
            tf.range(tf.shape(true_cls_flat)[0]), 
            true_cls_flat
        ], axis=1)

        pred_bbox_selected = tf.gather_nd(pred_bbox_flat, indices)

        mask = true_cls_flat != (self.num_classes - 1)
        masked_true_bbox = tf.boolean_mask(true_bbox_flat, mask)
        masked_pred_bbox = tf.boolean_mask(pred_bbox_selected, mask)

        bbox_loss = self.bbox_loss_fn(masked_true_bbox, masked_pred_bbox)

        self.cls_loss_metric.update_state(cls_loss)
        self.bbox_loss_metric.update_state(bbox_loss)
        self.cls_accuracy_metric.update_state(true_cls, pred_cls)

        return {m.name: m.result() for m in self.metrics}
    
    @property
    def metrics(self):
        return [self.cls_loss_metric, self.bbox_loss_metric, self.cls_accuracy_metric]
    
    def compile(self, optimizer, cls_loss_fn, bbox_loss_fn, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.cls_loss_fn = cls_loss_fn
        self.bbox_loss_fn = bbox_loss_fn

    def _weighted_cls_loss(self, true_cls, pred_cls):
        return self.cls_loss_fn(true_cls, pred_cls)  
        
    
    def train_step(self, data):
        x, y = data
        images, rois = x
        true_cls, true_bbox, valid_mask = y     

        with tf.GradientTape() as tape:
            pred_cls, pred_bbox = self((images, rois), training=True)
            
            cls_loss = self.cls_loss_fn(
                true_cls, pred_cls,
                sample_weight=valid_mask 
            )
            
            true_cls_flat = tf.reshape(true_cls, [-1])
            true_bbox_flat = tf.reshape(true_bbox, [-1, 4])
            pred_bbox_flat = tf.reshape(pred_bbox, [-1, self.num_classes, 4])
            valid_flat = tf.reshape(valid_mask, [-1])
            
            indices = tf.stack([
                tf.range(tf.shape(true_cls_flat)[0]), 
                true_cls_flat
            ], axis=1)
            
            pred_bbox_selected = tf.gather_nd(pred_bbox_flat, indices)
            
            mask = true_cls_flat != (self.num_classes - 1)
            mask = tf.logical_and(mask, valid_flat)
            masked_true_bbox = tf.boolean_mask(true_bbox_flat, mask)
            masked_pred_bbox = tf.boolean_mask(pred_bbox_selected, mask)
            
            bbox_loss = self.bbox_loss_fn(masked_true_bbox, masked_pred_bbox)
            
            total_loss = cls_loss + bbox_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.cls_loss_metric.update_state(cls_loss)
        self.bbox_loss_metric.update_state(bbox_loss)
        self.cls_accuracy_metric.update_state(true_cls, pred_cls)
        
        return {m.name: m.result() for m in self.metrics}
    
    def predict(self, image):
        rois = tf.expand_dims(tf.convert_to_tensor(get_rois(image)), axis=0)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        pred_classes_probs, pred_offsets = self((image, rois), training=False)
        pred_classes = tf.argmax(pred_classes_probs, axis=-1)
        non_background_indices = tf.where(pred_classes[0] != (self.num_classes - 1))
        pred_classes_probs = tf.gather(pred_classes_probs[0], non_background_indices)
        pred_classes = tf.gather(pred_classes[0], non_background_indices)
        pred_offsets = tf.gather(pred_offsets[0], non_background_indices)
        rois = tf.gather(rois[0], non_background_indices)

        pred_offsets_flat = tf.reshape(pred_offsets, [-1, self.num_classes, 4])
        indices = pred_classes
        pred_offsets = tf.gather(pred_offsets_flat, indices, axis=1, batch_dims=1)
        pred_offsets = tf.reshape(pred_offsets, [-1, 4])

        processed_boxes = []
        for i in range(rois.shape[0]):
            roi = rois[i][0]
            offsets = pred_offsets[i]
            box = self.process_boxes(image, roi, offsets)
            processed_boxes.append(box)
        processed_boxes = tf.stack(processed_boxes, axis=0)
        confidence_scores = tf.squeeze(tf.reduce_max(pred_classes_probs, axis=-1), axis=-1)

        selected_indices = self.nms(confidence_scores, processed_boxes)
        final_boxes = tf.gather(processed_boxes, selected_indices)
        final_classes = tf.gather(pred_classes, selected_indices)

        return final_classes, final_boxes
    
    def process_boxes(self, image, roi, offsets):
        w_im, h_im = tf.cast(tf.shape(image)[1], tf.float32), tf.cast(tf.shape(image)[2], tf.float32)

        x_roi, y_roi = roi[0], roi[1]
        w_roi = roi[2] - roi[0]
        h_roi = roi[3] - roi[1]

        dx, dy, dw, dh = offsets[0], offsets[1], offsets[2], offsets[3]

        x_roi, y_roi, w_roi, h_roi = tf.cast(x_roi, tf.float32), tf.cast(y_roi, tf.float32), tf.cast(w_roi, tf.float32), tf.cast(h_roi, tf.float32)

        xmin = (x_roi - dx * w_roi) * w_im
        ymin = (y_roi - dy * h_roi) * h_im
        xmax = (xmin + dw * w_roi) * w_im
        ymax = (ymin + dh * h_roi) * h_im

        x_min = tf.clip_by_value(xmin, 0, w_im - 1)
        y_min = tf.clip_by_value(ymin, 0, h_im - 1)
        x_max = tf.clip_by_value(xmax, 0, w_im - 1)
        y_max = tf.clip_by_value(ymax, 0, h_im - 1)


        return tf.stack([x_min, y_min, x_max, y_max], axis=-1)
        
    
    def nms(self, confidence_scores, boxes):
        selected_indices = tf.image.non_max_suppression(
            boxes,
            confidence_scores,
            max_output_size=10,
            iou_threshold=0.5
        )
        return selected_indices

