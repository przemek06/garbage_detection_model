import tensorflow as tf
from tensorflow.keras import layers, Model

class FastRCNN(Model):
    def __init__(self, classes):
        super(FastRCNN, self).__init__()
        self.classes = classes
        self.num_classes = len(classes)
        self._build_model()
        
        self.cls_loss_metric = tf.keras.metrics.Mean(name='cls_loss')
        self.bbox_loss_metric = tf.keras.metrics.Mean(name='bbox_loss')
        self.cls_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy')
    
    def _build_model(self):
        self.backbone = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet',input_shape=(None, None, 3))
        self.conv_reduce = layers.Conv2D(256, (1, 1), activation='relu')       
        self.roi_pooling = layers.MaxPooling2D(pool_size=(7, 7), strides=(1, 1), padding='same')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
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
        rois_feature = rois_scaled / 32.0  
        
        feature_shape = tf.cast(tf.shape(feature_maps)[1:3], tf.float32)  
        ymin = rois_feature[..., 1] / feature_shape[0]
        xmin = rois_feature[..., 0] / feature_shape[1]
        ymax = rois_feature[..., 3] / feature_shape[0]
        xmax = rois_feature[..., 2] / feature_shape[1]
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
    
    @property
    def metrics(self):
        return [self.cls_loss_metric, self.bbox_loss_metric, self.cls_accuracy_metric]
    
    def compile(self, optimizer, cls_loss_fn, bbox_loss_fn, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.cls_loss_fn = cls_loss_fn
        self.bbox_loss_fn = bbox_loss_fn
    
    def train_step(self, data):
        x, y = data
        images, rois = x
        true_cls, true_bbox = y

        with tf.GradientTape() as tape:
            pred_cls, pred_bbox = self((images, rois), training=True)
            
            cls_loss = self.cls_loss_fn(true_cls, pred_cls)
            
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
            
            bbox_loss = tf.cond(
                tf.size(masked_true_bbox) > 0,
                lambda: self.bbox_loss_fn(masked_true_bbox, masked_pred_bbox),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            
            total_loss = cls_loss + bbox_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.cls_loss_metric.update_state(cls_loss)
        self.bbox_loss_metric.update_state(bbox_loss)
        self.cls_accuracy_metric.update_state(true_cls, pred_cls)
        
        return {m.name: m.result() for m in self.metrics}