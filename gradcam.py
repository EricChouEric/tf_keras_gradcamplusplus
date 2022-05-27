# Copyright 2020 Samson Woof

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model


def grad_cam(model, img,
             layer_name="block5_conv3", label_name=None,
             category_id=None):
    """Get a heatmap by Grad-CAM.

    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list or None,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    img_tensor = np.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img_tensor)
        if category_id == None:
            category_id = np.argmax(predictions[0])
        if label_name:
            print(label_name[category_id])
        output = predictions[:, category_id]
        grads = gtape.gradient(output, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return np.squeeze(heatmap)

def grad_cam_plus(model, img,
                  layer_name="block5_conv3", label_name=None,
                  category_id=None):
    """Get a heatmap by Grad-CAM++.

    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list or None,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    img_tensor = tf.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)
                if category_id==None:
                    category_id = np.argmax(predictions[0])
                if label_name:
                    print(label_name[category_id])
                output = predictions[:, category_id]
            grads1 = gtape3.gradient(output, conv_output)
        grads2 = gtape2.gradient(grads1, conv_output)
    grads3 = gtape1.gradient(grads2, conv_output)
    feat_sum = tf.reduce_sum(conv_output, axis=(1, 2))
    alpha_denom = grads2*2.0 + grads3*feat_sum
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, 1)
    alpha = grads2/alpha_denom
    # alpha_normalization_constant = tf.reduce_sum(alphas, axis=(1, 2), keepdims=True)
    alpha_normalization_constant = tf.where(alpha == 0., 1, alpha)
    alpha /= alpha_normalization_constant
    weights = tf.reduce_sum(tf.keras.activations.relu(grads1)*alpha, axis=(1, 2), keepdims=True) #weights need to be use reduce sum
    grad_CAM_map = tf.reduce_sum(weights*conv_output, axis=3, keepdims=True)
    heatmap = tf.keras.activations.relu(grad_CAM_map)
    max_heat = tf.reduce_max(heatmap, axis=(1, 2))
    max_heat = tf.where(max_heat == 0., 1, max_heat)
    heatmap /= max_heat
    row = 224
    heatmap = tf.image.resize_with_pad(heatmap, row, row, method='bilinear')
    print(heatmap.shape)
    return heatmap.numpy().reshape(224, 224)

