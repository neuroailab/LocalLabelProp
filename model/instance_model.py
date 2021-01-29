from __future__ import division, print_function, absolute_import
import os, sys
import json
import numpy as np
import tensorflow as tf
import copy
import pdb
from collections import OrderedDict

from . import resnet_model
from . import alexnet_model
from . import vggnet_model
from .memory_bank import MemoryBank
from .self_loss import get_selfloss, assert_shape, DATA_LEN_IMAGENET_FULL
from .resnet_th_preprocessing import ColorNormalize


def get_cate_loss_accuracy(
        curr_output, input_label, num_classes=1000, 
        get_top5=False,
        **kwargs):
    if not get_top5:
        _, pred = tf.nn.top_k(curr_output, k=1)
        pred = tf.cast(tf.squeeze(pred), tf.int64)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(pred, input_label), tf.float32)
        )
    else:
        accuracy = tf.nn.in_top_k(curr_output, input_label, k=5)

    one_hot_labels = tf.one_hot(input_label, num_classes)
    loss = tf.losses.softmax_cross_entropy(
            one_hot_labels, curr_output, **kwargs)
    return loss, accuracy


def get_clstr_labels_and_index(clstr_path, indx_for_clstr):
    assert os.path.exists(clstr_path) and clstr_path.endswith('npy'), \
            "Cluster file does not exist or end with npy!"
    clstr_labels = np.load(clstr_path)
    num_of_labels = len(clstr_labels)
    if not indx_for_clstr:
        print('Will assume the labels are for the first part of images!')
        label_index = np.arange(num_of_labels)
    else:
        label_index = np.load(indx_for_clstr)
        assert len(label_index) == num_of_labels, \
                "Numbers of labels and indexes do not match!"
    return clstr_labels, label_index


def flatten(layer_out):
    curr_shape = layer_out.get_shape().as_list()
    if len(curr_shape) > 2:
        layer_out = tf.reshape(layer_out, [curr_shape[0], -1])
    return layer_out


def get_resnet_all_layers(ending_points, get_all_layers):
    ending_dict = OrderedDict()
    get_all_layers = get_all_layers.split(',')
    for idx, layer_out in enumerate(ending_points):
        if 'all_spatial' not in get_all_layers:
            if str(idx) in get_all_layers:
                layer_out = flatten(layer_out)
                ending_dict[str(idx)] = layer_out
            avg_name = '%i-avg' % idx
            if avg_name in get_all_layers:
                layer_out = tf.reduce_mean(layer_out, axis=[2,3])
                ending_dict[avg_name] = layer_out
        else:
            layer_out = tf.transpose(layer_out, [0,2,3,1])
            layer_name = 'encode_{layer_idx}'.format(layer_idx = idx+1)
            ending_dict[layer_name] = layer_out
    return ending_dict


def resnet_embedding(img_batch, dtype=tf.float32,
                     data_format=None, train=False,
                     resnet_size=18,
                     model_type='resnet',
                     resnet_version=resnet_model.DEFAULT_VERSION,
                     get_all_layers=None,
                     skip_final_dense=False,
                     with_cate=False,
):
    image = tf.cast(img_batch, tf.float32)
    image = tf.div(image, tf.constant(255, dtype=tf.float32))
    image = tf.map_fn(ColorNormalize, image)

    if model_type == 'resnet':
        model = resnet_model.ImagenetModel(
            resnet_size, data_format,
            resnet_version=resnet_version,
            dtype=dtype,
            with_cate=with_cate)

        if skip_final_dense and get_all_layers is None:
            return model(image, train, skip_final_dense=True)

        if skip_final_dense and get_all_layers.startswith('Mid-'):
            mid_layer_units = get_all_layers[4:]
            mid_layer_units = mid_layer_units.split(',')
            mid_layer_units = [int(each_unit) for each_unit in mid_layer_units]

            resnet_output = model(image, train, skip_final_dense=True)
            all_mid_layers = OrderedDict()
            with tf.variable_scope('instance', reuse=tf.AUTO_REUSE):
                init_builder = tf.contrib.layers.variance_scaling_initializer()
                for each_mid_unit in mid_layer_units:
                    now_name = 'mid{units}'.format(units=each_mid_unit)
                    mid_output = tf.layers.dense(
                            inputs=resnet_output, units=each_mid_unit,
                            kernel_initializer=init_builder,
                            trainable=True,
                            name=now_name)
                    all_mid_layers[now_name] = mid_output
            return all_mid_layers

        if get_all_layers:
            _, ending_points = model(
                    image, train, get_all_layers=get_all_layers)
            all_layers = get_resnet_all_layers(ending_points, get_all_layers)
            return all_layers

        model_out = model(image, train, skip_final_dense=False)
    else:
        raise ValueError('Model type not supported!')

    if not with_cate:
        return tf.nn.l2_normalize(model_out, axis=1) # [bs, out_dim]
    else:
        # First is the memory vector, second is the category prediction
        return tf.nn.l2_normalize(model_out[0], axis=1), model_out[1]


def repeat_1d_tensor(t, num_reps):
    ret = tf.tile(tf.expand_dims(t, axis=1), (1, num_reps))
    return ret


def get_scattered_label_index(index_for_labeled, label_for_labeled, data_len):
    index_for_labeled_ext = tf.expand_dims(index_for_labeled, axis=1)
    have_labels = tf.scatter_nd(
            index_for_labeled_ext,
            tf.ones(index_for_labeled.shape),
            shape=[data_len])
    have_labels = tf.cast(have_labels, tf.bool)
    semi_labels_scattered = tf.scatter_nd(
            index_for_labeled_ext,
            tf.cast(label_for_labeled, tf.int64),
            shape=[data_len])
    return have_labels, semi_labels_scattered, index_for_labeled_ext


class SemiModel(object):
    def __init__(self,
                 inputs, output,
                 memory_bank,
                 semi_psd_labels,
                 cate_output=None,
                 instance_k=4096,
                 instance_t=0.07,
                 instance_m=0.5,
                 semi_kNN=None,
                 use_conf=False,
                 local_weight_semi_kNN=False,
                 local_weight_K=50,
                 local_weight_for_labeled=None,
                 **kwargs):
        self.inputs = inputs
        self.embed_output = output
        self.batch_size, self.out_dim = self.embed_output.get_shape().as_list()
        self.memory_bank = memory_bank
        self.semi_psd_labels = semi_psd_labels
        self.semi_kNN = semi_kNN or 1
        self.use_conf = use_conf

        self.instance_data_len = memory_bank.size
        self.instance_k = instance_k
        self.instance_t = instance_t
        self.instance_m = instance_m

        self.new_semi_labels = None
        self.curr_conf = None
        if use_conf:
            self.confidence = tf.get_variable(
                    'confidence',
                    initializer=tf.ones_initializer,
                    shape=(self.instance_data_len,),
                    trainable=False,
                    dtype=tf.float32)

        self.local_weight_semi_kNN = local_weight_semi_kNN
        self.local_weight_for_labeled = local_weight_for_labeled
        self.local_weight_K = local_weight_K

        self.cate_output = cate_output

        self.all_dps = self.memory_bank.get_all_dot_products(self.embed_output)
        self.new_data_memory = self.updated_new_data_memory()

    def _softmax(self, dot_prods):
        instance_Z = tf.constant(
            2876934.2 / 1281167 * self.instance_data_len,
            dtype=tf.float32)
        return tf.exp(dot_prods / self.instance_t) / instance_Z

    def updated_new_data_memory(self):
        data_indx = self.inputs['index'] # [bs]
        data_memory = self.memory_bank.at_idxs(data_indx)
        new_data_memory = (data_memory * self.instance_m
                           + (1 - self.instance_m) * self.embed_output)
        return tf.nn.l2_normalize(new_data_memory, axis=1)

    def __get_lbl_equal(
            self, curr_cluster_labels, 
            top_idxs, k):
        batch_labels = tf.gather(
                curr_cluster_labels, 
                self.inputs['index'])
        if k > 0:
            top_cluster_labels = tf.gather(curr_cluster_labels, top_idxs)
            batch_labels = repeat_1d_tensor(batch_labels, k)
            curr_equal = tf.equal(batch_labels, top_cluster_labels)
        else:
            curr_equal = tf.equal(
                    tf.expand_dims(batch_labels, axis=1), 
                    tf.expand_dims(curr_cluster_labels, axis=0))
        return curr_equal

    def __get_prob_from_equal(self, curr_equal, exponents):
        each_prob = tf.where(
                curr_equal,
                x=exponents, y=tf.zeros_like(exponents))
        probs = tf.reduce_sum(each_prob, axis=1)
        probs /= tf.reduce_sum(exponents, axis=1)
        return probs

    def __get_part_dps_from_index(self, index):
        part_dps = tf.gather(self.all_dps, index, axis=1)
        return part_dps

    def __get_top_prob_labels_frm_semi(
            self, index_for_labeled, label_for_labeled):
        part_dps = self.__get_part_dps_from_index(index_for_labeled)

        if not self.local_weight_semi_kNN:
            top_dist, top_indices = tf.nn.top_k(
                    part_dps, k=self.semi_kNN, sorted=False)
            top_prob = tf.exp(top_dist / self.instance_t)
        else:
            part_probs = tf.exp(part_dps / self.instance_t)
            part_probs *= tf.expand_dims(self.local_weight_for_labeled, axis=0)
            top_prob, top_indices = tf.nn.top_k(
                    part_probs, k=self.semi_kNN, sorted=False)

        top_labels = tf.gather(label_for_labeled, top_indices)
        top_labels -= self.instance_data_len
        top_labels = tf.cast(top_labels, tf.int64)
        return top_prob, top_labels

    def __get_kNN_pred(self, index_for_labeled, label_for_labeled):
        top_prob, top_labels = self.__get_top_prob_labels_frm_semi(
                index_for_labeled, label_for_labeled)

        # TODO: make num_classes changable
        top_labels_one_hot = tf.one_hot(top_labels, 1000)
        top_labels_one_hot *= tf.expand_dims(top_prob, axis=-1)
        top_labels_one_hot = tf.reduce_mean(top_labels_one_hot, axis=1)
        curr_conf, curr_pred = tf.nn.top_k(top_labels_one_hot, k=1)
        curr_conf = tf.squeeze(curr_conf, axis=1)
        curr_conf /= tf.reduce_sum(top_labels_one_hot, axis=1)
        curr_pred = tf.squeeze(tf.cast(curr_pred, tf.int64), axis=1)
        curr_pred += self.instance_data_len
        return curr_conf, curr_pred

    def __overwrite_semi_labels_set_conf(
            self, curr_pred, curr_conf,
            index_for_labeled, label_for_labeled):
        have_labels, semi_labels_scattered, index_for_labeled_ext \
                = get_scattered_label_index(
                        index_for_labeled, label_for_labeled, 
                        self.instance_data_len)

        now_have_labels = tf.gather(have_labels, self.inputs['index'])
        now_semi_labels = tf.gather(
                semi_labels_scattered, self.inputs['index'])
        curr_pred = tf.where(
                now_have_labels, 
                now_semi_labels, curr_pred)

        if self.use_conf:
            semi_conf = tf.scatter_nd(
                    index_for_labeled_ext,
                    tf.ones(index_for_labeled.shape),
                    shape=[self.instance_data_len])
            semi_conf = tf.cast(semi_conf, curr_conf.dtype)
            now_semi_conf = tf.gather(semi_conf, self.inputs['index'])
            self.curr_conf = tf.where(
                    now_have_labels, 
                    now_semi_conf, curr_conf)
        return curr_pred

    def get_new_cate_semi_labels_conf(
            self, index_for_labeled, label_for_labeled):
        softmax_cate_out = tf.nn.softmax(self.cate_output)
        curr_conf, curr_pred = tf.nn.top_k(softmax_cate_out, k=1)
        curr_pred = tf.cast(tf.squeeze(curr_pred), tf.int64)
        curr_conf = tf.squeeze(curr_conf, axis=1)

        new_semi_labels = self.__overwrite_semi_labels_set_conf(
                curr_pred, curr_conf, 
                index_for_labeled, label_for_labeled)
        return new_semi_labels

    def get_new_KNN_semi_labels_conf(
            self, index_for_labeled, label_for_labeled):
        if self.new_semi_labels is not None:
            return self.new_semi_labels

        curr_conf, curr_pred = self.__get_kNN_pred(
                index_for_labeled, label_for_labeled)

        self.new_semi_labels = self.__overwrite_semi_labels_set_conf(
                curr_pred, curr_conf, 
                index_for_labeled, label_for_labeled)
        return self.new_semi_labels

    def __get_corresponding_conf(self):
        corresponding_conf = tf.gather(
                self.confidence, self.inputs['index'])
        return corresponding_conf

    def _apply_conf(self, loss):
        assert_shape(loss, [self.batch_size])
        corresponding_conf = self.__get_corresponding_conf()
        loss *= corresponding_conf
        return loss

    def _apply_complement_conf(self, loss):
        assert_shape(loss, [self.batch_size])
        corresponding_conf = self.__get_corresponding_conf()
        loss *= (1 - corresponding_conf)
        return loss

    def get_categorization_loss(self):
        cate_label = tf.gather(
                self.semi_psd_labels[0], 
                self.inputs['index'])
        loss_cate, _ = get_cate_loss_accuracy(
                self.cate_output, cate_label - self.instance_data_len, 
                reduction=tf.losses.Reduction.NONE)
        loss_cate = tf.where(
                tf.greater_equal(cate_label, self.instance_data_len),
                loss_cate,
                tf.zeros_like(loss_cate))

        if self.use_conf:
            loss_cate = self._apply_conf(loss_cate)

        loss_cate = tf.reduce_mean(loss_cate)
        return loss_cate

    def get_cluster_classification_loss(
            self, cluster_labels, k=None):
        if not k:
            k = self.instance_k
        # ignore all but the top k nearest examples
        top_dps, top_idxs = tf.nn.top_k(self.all_dps, k=k, sorted=False)
        if k > 0:
            exponents = self._softmax(top_dps)
        else:
            exponents = self._softmax(self.all_dps)

        no_kmeans = cluster_labels.get_shape().as_list()[0]
        all_equal = None
        for each_k_idx in range(no_kmeans):
            curr_equal = self.__get_lbl_equal(
                    cluster_labels[each_k_idx], top_idxs, k)

            if all_equal is None:
                all_equal = curr_equal
            else:
                all_equal = tf.logical_or(all_equal, curr_equal)
        probs = self.__get_prob_from_equal(all_equal, exponents)

        assert_shape(probs, [self.batch_size])
        loss = -tf.log(probs + 1e-7)
        return loss
    
    def get_semi_classification_loss(self, k=None):
        loss = self.get_cluster_classification_loss(self.semi_psd_labels, k)
        if self.use_conf:
            loss = self._apply_conf(loss)
        loss = tf.reduce_mean(loss)
        return loss
    
    def get_local_weight_update_op(self, index_for_labeled):
        local_weight_loop_index = tf.get_variable(
                    'local_weight_loop_index',
                    initializer=tf.zeros_initializer,
                    shape=(), trainable=False, dtype=tf.int64)
        # TODO: make this flexible
        local_weight_update_fre = 64
        all_labeled_mb = self.memory_bank.at_idxs(index_for_labeled)

        num_of_labeled = self.local_weight_for_labeled.get_shape().as_list()[0]
        curr_batch_index \
                = tf.range(local_weight_update_fre, dtype=tf.int64) \
                  + local_weight_loop_index
        curr_batch_index = tf.mod(curr_batch_index, num_of_labeled)

        curr_batch_labeled_mb = tf.gather(
                all_labeled_mb, curr_batch_index, axis=0)
        curr_labeled_all_dist = self.memory_bank.get_all_dot_products(
                curr_batch_labeled_mb)
        if self.local_weight_K > 0:
            curr_labeled_top_dist, _ = tf.nn.top_k(
                    curr_labeled_all_dist, 
                    k=self.local_weight_K, sorted=False)
        else:
            curr_labeled_top_dist = curr_labeled_all_dist
        curr_labeled_top_prob = tf.exp(curr_labeled_top_dist / self.instance_t)

        new_local_weight = 1 / tf.reduce_sum(curr_labeled_top_prob, axis=1)
        new_loop_index = local_weight_loop_index + local_weight_update_fre
        new_loop_index = tf.mod(new_loop_index, num_of_labeled)

        loop_index_update_op = tf.assign(
                local_weight_loop_index, new_loop_index)
        local_weight_update_op = tf.scatter_update(
                self.local_weight_for_labeled,
                curr_batch_index, new_local_weight)
        with tf.control_dependencies(
                [loop_index_update_op, local_weight_update_op]):
            update_no_op = tf.no_op()
        return [update_no_op, \
                (num_of_labeled // local_weight_update_fre) + 1]
    
    def get_local_semi_loss(self, cluster_labels, k=None):
        loss = self.get_cluster_classification_loss(cluster_labels, k)
        assert_shape(loss, [self.batch_size])
        if self.use_conf:
            loss = self._apply_complement_conf(loss)
        loss = tf.reduce_mean(loss)
        return loss


def build_output(
        inputs, train, 
        resnet_size=18,
        model_type='resnet',
        clstr_path=None,
        index_path=None,
        with_cate=False,
        **kwargs):
    # This will be stored in the db
    logged_cfg = {'kwargs': kwargs}

    data_len = kwargs.get('instance_data_len', DATA_LEN_IMAGENET_FULL)
    
    local_weight_semi_kNN = kwargs.get('local_weight_semi_kNN', False)
    local_weight_for_labeled = None
    label_for_labeled, index_for_labeled = get_clstr_labels_and_index(
            clstr_path, index_path)
    label_for_labeled += data_len

    def _build_all_labels_memory_bank():
        all_labels = tf.get_variable(
            'all_labels',
            initializer=tf.zeros_initializer,
            shape=(data_len,),
            trainable=False,
            dtype=tf.int64)
        # TODO: hard-coded output dimension 128
        memory_bank = MemoryBank(data_len, 128)
        return all_labels, memory_bank

    with tf.variable_scope('instance', reuse=tf.AUTO_REUSE):
        all_labels, memory_bank = _build_all_labels_memory_bank()

        lbl_init_values = tf.range(data_len, dtype=tf.int64)
        semi_lbl_init_values = tf.tile(
                tf.expand_dims(lbl_init_values, axis=0),
                [1, 1])
        semi_psd_labels = tf.get_variable(
            'cluster_labels',
            initializer=semi_lbl_init_values,
            trainable=False, dtype=tf.int64)

        if local_weight_semi_kNN:
            local_weight_for_labeled = tf.get_variable(
                    'local_weight_for_labeled',
                    initializer=tf.ones_initializer,
                    shape=index_for_labeled.shape,
                    trainable=False, dtype=tf.float32)

    output = resnet_embedding(
            inputs['image'], train=train, 
            resnet_size=resnet_size,
            model_type=model_type,
            with_cate=with_cate)
    if with_cate:
        output, cate_output = output
    else:
        cate_output = None

    if not train:
        all_dist = memory_bank.get_all_dot_products(output)
        ret_list = [all_dist, all_labels]
        if with_cate:
            _, accuracy_cate = get_cate_loss_accuracy(
                    cate_output, inputs['label'])
            ret_list.append(accuracy_cate)
        return ret_list, logged_cfg

    model_class = SemiModel(
        inputs=inputs, output=output,
        memory_bank=memory_bank,
        cate_output=cate_output,
        local_weight_for_labeled=local_weight_for_labeled,
        semi_psd_labels=semi_psd_labels,
        **kwargs)

    loss = model_class.get_semi_classification_loss()

    new_data_memory = model_class.new_data_memory
    ret_dict = {
        "loss": loss,
        "data_indx": inputs['index'],
        "memory_bank": memory_bank.as_tensor(),
        "new_data_memory": new_data_memory,
        "all_labels": all_labels,
    }
    if with_cate:
        loss_cate = model_class.get_categorization_loss()
        ret_dict["loss"] += loss_cate
        ret_dict["loss_cate"] = loss_cate

    ret_dict['semi_psd_labels'] = semi_psd_labels
    ret_dict['new_semi_psd_labels'] \
            = model_class.get_new_cate_semi_labels_conf(
                    index_for_labeled, label_for_labeled)

    # For update conf
    if model_class.use_conf:
        ret_dict["confidence"] = model_class.confidence
        ret_dict["new_conf"] = model_class.curr_conf
    # For update local weights for labeled images
    if model_class.local_weight_semi_kNN:
        ret_dict["local_weight_update_op"] \
                = model_class.get_local_weight_update_op(index_for_labeled)
    return ret_dict, logged_cfg
