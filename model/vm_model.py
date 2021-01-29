import os
import sys
import pdb
import tensorflow as tf
from argparse import Namespace
from . import resnet_th_preprocessing

sys.path.append(os.path.expanduser('~/visualmaster/combine_pred'))
sys.path.append(os.path.expanduser('~/visualmaster/normal_pred'))
import combinet_builder as cb
import cmd_parser
from models.config_parser import get_network_cfg
import models.mean_teacher_utils as mt_utils


def get_network_cfg_from_setting(setting_name):
    train_args = Namespace()
    train_args.load_setting_func = setting_name
    train_args = cmd_parser.load_setting(train_args)
    train_args.network_func_kwargs = getattr(
            train_args, 
            'network_func_kwargs', None)

    network_cfg = get_network_cfg(train_args)
    return network_cfg, train_args


def get_network_outputs(input_image, setting_name, want_key='encode_9'):
    network_cfg, args = get_network_cfg_from_setting(setting_name)

    def _build_network():
        combined_orders = ['encode']

        output_dict = {}
        for network_name in combined_orders:
            _, output_dict = cb.build_partnet(
                    input_image, 
                    cfg_initial=network_cfg, 
                    key_want=network_name, 
                    batch_name='_imagenet', 
                    all_out_dict=output_dict, 
                    init_type='xavier',
                    weight_decay=None, 
                    train=False,
                    ignorebname_new=1,
                    tpu_flag=1,
                    fixweights=False,
                    sm_bn_trainable=True)
        return output_dict

    if not args.mean_teacher:
        all_outs = _build_network()
        return all_outs[want_key]
    else:
        with mt_utils.name_variable_scope(
                "primary", "primary", reuse=tf.AUTO_REUSE) \
                        as (name_scope, var_scope):
            primary_outs = _build_network()

        # Build the teacher model using ema_variable_scope
        with mt_utils.ema_variable_scope(
                "ema", var_scope, decay=0.9997, 
                zero_debias=False, reuse=tf.AUTO_REUSE):
            ema_outs = _build_network()

        update_ops = tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)
        for each_v in update_ops:
            update_ops.remove(each_v)
        return ema_outs[want_key]


if __name__ == '__main__':
    setting_name = 'mt_part10_res18_fx'
    input_image = tf.zeros([1, 224, 224, 3], dtype=tf.float32)
    network_outputs = get_network_outputs(input_image, setting_name)
    print(network_outputs)
