import os
DATA_DIR = os.environ.get('DATA_DIR', '/data5/chengxuz/Dataset/llp_pub')
MG_PORT = int(os.environ.get('MG_PORT', '26001'))


def basic(args):
    args['port'] = MG_PORT
    args['db_name'] = 'llp_pub'
    args['col_name'] = 'test2'
    args['image_dir'] = os.path.join(DATA_DIR, 'tfrs')
    return args


def bs128(args):
    args['fre_filter'] = 100090
    args['fre_cache_filter'] = 10009
    return args


def load_full(args):
    args['load_exp'] = 'la_pub/test/res18_IR'
    args['load_port'] = MG_PORT
    args['load_step'] = 100090
    return args


def load_full50(args):
    args['load_exp'] = 'la_pub/test/res50_IR'
    args['load_port'] = MG_PORT
    args['load_step'] = 100090
    return args


def basic_part_cont(args, percent):
    args['clstr_path'] = os.path.join(DATA_DIR, 'metas', 'p%02i_label.npy' % percent)
    args['index_path'] = os.path.join(DATA_DIR, 'metas', 'p%02i_index.npy' % percent)
    args['instance_k'] = 0
    return args
