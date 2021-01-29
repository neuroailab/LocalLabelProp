from .basics import basic, bs128, load_full, basic_part_cont


def top10_wc_cf_lclw(args):
    args['semi_kNN'] = 10
    args['with_cate'] = True
    args['use_conf'] = True
    args['local_weight_semi_kNN'] = True
    return args


def p10():
    args = {}

    args = basic(args)
    args = bs128(args)
    args = load_full(args)
    args = basic_part_cont(args, 10)
    args = top10_wc_cf_lclw(args)

    args['exp_id'] = 'p10'
    args['lr_boundaries'] = '1941711,2392301,2992601'
    return args


def p05():
    args = {}

    args = basic(args)
    args = bs128(args)
    args = load_full(args)
    args = basic_part_cont(args, 5)
    args = top10_wc_cf_lclw(args)

    args['exp_id'] = 'p05'
    args['lr_boundaries'] = '1921711,2552333,3142811'
    return args


def p03():
    args = {}

    args = basic(args)
    args = bs128(args)
    args = load_full(args)
    args = basic_part_cont(args, 3)
    args = top10_wc_cf_lclw(args)

    args['exp_id'] = 'p03'
    args['lr_boundaries'] = '2522301,3403101,3903601'
    return args


def p01():
    args = {}

    args = basic(args)
    args = bs128(args)
    args = load_full(args)
    args = basic_part_cont(args, 1)
    args = top10_wc_cf_lclw(args)

    args['exp_id'] = 'p01'
    args['lr_boundaries'] = '1761630,2090011,2330011'
    return args
