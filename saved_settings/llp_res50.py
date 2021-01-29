from saved_settings.basics import basic, bs128, load_full50, basic_part_cont
from saved_settings.semi import top10_wc_cf_lclw


def p10():
    args = {}

    args = basic(args)
    args = bs128(args)
    args = load_full50(args)
    args = basic_part_cont(args, 10)
    args = top10_wc_cf_lclw(args)

    args['exp_id'] = 'res50_p10'
    args['resnet_size'] = 50
    args['lr_boundaries'] = '2022011,2562333,2762555'
    return args


def p01():
    args = {}

    args = basic(args)
    args = bs128(args)
    args = load_full50(args)
    args = basic_part_cont(args, 10)
    args = top10_wc_cf_lclw(args)

    args['exp_id'] = 'res50_p01'
    args['resnet_size'] = 50
    args['lr_boundaries'] = '1810011,2470011,2710011'
    return args
