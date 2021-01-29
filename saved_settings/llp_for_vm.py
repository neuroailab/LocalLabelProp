from .basics import basic_part_cont
from . import llp_res18 as prev_llp


def p01_s1():
    args = prev_llp.p01()
    args['exp_id'] = 'p01_s1'
    return args


def p01_s2():
    args = prev_llp.p01()
    args['exp_id'] = 'p01_s2'
    return args


def p03_s1():
    args = prev_llp.p03()
    args['exp_id'] = 'p03_s1'
    return args


def p03_s2():
    args = prev_llp.p03()
    args['exp_id'] = 'p03_s2'
    return args


def p05_s1():
    args = prev_llp.p05()
    args['exp_id'] = 'p05_s1'
    return args


def p05_s2():
    args = prev_llp.p05()
    args['exp_id'] = 'p05_s2'
    return args


def p10_s1():
    args = prev_llp.p10()
    args['exp_id'] = 'p10_s1'
    return args


def p10_s2():
    args = prev_llp.p10()
    args['exp_id'] = 'p10_s2'
    return args


def p02_s0():
    args = prev_llp.p10()
    args = basic_part_cont(args, 2)
    args['exp_id'] = 'p02_s0'
    return args


def p02_s1():
    args = p02_s0()
    args['exp_id'] = 'p02_s1'
    return args


def p02_s2():
    args = p02_s0()
    args['exp_id'] = 'p02_s2'
    return args


def p04_s0():
    args = prev_llp.p10()
    args = basic_part_cont(args, 4)
    args['exp_id'] = 'p04_s0'
    return args


def p04_s1():
    args = p04_s0()
    args['exp_id'] = 'p04_s1'
    return args


def p04_s2():
    args = p04_s0()
    args['exp_id'] = 'p04_s2'
    return args


def p06_s0():
    args = prev_llp.p10()
    args = basic_part_cont(args, 6)
    args['exp_id'] = 'p06_s0'
    return args


def p06_s1():
    args = p06_s0()
    args['exp_id'] = 'p06_s1'
    return args


def p06_s2():
    args = p06_s0()
    args['exp_id'] = 'p06_s2'
    return args


def p20_s0():
    args = prev_llp.p10()
    args = basic_part_cont(args, 20)
    args['exp_id'] = 'p20_s0'
    return args


def p20_s1():
    args = p20_s0()
    args['exp_id'] = 'p20_s1'
    return args


def p20_s2():
    args = p20_s0()
    args['exp_id'] = 'p20_s2'
    return args


def p50_s0():
    args = prev_llp.p10()
    args = basic_part_cont(args, 50)
    args['exp_id'] = 'p50_s0'
    return args


def p50_s1():
    args = p50_s0()
    args['exp_id'] = 'p50_s1'
    return args


def p50_s2():
    args = p50_s0()
    args['exp_id'] = 'p50_s2'
    return args
