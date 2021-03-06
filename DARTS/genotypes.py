from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

darts_v2_cifar100 = DARTS_V2

darts_v2_svhn = DARTS_V2

inner_acc1_outer_nop_mgda_temperature = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda_min_temperature = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))


inner_acc1_outer_nop_mgda_min_temperatureC = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda_min_temperatureD = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda_min_tempGumbelA = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda_min_tempGumbelB = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

inner_acc1_outer_adv_nop_mgda_min = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda_min = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda_min_tempGumbelB = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda_min_tempC = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda_tempGumbelA = Genotype(normal=[('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 2), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda_tempA = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('skip_connect', 3), ('skip_connect', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_3x3', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda_tempA = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))

inner_acc1_outer_nop_mgda_tempGumbelA = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))

inner_acc1_outer_adv_nop_mgda_tempA = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))

bigAlpha_inner_acc1_outer_nop_mgda_min = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
bigAlpha_inner_acc1_outer_nop_mgda_min = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
bigAlpha_inner_adv1_acc1_outer_nop_mgda_min30_tempA = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

bigAlpha_inner_adv1_acc1_outer_nop_mgda_min40_tempA = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

bigAlpha_inner_adv1_acc1_outer_nop_mgda_min20_tempA = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

bigAlpha_inner_adv1_acc1_outer_nop_mgda_min10_tempA = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

bigAlpha_inner_adv1_acc1_outer_nop_mgda_min40_tempA = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

bigAlpha_inner_adv1_acc1_outer_nop_mgda_min20_tempA = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_nop_mgda_min0 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_nop_mgda_min0_tempA = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

bigAlphaInit_inner_acc1_outer_nop_mgda_min0 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_nop_mgda_min0_inner_adv1_acc1_outer_nop_mgda_min0 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 2), ('skip_connect', 4), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_nop_mgda_min0_tempA_inner_adv1_acc1_outer_nop_mgda_min0_tempA = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

bigAlphaInit_inner_acc1_outer_nop_mgda_min10 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_nop_mgda_min10_tempA = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_nop_mgda_min10 = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

bigAlphaInit_inner_adv1_acc1_outer_nop_mgda_min10 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 3), ('skip_connect', 4), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

randomInit_inner_adv1_acc1_outer_nop_mgda_min10 = Genotype(normal=[('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 2), ('skip_connect', 4), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

randomInit_inner_adv1_acc1_outer_nop_mgda_min10_tempA = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_nop_mgda_min20 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 1), ('skip_connect', 1), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

bigAlphaInit_inner_acc1_outer_nop_mgda_min20 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 3), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_nop_mgda_min20_tempA = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('skip_connect', 1), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

randomInit_inner_adv1_acc1_outer_nop_mgda_min20 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 2), ('skip_connect', 4), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

bigAlphaInit_inner_adv1_acc1_outer_nop_mgda_min20 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('skip_connect', 4), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 3), ('skip_connect', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

randomInit_inner_adv1_acc1_outer_nop_mgda_min20_tempA = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_nop_mgda_min30 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

bigAlphaInit_inner_acc1_outer_nop_mgda_min30 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_nop_mgda_min30_tempA = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_fxSqr = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

randomInit_inner_adv1_acc1_outer_nop_mgda_min30 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

bigAlphaInit_inner_adv1_acc1_outer_nop_mgda_min30 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_fxCub = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min10 = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_fxExp = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min20 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min10 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_fxSqr = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_cifar100_inner_acc1_outer_adv_nop_mgda_min0 = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

randomInit_cifar100_inner_acc1_outer_adv_nop_mgda_min0_fxSqr = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min30 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min20 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_fxCub = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

randomInit_cifar100_inner_acc1_outer_adv_nop_mgda_min0_fxCub = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_cifar100_inner_acc1_outer_adv_nop_mgda_min10 = Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min30 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_fxExp = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_cifar100_inner_acc1_outer_adv_nop_mgda_min0_fxExp = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_fxTan = Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_nopLater30 = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 3), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_cifar100_inner_acc1_outer_adv_nop_mgda_min0_fxTan = Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_fxSqr_nopLater30 = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_nopLater30_advLater30 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_fxSqr_nopLater30_advLater30 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_fxCub_nopLater30_advLater30 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_min0_fxExp_nopLater30_advLater30 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_min0 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_min10 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_abs30 = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('sep_conv_5x5', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_abs20 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_min20 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_abs10 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_abs30_epoch100 = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_abs40 = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_abs30_gnloss = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_abs20_gnloss = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_abs30 = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_abs40_gnl2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_5x5', 2), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('sep_conv_5x5', 1), ('sep_conv_5x5', 4), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_mgda_abs40_gnlossplus = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_abs20 = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_abs10 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

randomInit_cifar100_inner_acc1_outer_adv_nop_mgda_abs30 = Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

randomInit_cifar100_inner_acc1_outer_adv_nop_abs30 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_ood_mgda_abs30 = Genotype(normal=[('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('skip_connect', 2), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))

randomInit_cifar100_inner_acc1_outer_adv_nop_mgda_abs10 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_ood_mgda_abs20 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 1), ('sep_conv_5x5', 3), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

randomInit_cifar100_inner_acc1_outer_adv_nop_ood_flp_mgda_abs30 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_ood_mgda_abs10 = Genotype(normal=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_flp_mgda_abs30 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_ood_flp_mgda_abs30 = Genotype(normal=[('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_ood1_outer_adv_nop_flp_mgda_abs30 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 4), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_ood1_outer_adv_nop_ood_flp_mgda_abs30 = Genotype(normal=[('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_ood_flp_mgda_abs30 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_outer_adv_nop_ood_flp_mgda_abs20 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('skip_connect', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 4), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

randomInit_inner_acc1_ood1_outer_adv_nop_ood_flp_mgda_abs20 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('sep_conv_5x5', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

LL_acc1_UL_adv_nop_ood_flp_mgda_abs30_gnl2 = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 3), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))

randomInit_inner_acc1_ood1_outer_adv_nop_flp_mgda_abs20 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_flp_mgda_abs30_gnl2 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 3), ('skip_connect', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_flp_mgda_abs30_gnl2_cifar100 = Genotype(normal=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_flp_mgda_abs30_gnl2 = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 3), ('skip_connect', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

LL_acc1_UL_adv_nop_ood_flp_mgda_abs30_gnl2 = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 3), ('skip_connect', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_flp_abs30_gnl2 = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_flp_mgda_abs40_gnl2 = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_5x5', 3), ('skip_connect', 0), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_flp_abs40_gnl2 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3), ('avg_pool_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))

LL_acc1_UL_adv_nop_ood_flp_mgda_abs40_gnl2 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_flp_mgda_abs20_gnl2 = Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_flp_abs20_gnl2 = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))

LL_acc1_UL_adv_nop_ood_flp_mgda_abs20_gnl2 = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 0), ('skip_connect', 4), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_flp_mgda_abs40_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_UL_adv_nop_ood_flp_mgda_abs30_gnl2 = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 0), ('sep_conv_5x5', 4), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_UL_adv_nop_ood_flp_mgda_abs40_gnl2 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 2), ('skip_connect', 3), ('sep_conv_5x5', 4), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs40_gnl2 = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_oodS1_UL_adv_nop_flp_mgda_abs40_gnl2 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_UL_adv_nop_ood_flp_mgda_abs40_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_oodS1_UL_adv_nop_ood_flp_mgda_abs40_gnl2 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs50_gnl2 = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_oodS1_UL_adv_nop_flp_mgda_abs50_gnl2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_UL_adv_nop_ood_flp_mgda_abs50_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_oodS1_UL_adv_nop_flp_mgda_abs60_gnl2 = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_UL_adv_nop_ood_flp_mgda_abs60_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_oodS1_UL_adv_nop_ood_flp_mgda_abs50_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_oodS1_UL_adv_nop_ood_flp_mgda_abs60_gnl2_cifar100 = Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs48_gnl2 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs50_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs60_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs60_gnl2 = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 4), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs40_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))

LL_adv1_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs60_gnl2 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_ood_flp_abs50_gnl2 = Genotype(normal=[('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 3), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_ood_flp_abs40_gnl2 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 3), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

woOodAug_LL_acc1_UL_adv_nop_ood_flp_mgda_abs50_gnl2 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))

LL_acc1_ood1_UL_adv_nop_ood_flp_abs60_gnl2 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

LL_adv1_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs40_gnl2 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))

lossNorm_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs70_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))

lossNorm_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs60_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

lossNorm_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs55_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

lossNorm_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs75_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

lossNorm_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs50_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

lossNorm_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs100_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 4), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

lossNorm_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs90_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 3), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

layerNorm_LL_acc1_UL_adv_nop_ood_flp_mgda_abs100_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 3), ('skip_connect', 2), ('skip_connect', 2), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

layerNorm_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs100_gnl2_cifar100 = Genotype(normal=[('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('skip_connect', 0), ('skip_connect', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

lossNorm_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs80_gnl2_cifar100 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

layerNorm_LL_acc1_ood1_UL_adv_nop_ood_flp_mgda_abs80_gnl2_cifar100 = Genotype(normal=[('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))

layerNorm_LL_acc1_UL_adv_nop_ood_flp_mgda_abs90_gnl2_cifar100 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 3), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
