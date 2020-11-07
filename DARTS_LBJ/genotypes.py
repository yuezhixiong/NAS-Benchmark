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


DARTS_mgda = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))


DARTS_uniform = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

DARTS = DARTS_uniform

DARTS = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

fgsm_nop_max_etp_cutout_batchsize = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 3), ('avg_pool_3x3', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 3), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

search_ablation_nop = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

search_ablation_adv = Genotype(normal=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
pgd_nop_max_etp_cutout_batchsize = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))
adv_nop = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
ablation_mgda = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('avg_pool_3x3', 3), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))
adv_nop_min = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))

adv_nop_cifar100 = Genotype(normal=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))
ablation_gradnorm = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('avg_pool_3x3', 3), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))

fgsm_nop_min = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 2), ('skip_connect', 4), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

fgsm_nop_min5e5 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 2), ('skip_connect', 4), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
pgd_nop_min = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

fgsm_nop_min_cifar100 = Genotype(normal=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('skip_connect', 4), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

pgd_nop_min_cifar100 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

darts_v2_cifar100 = DARTS_V2

darts_v2_svhn = DARTS_V2

fgsm_nop_min_svhn = Genotype(normal=[('skip_connect', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 3), ('skip_connect', 4), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 3), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

fgsm_nop_min5e5 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 2), ('skip_connect', 4), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

fgsm_nop_min8e5 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('skip_connect', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

nop_original = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

fgsm_original = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('avg_pool_3x3', 3), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))

noMGDA_original = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('skip_connect', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 4), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))


fgsm_nop_min12e5 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('skip_connect', 3), ('sep_conv_3x3', 1), ('skip_connect', 4), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

nop = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('avg_pool_3x3', 3), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))

nop_min1e6 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 4), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

fgsm_nop_min1e6_outer = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
