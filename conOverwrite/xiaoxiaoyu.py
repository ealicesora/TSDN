_base_ = './default.py'

expname = 'vrig/xiaoxiaoyu'
basedir = './logs/vrig_data'

data = dict(
    datadir='./DynamicNeRF-Hyper/xiaoxiaoyu',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)