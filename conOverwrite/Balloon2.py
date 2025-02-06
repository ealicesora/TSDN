_base_ = './default.py'

expname = 'vrig/Balloon2'
basedir = './logs/vrig_data'

data = dict(
    datadir='./DynamicNeRF-Hyper/Balloon2',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)