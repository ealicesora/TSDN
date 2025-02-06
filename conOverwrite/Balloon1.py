_base_ = './default.py'

expname = 'vrig/Balloon1'
basedir = './logs/vrig_data'

data = dict(
    datadir='./DynamicNeRF-Hyper/Balloon1',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)