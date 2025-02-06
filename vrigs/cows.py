_base_ = './cows_base.py'

expname = 'vrig/cows'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig_dataset/cows',
    dataset_type='hyper_dataset',
    white_bkgd=False,
        isColMap = False,
)