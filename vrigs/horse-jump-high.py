_base_ = './hyper-horse-jump-high.py'

expname = 'vrig/horse-jump-high'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig_dataset/vrig-horse-jump-high',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    isColMap = False,
)