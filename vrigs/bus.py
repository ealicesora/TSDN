_base_ = './hyper-horse-jump-high.py'

expname = 'vrig/bus'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig_dataset/bus',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    isColMap = False,
)