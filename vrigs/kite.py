_base_ = './hyper-horse-jump-high.py'

expname = 'vrig/kite'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig_dataset/kite',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    isColMap = False,
)