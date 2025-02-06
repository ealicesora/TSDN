_base_ = './hyper-horse-jump-high.py'

expname = 'vrig/car-shadow'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig_dataset/car-shadow',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    isColMap = False,
)