_base_ = './hyper-horse-jump-high.py'

expname = 'vrig/rhino'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig_dataset/rhino',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    isColMap = False,
)