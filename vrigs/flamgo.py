_base_ =  './hyper-horse-jump-high.py'


expname = 'vrig/flamo'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig_dataset/flamo',
    dataset_type='hyper_dataset',
    white_bkgd=False,
        isColMap = False,
)