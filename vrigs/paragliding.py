_base_ = './hyper-horse-jump-high.py'

expname = 'vrig/paragliding'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig_dataset/paragliding',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    isColMap = False,
)