_base_ = './hyper-horse-jump-high.py'

expname = 'vrig/soccerball'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig_dataset/soccerball',
    dataset_type='hyper_dataset',
    white_bkgd=False,
    isColMap = False,
)