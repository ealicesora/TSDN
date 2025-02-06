_base_ = './hyper_default.py'

expname = 'vrig/base-horse'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig_dataset/horse',
    dataset_type='hyper_dataset',
    white_bkgd=False,
        use_bg_points=False,
    add_cam=True,
)