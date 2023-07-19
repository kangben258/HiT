class EnvironmentSettings:
    def __init__(self):

        self.workspace_dir = '/home/kb/kb/HiT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/kb/kb/HiT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/kb/kb/HiT/pretrained_networks'
        self.lasot_dir = '/media/kb/08AC1CC7272DCD64/trackingdata/train/LaSOTBenchmark'
        self.got10k_dir = '/media/kb/08AC1CC7272DCD64/trackingdata/train/GOT10K/train'
        self.lasot_lmdb_dir = '/root/autodl-extra/data/tracking_data/lasot_lmdb'
        self.got10k_lmdb_dir = '/root/autodl-extra/data/tracking_data/got10k_lmdb'
        self.trackingnet_dir = '/media/kb/08AC1CC7272DCD64/trackingdata/train/TrackingNet'
        self.trackingnet_lmdb_dir = '/root/autodl-extra/data/tracking_data/trackingnet_lmdb'
        self.coco_dir = '/media/kb/08AC1CC7272DCD64/trackingdata/train/COCO2017'
        self.coco_lmdb_dir = '/root/autodl-extra/data/tracking_data/coco_lmdb'
        self.imagenet1k_dir = '/home/kb/EdgeTrack-main/data/imagenet1k'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/kb/EdgeTrack-main/data/vid'
        self.imagenet_lmdb_dir = '/home/kb/EdgeTrack-main/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
