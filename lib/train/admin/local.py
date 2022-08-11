class EnvironmentSettings:
    def __init__(self):

        self.workspace_dir = '/root/EdgeTrack-master'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/root/EdgeTrack-master/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = 'EdgeTrack-master/pretrained_networks'
        self.lasot_dir = '/home/kb/EdgeTrack-main/data/LaSOTBenchmark'
        self.got10k_dir = '/home/sp3090/kangben/EdgeTrack-main/data/got10k'
        self.lasot_lmdb_dir = '/root/autodl-extra/data/tracking_data/lasot_lmdb'
        self.got10k_lmdb_dir = '/root/autodl-extra/data/tracking_data/got10k_lmdb'
        self.trackingnet_dir = '/root/autodl-extra/data/tracking_data/trackingnet_lmdb'
        self.trackingnet_lmdb_dir = '/root/autodl-extra/data/tracking_data/trackingnet_lmdb'
        self.coco_dir = '/home/sp3090/kangben/EdgeTrack-main/data/coco'
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
