from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/kb/EdgeTrack-main/data/got10k_lmdb'
    settings.got10k_path = '/root/autodl-tmp/got10k_test'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/root/autodl-extra/data/tracking_data/lasot_lmdb'
    settings.lasot_path = '/root/autodl-extra/data/tracking_data/lasot_lmdb'
    settings.network_path = '/root/EdgeTrack-master/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/kb/EdgeTrack-main/data/nfs'
    settings.otb_path = '/home/kb/EdgeTrack-main/data/OTB2015'
    settings.prj_dir = '/root/EdgeTrack-master'
    settings.result_plot_path = '/root/EdgeTrack-master/test/result_plots'
    settings.results_path = '/root/EdgeTrack-master/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/root/autodl-extra'
    settings.segmentation_path = '/home/kb/EdgeTrack-main/test/segmentation_results'
    settings.tc128_path = '/home/kb/EdgeTrack-main/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/root/autodl-tmp/TrackingNet'
    settings.uav_path = '/home/kb/EdgeTrack-main/data/UAV123'
    settings.vot_path = '/home/kb/EdgeTrack-main/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

