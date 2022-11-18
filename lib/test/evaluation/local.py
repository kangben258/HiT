from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/data/got10k_lmdb'
    settings.got10k_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/data/lasot_lmdb'
    settings.lasot_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/data/lasot'
    settings.network_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/data/nfs'
    settings.otb_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/data/OTB2015'
    settings.prj_dir = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround'
    settings.result_plot_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/test/result_plots'
    settings.results_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround'
    settings.segmentation_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/test/segmentation_results'
    settings.tc128_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/data/trackingnet'
    settings.uav_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/data/UAV123'
    settings.vot_path = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

