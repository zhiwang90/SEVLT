from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.mgit_path = '/data3/hly/mgit'
    settings.got10k_lmdb_path = '/home/zyh/datasets/got10k_lmdb'
    settings.got10k_path = '/home/zyh/datasets/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/zyh/datasets/itb'
    settings.lasot_extension_subset_path_path = '/home/zyh/datasets/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/zyh/datasets/lasot_lmdb'
    settings.lasot_path = '/home/zyh/datasets/lasot'
    settings.network_path = '/home/zyh/DAVLT/output/test/networks'  # Where tracking networks are stored.
    settings.nfs_path = '/home/zyh/datasets/nfs'
    settings.otb99_path = '/home/zyh/datasets/OTB_sentences/'
    settings.prj_dir = '/home/zyh/DAVLT'
    settings.result_plot_path = '/home/zyh/DAVLT/output/test/result_plots'
    settings.results_path = '/home/zyh/DAVLT/output/test/tracking_results'  # Where to store tracking results
    settings.save_dir = '/home/zyh/DAVLT/output'
    settings.segmentation_path = '/home/zyh/DAVLT/output/test/segmentation_results'
    settings.tc128_path = '/home/zyh/datasets/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/zyh/datasets/tnl2k/TNL2K_test_subset'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/zyh/datasets/trackingnet'
    settings.uav_path = '/home/zyh/datasets/uav'
    settings.vot18_path = '/home/zyh/datasets/vot2018'
    settings.vot22_path = '/home/zyh/datasets/vot2022'
    settings.vot_path = '/home/zyh/datasets/VOT2019'
    settings.youtubevos_dir = ''

    return settings

