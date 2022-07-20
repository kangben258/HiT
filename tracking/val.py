import os
import yaml
import argparse
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate trackers on got10k-val.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb',
                        help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)

    args = parser.parse_args()

    if args.dataset_name in ['trackingnet', 'lasot']:
        save_dataset_name = args.dataset_name
    elif 'got10k' in args.dataset_name:
        save_dataset_name = 'got10k'
    else:
        save_dataset_name = ''

    root_path = os.path.join(os.path.dirname(__file__), "..")
    model_root = os.path.join(root_path, "checkpoints/train", args.tracker_name, args.tracker_param)
    config = os.path.join(root_path, "experiments", args.tracker_name, args.tracker_param+'.yaml')
    config_back = os.path.join(root_path, "experiments", args.tracker_name, args.tracker_param+'.back.yaml')
    os.system("cp %s %s" % (config, config_back))
    save_root = os.path.join(root_path, "test/tracking_results", args.tracker_name, args.tracker_param,
                             save_dataset_name)
    record_root = os.path.join(root_path, "test/tracking_results", args.tracker_name, args.tracker_param, "record.txt")
    # model_root='/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/vittrack_baseline/baseline'
    # config = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/experiments/vittrack_baseline/baseline.yaml'

    model_names = os.listdir(model_root)
    # print(model_names)
    indexes = [int(name.split('.')[0][-3:]) for name in model_names]
    # print(indexes)
    # f=open(config)
    dataset = get_dataset(args.dataset_name)
    for index in indexes:
        save_root_target = os.path.join(root_path, "test/tracking_results", args.tracker_name, args.tracker_param, save_dataset_name+str(index))
        with open(config) as f:
            list_config = yaml.safe_load(f)
            print(list_config['TEST']['EPOCH'])
            list_config['TEST']['EPOCH']=index
        with open(config, "w") as f:
            yaml.dump(list_config,f)

        ####run_tracker##############
        try:
            seq_name = int(args.sequence)
        except:
            seq_name = args.sequence

        if seq_name is not None:
            dataset = [dataset[seq_name]]

        trackers = [Tracker(args.tracker_name, args.tracker_param, args.dataset_name, args.runid)]

        if not os.path.exists(save_root_target):
            run_dataset(dataset, trackers, args.debug, args.threads, num_gpus=args.num_gpus)
        else:
            print("%s results exists" % index)
            os.system("mv %s %s" % (save_root_target, save_root))

        ##############eval###############
        scores = print_results(trackers, dataset, args.dataset_name,
                               merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
                               force_evaluation=True)
        ####write##################
        record = open(record_root, 'a')
        record.write("epoch:"+str(index))
        record.write(scores)
        record.close()
        #####################

        os.system("mv %s %s" % (save_root, save_root_target))

    os.system("cp %s %s" % (config_back, config))
    os.system("rm %s" % (config_back))

if __name__ == '__main__':
    main()

