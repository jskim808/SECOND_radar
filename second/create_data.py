import copy
from pathlib import Path
import pickle

import fire

import second.data.kitti_dataset as kitti_ds
import second.data.nuscenes_dataset as nu_ds
from second.data.all_dataset import create_groundtruth_database

def kitti_data_prep(root_path):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database("KittiDataset", root_path, Path(root_path) / "kitti_infos_train.pkl")

def nuscenes_data_prep(root_path, version, dataset_name, max_sweeps=10):
    root_path = '/mnt/sdd/jhyoo/dataset/NUSCENES'
    home_path = Path('/home/spalab/jskim_2/second.pytorch/second/dataset')
    # nu_ds.create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps)
    name = "infos_train.pkl"
    if version == "v1.0-test":
        name = "infos_test.pkl"
    database_save_path = home_path / 'gt_database'
    db_info_save_path = home_path / "kitti_dbinfos_train.pkl"
    create_groundtruth_database(dataset_class_name=dataset_name, data_path=root_path,info_path=home_path / name, database_save_path=database_save_path,db_info_save_path=db_info_save_path)

if __name__ == '__main__':
    fire.Fire()
