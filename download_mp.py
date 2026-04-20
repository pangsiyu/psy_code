#!/usr/bin/env python
# Downloads MP public data release (Optimized for Habitat scenes only)
# Run with python download_mp.py -o data/scene_datasets/mp3d
# -*- coding: utf-8 -*-
import argparse
import os
import tempfile
from urllib import request

BASE_URL = 'http://kaldir.vc.in.tum.de/matterport/'
RELEASE_TASKS = 'v1/tasks/'

# 我们只保留我们需要下载的 task_data 字典
TASK_FILES = {
    'habitat': ['mp3d_habitat.zip']
}

def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isfile(out_file):
        print('\tDownloading ' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        request.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)

def download_task_data(task_data, out_dir):
    print('Downloading MP task data for ' + str(task_data) + ' ...')
    for task_data_id in task_data:
        if task_data_id in TASK_FILES:
            file = TASK_FILES[task_data_id]
            for filepart in file:
                url = BASE_URL + RELEASE_TASKS + '/' + filepart
                localpath = os.path.join(out_dir, filepart)
                localdir = os.path.dirname(localpath)
                if not os.path.isdir(localdir):
                    os.makedirs(localdir)
                download_file(url, localpath)
                print('Downloaded task data ' + task_data_id)

def main():
    parser = argparse.ArgumentParser(description=
        '''
        Downloads MP public data release (Habitat Scenes Only).
        Example invocation:
          python download_mp.py -o data/scene_datasets/mp3d
        ''',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    args = parser.parse_args()

    # 强制只下载 habitat 场景包
    task_to_download = ['habitat']
    out_dir = os.path.join(args.out_dir, RELEASE_TASKS)
    
    print("==================================================")
    print("🚀 Starting download for Habitat Scene files ONLY...")
    print("This will download the 'mp3d_habitat.zip' containing .glb and .navmesh files.")
    print("==================================================")
    
    download_task_data(task_to_download, out_dir)
    
    print("==================================================")
    print("✅ Download complete! Please unzip the file:")
    print(f"cd {out_dir} && unzip mp3d_habitat.zip")
    print("Then ensure the extracted scene folders (e.g., zsNo4HB9uLZ/) are placed directly under data/scene_datasets/mp3d/")
    print("==================================================")

if __name__ == "__main__": 
    main()