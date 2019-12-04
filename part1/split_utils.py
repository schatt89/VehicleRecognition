import os
import math
import random
import tempfile

def _copy_symlinks(files, src_dir, dst_dir):
    for i in files:
        base_file_name = os.path.basename(i)
        src_file_path = os.path.join(src_dir, base_file_name)
        dst_file_path = os.path.join(dst_dir, base_file_name)
        src_file_path = os.path.abspath(src_file_path)
        dst_file_path = os.path.abspath(dst_file_path)
        os.symlink(src_file_path, dst_file_path)

def train_valid_split(original_dir, validation_split=0.1, seed=None):
    if seed is not None:
        random.seed(seed)    
    if not os.path.isdir(original_dir):
        raise NotADirectoryError
    tmp_dir = tempfile.TemporaryDirectory()
    train_dir = os.path.join(tmp_dir.name, 'train')
    valid_dir = os.path.join(tmp_dir.name, 'validation')

    # make subdirs in train tmp and valid tmp
    for root, dirs, files in os.walk(original_dir):
        if root == original_dir:
            continue
        sub_dir_name = os.path.basename(root)
        train_sub_dir_path = os.path.join(train_dir, sub_dir_name)
        valid_sub_dir_path = os.path.join(valid_dir, sub_dir_name)
        if not os.path.exists(train_sub_dir_path):
            os.makedirs(train_sub_dir_path)
        if not os.path.exists(valid_sub_dir_path):
            os.makedirs(valid_sub_dir_path)

    # distribute symlinks to train_tmp, test_tmp
    for root, dirs, files in os.walk(original_dir):
        if root == original_dir:
            continue
        sub_dir_name = os.path.basename(root)
        train_sub_dir_path = os.path.join(train_dir, sub_dir_name)
        valid_sub_dir_path = os.path.join(valid_dir, sub_dir_name)
        files = [os.path.join(root, f) for f in files]
        random.shuffle(files)
        valid_idx = math.ceil(validation_split * len(files))
        train_files = files[valid_idx:]
        valid_files = files[:valid_idx]
        _copy_symlinks(train_files, root, train_sub_dir_path)
        _copy_symlinks(valid_files, root, valid_sub_dir_path)
    return tmp_dir, train_dir, valid_dir