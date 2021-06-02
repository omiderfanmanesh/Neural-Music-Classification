# encoding: utf-8

from torch.utils import data
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.datasets import DatasetFolder
from torchvision.datasets.vision import VisionDataset
import numpy as np
import torch
import numpy as np

from .dataset.GTZAN import GTZANDataset
from .transforms import build_transforms
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, 'C:\\Users\\Giovanni Calleris\\Desktop\\Phyton\\PyCharm\\NeuralMusicClassification')
sys.path.append('.')
from NeuralMusicClassification.utils.spec_augment import freq_mask

from NeuralMusicClassification.config import cfg

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import pandas as pd
import os




def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    ### !!! MY MODIFICATOIN !!!
    # for target_class in sorted(class_to_idx.keys()):
    class_index = class_to_idx[list(class_to_idx.keys())[0]] #class_to_idx[target_class]
    """target_dir = os.path.join(directory, class_to_idx[target_class])
    if not os.path.isdir(target_dir):
        continue"""
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                item = path, class_index
                instances.append(item)
    return instances


class MyFolder_from_DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(MyFolder_from_DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [dir.split("\\")[-1]]  ### !!! MY MODIFICATOIN !!! -> [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

def npy_loader(path):

    """"

    Loads the numpy array to the training set

    It randomly pick a frequency range
    It shifts the frequency down randomly
    It inverts randomly the first and the second half of the clip along the time axis

    """
    counter_of_slices = 5
    slice1 = np.random.randint(2, counter_of_slices)
    pitch_shift = np.random.randint(-6,0)
    # song_time = np.random.randint(0, 2)
    song_time = 0
    song_lenght_size = cfg.DATALOADER.SONG_LENGTH
    song_freq_size = 128
    array_to_import = np.load(path)
    step = 12
    x = array_to_import[ (slice1 * step - pitch_shift):(slice1 * step + cfg.MODEL.FREQ_RANGE - pitch_shift), :]
    array_of_zeros = np.ones((step, x.shape[1]))*(-32)
    counter = 2
    counter_fix = 2
    list_x = []
    while counter < counter_of_slices:
        if counter == slice1:
            list_x.append(x)
        else:
            list_x.append(array_of_zeros)
        counter += 1

    array_to_import = np.concatenate(list_x, axis=0)
    # array_to_import = np.reshape(array_to_import, (cfg.MODEL.FREQ_RANGE,song_lenght_size))
    if array_to_import.shape == (cfg.MODEL.FREQ_RANGE+(counter_of_slices-counter_fix-1)*step, song_lenght_size):

        if song_time == 1:
            # array_to_import = np.flip(array_to_import, axis=-1).copy()
            a = array_to_import[ :, :47]
            b = array_to_import[ :, 47:]
            array_to_import = np.stack([b, a], axis=-1)
            array_to_import = np.reshape(array_to_import, (cfg.MODEL.FREQ_RANGE, song_lenght_size))
        # array_to_import = np.stack([np.real(array_to_import), np.imag(array_to_import)], axis=1)
        # #
        # resize_shape = list(array_to_import.shape)[0] * list(array_to_import.shape)[1]
        # array_to_import = np.reshape(array_to_import,
        #                        [resize_shape, list(array_to_import.shape)[2], list(array_to_import.shape)[3]])
        # array_to_import1 = (np.round(array_to_import[:,0:92]-array_to_import[:,1:93])==0) + 1
        # array_to_import2 = (np.round(array_to_import[:,1:93]-array_to_import[:,2:song_lenght_size])==0) + 1
        # array_to_import = (np.round(array_to_import1 - array_to_import2)==0) + 1
        # array_to_import = np.round(array_to_import[:,0:92]-array_to_import[:,1:93])==0
        # array_to_import_delta = array_to_import[:, 0:92] - array_to_import[:, 1:93]
        # array_to_import = np.stack([array_to_import[:,0:92], array_to_import_delta], axis=0)

        array_to_import = np.reshape(array_to_import, (1, cfg.MODEL.FREQ_RANGE+(counter_of_slices-counter_fix-1)*step, song_lenght_size))

        tensor = torch.from_numpy(array_to_import).type(torch.FloatTensor)
        # if np.random.randn() > 0:
        #     tensor = freq_mask(tensor, F=5, num_masks=1, replace_with_zero=True)
        return [tensor, slice1-counter_fix, song_time]
    else:
        return [torch.zeros([1, cfg.MODEL.FREQ_RANGE+(counter_of_slices-counter_fix-1)*step, song_lenght_size]), 0, song_time]
        # return torch.zeros([4, 32, song_lenght_size])

def npy_loader_test(path):

    """"

    Loads the numpy array to the training set

    """

    song_time = 0
    array_to_import = np.load(path)[:,:]
    song_lenght_size = cfg.DATALOADER.SONG_LENGTH
    song_freq_size = 47

    # array_to_import = np.stack([np.real(array_to_import), np.imag(array_to_import)], axis=1)
    #
    # resize_shape = list(array_to_import.shape)[0] * list(array_to_import.shape)[1]
    # array_to_import = np.reshape(array_to_import,
    #                              [resize_shape, list(array_to_import.shape)[2], list(array_to_import.shape)[3]])


    if array_to_import.shape == (song_freq_size, song_lenght_size):
        # array_to_import1 = (np.round(array_to_import[:, 0:92] - array_to_import[:, 1:93]) == 0) + 1
        # array_to_import2 = (np.round(array_to_import[:, 1:93] - array_to_import[:, 2:song_lenght_size]) == 0) + 1
        # array_to_import = (np.round(array_to_import1 - array_to_import2) == 0) + 1
        # array_to_import_delta = array_to_import[:, 0:92] - array_to_import[:, 1:93]
        #
        # array_to_import = np.stack([array_to_import[:,0:92], array_to_import_delta], axis=0)

        array_to_import = np.reshape(array_to_import, (1, song_freq_size, song_lenght_size))

        tensor = torch.from_numpy(array_to_import).type(torch.FloatTensor)
        return [tensor, 0, song_time]
    else:
        return [torch.zeros([1, song_freq_size, song_lenght_size]), 0, song_time]
        # return torch.zeros([4, song_freq_size, song_lenght_size])

def npy_loader_based_model(path):

    """"

    Loads the numpy array to the training set

    """

    array_to_import = np.load(path)
    song_lenght_size = cfg.DATALOADER.SONG_LENGTH
    song_freq_size = 128
    # std = np.std(array_to_import, axis=1)
    # mean = np.mean(array_to_import, axis=1)
    # array_to_import = np.stack([mean, std], axis=1)
    # array_to
    if array_to_import.shape == (song_freq_size, song_lenght_size):
        array_to_import = np.reshape(array_to_import, (1, song_freq_size, song_lenght_size))

        tensor = torch.from_numpy(array_to_import).type(torch.FloatTensor)

        return tensor
    else:
        return torch.zeros([1, song_freq_size, song_lenght_size])

def npy_loader_final_plus_features(path):

    """"

    Loads the numpy array to the training set

    """
    global scaler_df
    dict_to_label = {"blues": 0, "classical": 1, "country": 2, "disco": 3,
                     "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7,
                     "reggae": 8, "rock": 9}
    file = path.split("\\")[-1][:-4]
    file = file.split("_")[0] + "." + file.split("_")[1] + "." + str(int(np.round(int(file.split("_")[2][3:])/3))) + ".wav"
    features = df.loc[df["filename"] == file]
    if not features.empty:
        label = torch.from_numpy(features["label"].apply(lambda name: dict_to_label[name]).to_numpy().astype("int32"))
        features = features.drop(columns=["filename", "label", "length"])
        features = pd.DataFrame(scaler_df.transform(features), columns=features.columns)
        features = torch.from_numpy(features.to_numpy().astype("float32")[0])

    else:
        features = torch.zeros([57])
        label =  torch.zeros([1])

    array_to_import = np.load(path)
    song_lenght_size = cfg.DATALOADER.SONG_LENGTH
    song_freq_size = 128
    # std = np.std(array_to_import, axis=1)
    # mean = np.mean(array_to_import, axis=1)
    # array_to_import = np.stack([mean, std], axis=1)
    # array_to
    if array_to_import.shape == (song_freq_size, song_lenght_size):
        array_to_import = np.reshape(array_to_import, (1, song_freq_size, song_lenght_size))

        tensor = torch.from_numpy(array_to_import).type(torch.FloatTensor)

        return tensor.cuda(), features.cuda(), label.cuda()
    else:
        return torch.zeros([1, song_freq_size, song_lenght_size]).cuda(), torch.zeros([57]).cuda(), torch.zeros([1]).cuda()

def npy_loader_features_only(path):

    """"

    Loads the numpy array to the training set

    """
    global scaler_df
    # dict_to_label = {"blues": 0, "classical": 1, "country": 2, "disco": 3,
    #                  "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7,
    #                  "reggae": 8, "rock": 9}
    file = path.split("\\")[-1][:-4]
    file = file.split("_")[0] + "." + file.split("_")[1] + "." + str(int(np.round(int(file.split("_")[2][3:])/3))) + ".wav"
    features = df.loc[df["filename"] == file]
    if not features.empty:
        # label = torch.from_numpy(features["label"].apply(lambda name: dict_to_label[name]).to_numpy().astype("int32"))
        features = features.drop(columns=["filename", "label", "length"])
        features = pd.DataFrame(scaler_df.transform(features), columns=features.columns)
        features = torch.from_numpy(features.to_numpy().astype("float32")[0])
        return features.cuda()
    else:
        return torch.zeros([57]).cuda()

def npy_loader_features_only_for_scaling(path):

    """"

    Loads the numpy array to the training set

    """
    dict_to_label = {"blues": 0, "classical": 1, "country": 2, "disco": 3,
                     "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7,
                     "reggae": 8, "rock": 9}
    file = path.split("\\")[-1][:-4]
    file = file.split("_")[0] + "." + file.split("_")[1] + "." + str(int(np.round(int(file.split("_")[2][3:])/3))) + ".wav"
    mask = df["filename"] == file
    features = df.loc[mask]
    if not features.empty:
        label = torch.from_numpy(features["label"].apply(lambda name: dict_to_label[name]).to_numpy().astype("int32"))
        features = features.drop(columns=["filename", "label", "length"])
        features = features.to_numpy(dtype=np.float64)[0]
        return features, label
    else:
        print(file)
        return substitute, torch.zeros([1])


def npy_loader_pretrained(path):

    """"

    Loads the numpy array to the training set

    """

    array_to_import = np.load(path, allow_pickle=True)
    song_lenght_size = cfg.DATALOADER.SONG_LENGTH
    song_freq_size = 128
    y = array_to_import[1]
    array_to_import = array_to_import[0]
    # std = np.std(array_to_import, axis=1)
    # mean = np.mean(array_to_import, axis=1)
    # array_to_import = np.stack([mean, std], axis=1)
    # array_to
    if array_to_import.shape == (song_freq_size, song_lenght_size):
        array_to_import = np.reshape(array_to_import, (1, song_freq_size, song_lenght_size))

        tensor = torch.from_numpy(array_to_import).type(torch.FloatTensor)
        return tensor, y
    else:
        tensor, y = torch.zeros([1, song_freq_size, song_lenght_size]), torch.randint(0,2,[188])
        return tensor, y



def build_dataset(cfg, transforms):
    train_dataset, test_dataset, validation_dataset  = GTZANDataset(cfg=cfg, transforms=transforms)
    return train_dataset, test_dataset, validation_dataset


if (cfg.MODEL.COUNTER_FOR_SCALING_ONLY_ONCE == 0) and (cfg.MODEL.USE_FEATURES_ONLY or cfg.MODEL.USE_FEATURES):
    print(cfg.MODEL.COUNTER_FOR_SCALING_ONLY_ONCE)
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
    my_absolute_dirpath = my_absolute_dirpath + "\\" + "features_3_sec.csv"
    df = pd.read_csv(my_absolute_dirpath)
    scaler_df = StandardScaler()
    num_workers = 0
    batch_size = cfg.TEST.IMS_PER_BATCH
    print("Preparing loaders for SCALING features only")
    loader_training = npy_loader_features_only_for_scaling
    loader_test = npy_loader_features_only_for_scaling

    train_data = datasets.DatasetFolder(cfg.DATALOADER.NPY_SAMPLES_TRAINING_DATASET_ADDRESS,
                                        extensions="npy",
                                        loader=loader_training)

    test_data = datasets.DatasetFolder(cfg.DATALOADER.NPY_SAMPLES_TEST_DATASET_ADDRESS,
                                       extensions="npy",
                                       loader=loader_test)

    validation_data = datasets.DatasetFolder(cfg.DATALOADER.NPY_SAMPLES_VALIDATION_DATASET_ADDRESS,
                                             extensions="npy",
                                             loader=loader_test)

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.SOLVER.IMS_PER_BATCH_VAL_AND_TEST, shuffle=False, drop_last=True,
                                                    num_workers=num_workers)  # , num_workers=4
    validation_data_loader = torch.utils.data.DataLoader(validation_data,
                                                         batch_size=cfg.SOLVER.IMS_PER_BATCH_VAL_AND_TEST,
                                                         shuffle=False, drop_last=False, num_workers=num_workers)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.SOLVER.IMS_PER_BATCH_VAL_AND_TEST,
                                                   shuffle=False, drop_last=False, num_workers=num_workers)

    df_to_concat = pd.DataFrame()
    df_labels_to_concat = pd.DataFrame()
    if cfg.MODEL.CREATE_DATASET:
        name_loader = {0: "train",1: "test",2: "validation"}
        for name, loader in enumerate([train_data_loader, test_data_loader, validation_data_loader]):
            df_to_concat = pd.DataFrame()
            df_labels_to_concat = pd.DataFrame()
            for x in loader:
                label = pd.DataFrame(x[1].numpy())
                x = pd.DataFrame(x[0][0].numpy())
                substitute = x.iloc[0,:]
                df_to_concat = pd.concat([df_to_concat, x], axis=0)
                df_labels_to_concat = pd.concat([df_labels_to_concat, label], axis=0)
            if name_loader[name] == "train":
                scaler_df = scaler_df.fit(df_to_concat)
            if name_loader[name] == "test":
                print("HELLO")
            df_to_concat = pd.DataFrame(scaler_df.transform(df_to_concat))
            df_labels_to_concat.to_csv(f"features_label_3_sec_{name_loader[name]}.csv")
            df_to_concat.to_csv(f"features__3_sec_{name_loader[name]}.csv")
    else:
        df_to_concat = pd.DataFrame()
        df_labels_to_concat = pd.DataFrame()
        for x in train_data_loader:
            label = pd.DataFrame(x[1].numpy())
            x = pd.DataFrame(x[0][0].numpy())
            substitute = x.iloc[0,:]
            df_to_concat = pd.concat([df_to_concat, x], axis=0)
            df_labels_to_concat = pd.concat([df_labels_to_concat, label], axis=0)
        scaler_df.fit(df_to_concat)
    cfg.MODEL.COUNTER_FOR_SCALING_ONLY_ONCE += 1



def make_data_loader(cfg, test_size=0.10, validation_size=0.10, shuffle=True):
    batch_size = cfg.TEST.IMS_PER_BATCH
    batch_size2 = cfg.TEST.IMS_PER_BATCH
    num_workers = 0

    if cfg.MODEL.USE_FEATURES_ONLY or cfg.MODEL.USE_FEATURES:

        num_workers = 0

    if cfg.MODEL.USE_FEATURES:
        print("Preparing loaders for final + features")
        loader_training = npy_loader_final_plus_features
        loader_test = npy_loader_final_plus_features

    elif cfg.MODEL.USE_FEATURES_ONLY:
        print("Preparing loaders for features only")
        loader_training = npy_loader_features_only
        loader_test = npy_loader_features_only

    elif cfg.MODEL.BASED_MODEL:
        print("Preparing loaders for paper's based model")
        loader_training = npy_loader_based_model
        loader_test = npy_loader_based_model

    else:
        print("Preparing loaders for model extension")
        loader_training = npy_loader
        loader_test = npy_loader_test

    if cfg.MODEL.PRETRAIN:
        print("Loading MAGNATAGATUNE")
        if cfg.MODEL.LOAD_MODEL:
            num_workers = 0
        else:
            num_workers = cfg.DATALOADER.NUM_WORKERS

        train_data = MyFolder_from_DatasetFolder(cfg.DATALOADER.NPY_SAMPLES_TRAINING_DATASET_ADDRESS_PRETRAIN ,
                               extensions="npy",
                               loader=npy_loader_pretrained)


        test_data = MyFolder_from_DatasetFolder(cfg.DATALOADER.NPY_SAMPLES_TEST_DATASET_ADDRESS_PRETRAIN ,
                                           extensions="npy",
                                           loader=npy_loader_pretrained)

        validation_data = MyFolder_from_DatasetFolder(cfg.DATALOADER.NPY_SAMPLES_VALIDATION_DATASET_ADDRESS_PRETRAIN ,
                                           extensions="npy",
                                           loader=npy_loader_pretrained)

        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size2, shuffle=True,
                                                        drop_last=True, num_workers=num_workers)  # , num_workers=4
        validation_data_loader = torch.utils.data.DataLoader(validation_data,
                                                             batch_size=9,
                                                             shuffle=False, drop_last=True, num_workers=num_workers)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=9,
                                                       shuffle=False, drop_last=True, num_workers=num_workers)

        m = []
        s = []
        if cfg.MODEL.SCALE and cfg.MODEL.PRETRAIN :
            print("SCALING TO MAGNATAGATUNE")
            limiter = 0
            for x in train_data_loader:
                if limiter > 200:
                    break
                m.append(x[0][0].mean())
                s.append(x[0][0].std())
                limiter += 1

            cfg.MODEL.MEAN = float(np.mean(m))
            cfg.MODEL.STD = float(np.max(s))


    if not cfg.MODEL.PRETRAIN or (cfg.MODEL.PRETRAIN and cfg.MODEL.LOAD_MODEL):
        print("Loading GTZAN")

        train_data = datasets.DatasetFolder(cfg.DATALOADER.NPY_SAMPLES_TRAINING_DATASET_ADDRESS,
                                            extensions="npy",
                                            loader=loader_training)

        test_data = datasets.DatasetFolder(cfg.DATALOADER.NPY_SAMPLES_TEST_DATASET_ADDRESS,
                                           extensions="npy",
                                           loader=loader_test)

        validation_data = datasets.DatasetFolder(cfg.DATALOADER.NPY_SAMPLES_VALIDATION_DATASET_ADDRESS,
                                                 extensions="npy",
                                                 loader=loader_test)

        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)  # , num_workers=4
        validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=cfg.SOLVER.IMS_PER_BATCH_VAL_AND_TEST, shuffle=False, drop_last=False, num_workers=num_workers)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.SOLVER.IMS_PER_BATCH_VAL_AND_TEST, shuffle=False, drop_last=False, num_workers=num_workers)

    return train_data_loader, test_data_loader, validation_data_loader
