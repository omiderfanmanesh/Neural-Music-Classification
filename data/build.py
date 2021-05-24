# encoding: utf-8

from torch.utils import data

from .dataset.GTZAN import GTZANDataset


def build_dataset(cfg, transforms):
    train_dataset = GTZANDataset(cfg=cfg, training_type=0, transforms=transforms)
    validation_dataset = GTZANDataset(cfg=cfg, training_type=1, transforms=transforms)
    test_dataset = GTZANDataset(cfg=cfg, training_type=2, transforms=transforms)
    return train_dataset, validation_dataset, test_dataset


def make_data_loader(cfg, test_size=0.10, validation_size=0.10, shuffle=True):
    # transforms = build_transforms(cfg)
    train_dataset, validation_dataset, test_dataset = build_dataset(cfg=cfg, transforms=None)

    # dataset_size = len(datasets)
    # test_size = int(test_size * dataset_size)
    # train_size = dataset_size - test_size
    #
    # train_dataset, test_dataset = random_split(datasets,
    #                                            [train_size, test_size])
    #
    # validation_size = int(validation_size * train_size)
    # train_size = train_size - validation_size
    #
    # train_dataset, validation_dataset = random_split(train_dataset,
    #                                                  [train_size, validation_size])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.TEST.IMS_PER_BATCH

    train_data_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    test_data_loader = data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=num_workers
    )

    validation_data_loader = data.DataLoader(
        validation_dataset, batch_size=8, shuffle=False, num_workers=num_workers
    )

    return train_data_loader, test_data_loader, validation_data_loader
