from datasets.preprocessing import global_contrast_normalization, zca_whitening

def gcn_zca(train_set, valid_set, test_set, dataset):
    # Apply gcn
    train_set = global_contrast_normalization(train_set)
    valid_set = global_contrast_normalization(valid_set)
    test_set = global_contrast_normalization(test_set)

    # Apply zca
    train_set, mean, W = zca_whitening(train_set)
    valid_set, _, _ = zca_whitening(valid_set, mean=mean, whitening=W)
    test_set, _, _ = zca_whitening(test_set, mean=mean, whitening=W)

    return train_set, valid_set, test_set
