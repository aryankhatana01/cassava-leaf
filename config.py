class CFG:
    nFolds = 5
    seed = 2022
    model_arch = 'tf_efficientnet_b3_ap'
    img_size = 512
    epochs = 12
    trainbs = 16
    validbs = 32
    T0 = 10
    learning_rate = 1e-4
    min_lr = 1e-6
    decay = 1e-6
    num_workers = 4
    accum_iter = 2
    verbose = 1
    device = 'cuda:0'