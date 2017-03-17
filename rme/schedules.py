from callbacks import Step

def onetenth_200_230(dataset, lr):
    steps = [200, 230]
    lrs = [lr, lr/10, lr/100]
    return Step(steps, lrs)

def dsn_step_200_230(dataset, lr):
    steps = [200, 230]
    lrs = [lr, lr/2.5, lr/25]
    return Step(steps, lrs)

def nin_nobn_mnist(dataset, lr):
    steps = [40, 50]
    lrs = [lr, lr/2, lr/10]
    return Step(steps, lrs)

def dsn_step_20_30(dataset, lr):
    steps = [20, 30]
    lrs = [lr, lr/2.5, lr/25]
    return Step(steps, lrs)


def dsn_step_40_60(dataset, lr):
    steps = [40, 60]
    lrs = [lr, lr/2.5, lr/25]
    return Step(steps, lrs)

def wideresnet_step(dataset, lr):
    steps = [60, 120, 160]
    lrs = [lr, lr/5, lr/25, lr/125]
    return Step(steps, lrs)
