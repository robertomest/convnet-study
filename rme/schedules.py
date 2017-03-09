from callbacks import Step

def onetenth_200_230(dataset, lr):
    steps = [200, 230]
    lrs = [lr, lr/10, lr/100]
    return Step(steps, lrs)
