class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one window as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class SingleViewGenerator(object):
    """Take only one random crops of one window as the query and key."""
    def __init__(self, toTensor, base_transform):
        self.toTensor = toTensor
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.toTensor(x), self.base_transform(x)]