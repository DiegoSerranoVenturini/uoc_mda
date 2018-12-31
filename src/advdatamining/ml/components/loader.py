from torch.utils.data.dataloader import DataLoader, _DataLoaderIter
from torch.autograd import Variable


class TrainingLoaderIter(_DataLoaderIter):

    pass


class TrainingLoader(DataLoader):

    def __iter__(self):

        iterable = _DataLoaderIter(self)
        (features, target) = iterable.next()

        return (Variable(features.float()), Variable(target))

