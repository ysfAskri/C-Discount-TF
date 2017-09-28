import io
from itertools import islice

import bson
import numpy as np
from skimage.data import imread  # or, whatever image library you prefer
from sklearn.utils import shuffle

step = 1383
NB_IT = 5532


def getListCategory(category_tab, category_id):
    try:
        index = category_tab.index(category_id)
    except ValueError:
        category_tab.append(category_id)
        index = category_tab.index(category_id)

    return category_tab, index


def load_train(train_path, next_step, NB_CLASSES=5270):
    data = bson.decode_file_iter(open(train_path, 'rb'))
    classes = []
    images = []
    labels = []
    img_names = []
    cls = []

    i = next_step
    nbImg = 0

    for c, d in enumerate(islice(data, next_step, next_step + step)):
        product_id = d['_id']
        category_id = d['category_id']  # This won't be in Test data
        classes, index = getListCategory(classes, category_id)
        i += 1

        for e, pic in enumerate(d['imgs']):
            nbImg += 1
            print('number of produit=', i, ' nombre images= ', nbImg, ' category length= ', len(classes))

            picture = imread(io.BytesIO(pic['picture']))
            images.append(picture.astype(np.float32))

            img_names.append(product_id)

            label = np.zeros(NB_CLASSES)
            label[index] = 1.0
            labels.append(label)

            cls.append(category_id)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    return images, labels, img_names, cls, next_step


class DataSet(object):
    def __init__(self, images, labels, img_names, cls):
        print("images.shape[0]", images.shape[0])
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, num_classes, next_step, validation_size):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, img_names, cls, next_step = load_train(train_path, next_step, NB_CLASSES=num_classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

    return data_sets, len(images), cls, img_names, images, next_step


def get_x_batch(test_path):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, img_names, cls = load_train(test_path)

    data_sets.data = DataSet(images, labels, img_names, cls)

    x_batch, _, img_names, _ = data_sets.data.next_batch(len(images))
    return x_batch, img_names, cls, len(images)
