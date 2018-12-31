import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from advdatamining.utils import reader, plots
from advdatamining.ml.nets.cnn import CNN, WeightedCNN
from advdatamining.ml.nets.boosting import BoostingNet


def run_mlapp():
    images = reader.ImageReader().read_images()

    # plots.plot_random_imgs(images["images"], images["labels"])

    # TODO: reimplement with pyspark
    X_train, X_test, y_train, y_test = train_test_split(images["images"], images["labels"], test_size=0.2, random_state=2017, stratify=images["labels"])

    pca = PCA(n_components=100, random_state=2017)
    pca_fit = pca.fit(X_train)
    X_train_pca = pca_fit.transform(X_train)
    X_test_pca = pca_fit.transform(X_test)
    images_train = {"images": X_train_pca, "labels": y_train}

    # images_train = {"images": X_train, "labels": y_train}
    # cnn = WeightedCNN(n_hidden=32, n_output=5, inputCols="images", outputCol="labels").\
    # build_net(optimizer=torch.optim.Adam, loss_function=CrossEntropyLoss)
    # cnn.fit(images_train, batch_size=64, num_epochs=20, init_lr=0.01)
    # predictions = cnn.predict(X_test_pca)

    net = BoostingNet(n_input=X_train_pca.shape[1], hidden_layers=[64, 32], n_output=5, boosting_stack=1,
                      inputCols="images", outputCol="labels").\
        build_net(optimizer=torch.optim.Adam, loss_function=nn.CrossEntropyLoss)

    net.fit(images_train, batch_size=64, num_epochs=20, init_lr=0.01)
    predictions = net.predict(X_test_pca)

    # calculate balanced accuracy and confusion matrix for following questions:
    balanced_accuracy = balanced_accuracy_score(y_pred=predictions, y_true=y_test)
    conf_matrix = confusion_matrix(y_pred=predictions, y_true=y_test)

    print("La precision balanceada del modelo es: %s" % balanced_accuracy)
    plots.plot_confusion_matrix(conf_matrix, set(images["labels"].tolist()))

    print(net)


if __name__ == '__main__':
    run_mlapp()
