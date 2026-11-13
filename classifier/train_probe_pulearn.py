import torch
import torch.nn as nn

from sklearn.neighbors import KNeighborsClassifier
from .dataset_probe import probe_dataset_loader
import bacpipe

import logging

logger = logging.getLogger("bacpipe")


class LinearProbe(nn.Module):
    """
    Linear probing classifier head mapping embeddings to class logits.
    """

    def __init__(self, in_dim, out_dim, device="cpu", **kwargs):
        """
        Linear classification layer.

        Parameters
        ----------
        in_dim : int
            number of input dimensions (dictated by embeddings)
        out_dim : int
            number of output dimensions (dictated by classes in ground truth)
        device : str, optional
            device the probe is moved to, by default "cpu"
        """
        super(LinearProbe, self).__init__()
        self.probe = nn.Linear(in_dim, out_dim)
        self.probe.to(device)

    def forward(self, x):
        """
        Run the linear probe on the input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            input embeddings

        Returns
        -------
        torch.Tensor
            logits for each class
        """
        return self.probe(x)


def train_linear_probe(
    linear_classifier,
    train_dataloader,
    learning_rate,
    num_epochs,
    device="cpu",
    **kwargs,
):
    """
    Linear classification training pipeline. Hyperparameters are specified
    in settings.yaml file and passed to this function.

    Parameters
    ----------
    linear_classifier : object
        classification object
    train_dataloader : DataLoader object
        dataset loader to iterate over
    learning_rate : float
        learning rate
    num_epochs : int
        number of epochs for training
    device : str, optional
        'cpu' or 'cuda', by default "cpu"

    Returns
    -------
    object
        trained linear classification object
    """
    device = torch.device(device)
    try:
        linear_classifier = linear_classifier.to(device)
    except RuntimeError:
        logger.error("Traceback", exc_info=True)
        logger.info(
            "This problem is likely caused by tensorflow hogging all the gpu vram. "
            "The best fix for this is to simply restart bacpipe with the same settings, "
            "that way the GPU should be available for pytorch. Alternatively select "
            "`cpu` for device in the settings.yaml file."
        )
        import sys

        sys.exit(0)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(
        linear_classifier.parameters(), lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        linear_classifier.train()
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for embeddings, y in train_dataloader:
            embeddings, y = embeddings.to(device), y.to(device)

            # Forward pass through linear classifier
            outputs = linear_classifier(embeddings)

            # Compute loss
            loss = criterion(outputs, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training loss and accuracy
            running_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += embeddings.size(0)
            correct_train += (predicted == y).sum().item()

        train_loss = running_loss / len(train_dataloader.dataset)
        train_accuracy = 100 * correct_train / total_train

        # logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}")

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%"
        )

    return linear_classifier


class KNNProbe(nn.Module):
    """
    K-nearest-neighbor probing classifier operating on raw embeddings.
    """

    def __init__(self, n_neighbors=15, testing=False, **kwargs):
        """
        K-nearest neighbor classifier.

        Parameters
        ----------
        n_neighbors : int, optional
            hyperparameter specified in settings.yaml file, by default 15
        testing : bool, optional
            whether the classifier is used for testing purposes,
            by default False
        """
        super(KNNProbe, self).__init__()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.is_trained = False  # Flag to track if KNN is trained

    def fit(self, x, y):
        """
        Train KNN classifier with numpy data

        Parameters
        ----------
        x : torch.Tensor
            embedding tensors
        y : torch.Tensor
            label tensors
        """
        x_np = x.cpu().detach().numpy()  # Convert tensor to NumPy
        y_np = y.cpu().detach().numpy()
        self.knn.fit(x_np, y_np)
        self.is_trained = True

    def forward(self, x):
        """
        Predict using KNN (only after it's trained)

        Parameters
        ----------
        x : torch.Tensor
            embedding tensors

        Returns
        -------
        torch.Tensor
            predicted labels
        torch.Tensor
            prediction probabilities
        """
        if not self.is_trained:
            error = "\nKNN model is not trained. Call `fit()` first."
            logger.exception(error)
            raise ValueError(error)

        x_np = x.cpu().detach().numpy()
        preds = self.knn.predict(x_np)  # Predict labels
        probs = self.knn.predict_proba(x_np)  # Predict probabilities

        preds_tensor = torch.tensor(preds, dtype=torch.long, device=x.device)
        probs_tensor = torch.tensor(
            probs, dtype=torch.float32, device=x.device
        )

        return preds_tensor, probs_tensor


def train_knn_probe(knn_classifier, train_dataloader, device="cpu", **kwargs):
    """
    Pipeline for knn classifier training.

    Parameters
    ----------
    knn_classifier : object
        classifier object
    train_dataloader : DataLoader object
        iterator for dataset
    device : str, optional
        'cpu' or 'cuda', by default "cpu"

    Returns
    -------
    object
        classifier object
    """
    device = torch.device(device)
    knn_classifier.to(device)

    all_embeddings = []
    all_labels = []

    # Collect all embeddings and labels to train KNN
    for embeddings, y in train_dataloader:
        embeddings, y = embeddings.to(device), y.to(device)
        all_embeddings.append(embeddings)
        all_labels.append(y)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Train KNN
    knn_classifier.fit(all_embeddings, all_labels)
    logger.info("KNN Training Complete!")

    return knn_classifier



class PulearnProbe(nn.Module):
    """One-vs-rest positive-unlabeled probe using pulearn's Elkan-Noto estimator.

    Rows matching a class are labeled positives; all other rows are treated as
    unlabeled. Use only when non-positive rows should be unlabeled, not confirmed
    negatives.
    """

    def __init__(self, classes=None, C=0.01, hold_out_ratio=0.2,
                 max_iter=2000, random_state=42, **kwargs):
        super().__init__()
        self.classes = None if classes is None else list(classes)
        self.C = C
        self.hold_out_ratio = hold_out_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self.models = {}
        self.is_trained = False

    def fit(self, x, y):
        try:
            from pulearn import ElkanotoPuClassifier
        except ImportError as exc:
            raise ImportError(
                "The pulearn probe requires `pip install pulearn` in bacpipe's environment."
            ) from exc
        from sklearn.linear_model import LogisticRegression

        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        x_np = x_np.astype("float32", copy=False)
        y_np = y_np.reshape(-1)
        if self.classes is None:
            self.classes = sorted(set(y_np.tolist()))

        self.models = {}
        for class_index in self.classes:
            y_pu = (y_np == class_index).astype("int64")
            n_positive = int(y_pu.sum())
            n_unlabeled = int(len(y_pu) - n_positive)
            if n_positive < 2 or n_unlabeled < 1:
                logger.warning(
                    "Skipping PU class %r: found %d positives and %d unlabeled rows.",
                    class_index, n_positive, n_unlabeled,
                )
                continue
            model = ElkanotoPuClassifier(
                estimator=LogisticRegression(C=self.C, max_iter=self.max_iter),
                hold_out_ratio=self.hold_out_ratio,
                random_state=self.random_state,
            )
            model.fit(x_np, y_pu)
            self.models[class_index] = model

        if not self.models:
            raise ValueError(
                "No PU models could be fitted. Each class needs at least two "
                "labeled positives and at least one unlabeled example."
            )
        self.is_trained = True
        return self

    def forward(self, x):
        if not self.is_trained:
            raise ValueError("PU model is not trained. Call fit() first.")
        import numpy as np

        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        scores = np.zeros((len(x_np), len(self.classes)), dtype=np.float32)
        for col, class_index in enumerate(self.classes):
            model = self.models.get(class_index)
            if model is not None:
                scores[:, col] = model.predict_proba(x_np)[:, 1]
        preds = np.asarray(self.classes)[scores.argmax(axis=1)]
        device = x.device if isinstance(x, torch.Tensor) else "cpu"
        return (
            torch.as_tensor(preds, dtype=torch.long, device=device),
            torch.as_tensor(scores, dtype=torch.float32, device=device),
        )


def train_pulearn_probe(pulearn_classifier, train_dataloader, device="cpu", **kwargs):
    """Collect embedding batches and fit a pulearn PU probe."""
    all_embeddings, all_labels = [], []
    for embeddings, y in train_dataloader:
        all_embeddings.append(embeddings.detach().cpu())
        all_labels.append(y.detach().cpu())
    if not all_embeddings:
        raise ValueError("The PU training dataloader produced no batches.")
    pulearn_classifier.fit(torch.cat(all_embeddings, dim=0),
                           torch.cat(all_labels, dim=0))
    logger.info("PU learning training complete (%d one-vs-rest models).",
                len(pulearn_classifier.models))
    return pulearn_classifier

def train_probe(
    embeds,
    df,
    label2index,
    config="linear",
    learning_rate=None,
    num_epochs=None,
    n_neighbors=None,
    pu_C=0.01,
    pu_hold_out_ratio=0.2,
    pu_max_iter=2000,
    pu_random_state=42,
    **kwargs,
):
    """
    Classification pipeline. First the classification dataframe is loaded,
    then a dict is created to link labels to ints, then the dataset loaders
    are created to iterate over. Next depending of the specified config
    a linear or KNN classification is performed. Finally the classifiers are
    used for inference and based on that performance metrics are created.

    Parameters
    ----------
    embeds : np.array
        the embeddings
    df : pandas.DataFrame
        classification dataframe
    label2index : dict
        link labels to ints
    config : str, optional
        type of classification, by default 'linear'
    learning_rate : float, optional
        learning rate for the linear probe, by default None
    num_epochs : int, optional
        number of training epochs, by default None
    n_neighbors : int, optional
        number of neighbors for the KNN classifier, by default None

    Returns
    -------
    object
        trained classifier object
    """

    # generate the loaders
    train_gen = probe_dataset_loader(
        "train", df, embeds, label2index, **kwargs
    )

    embed_size = embeds[0].shape[-1]

    if config == "linear":
        if learning_rate is None:
            learning_rate = bacpipe.settings.probe_configs["config_1"][
                "learning_rate"
            ]
        if num_epochs is None:
            num_epochs = bacpipe.settings.probe_configs["config_1"][
                "num_epochs"
            ]
        probe = LinearProbe(
            in_dim=embed_size, out_dim=len(df.label.unique()), **kwargs
        )
        probe = train_linear_probe(
            probe,
            train_gen,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            **kwargs,
        )

    elif config in ("pulearn", "pu"):
        probe = PulearnProbe(
            classes=sorted(label2index.values()),
            C=pu_C,
            hold_out_ratio=pu_hold_out_ratio,
            max_iter=pu_max_iter,
            random_state=pu_random_state,
        )
        probe = train_pulearn_probe(probe, train_gen, **kwargs)

    elif config == "knn":
        if n_neighbors is None:
            n_neighbors = bacpipe.settings.probe_configs["config_2"][
                "n_neighbors"
            ]
        if len(df[df.predefined_set == "test"]) < n_neighbors:
            kwargs["n_neighbors"] = len(df[df.predefined_set == "test"]) - 1
        probe = KNNProbe(**kwargs)
        probe = train_knn_probe(probe, train_gen, **kwargs)

    else:
        raise ValueError(
            f"Unknown probe config {config!r}. Expected 'linear', 'knn', or 'pulearn'."
        )

    return probe
