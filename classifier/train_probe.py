import torch
import torch.nn as nn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from .dataset_probe import probe_dataset_loader
from .hierarchy import (
    DEFAULT_HIERARCHY_PATH,
    HierarchyProbe,
    HierarchyTree,
    focal_tree_min_logit_components,
    get_species_label_map,
)
import bacpipe

import logging
import math

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
        self.is_calibrated = False

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

        if x_np.ndim == 1:
                x_np = x_np.reshape(1, -1)
        elif x_np.ndim != 2:
            raise ValueError(
                f"Expected 1D or 2D embeddings, received shape {x_np.shape}"
            )


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

    def fit_calibrators(self, X_cal, y_cal, method="platt"):
        import numpy as np
        from pulearn.calibration import PUCalibrator

        if not self.is_trained:
            raise ValueError("Fit the PU probe before calibrating it.")

        X_cal = (
            X_cal.detach().cpu().numpy()
            if isinstance(X_cal, torch.Tensor)
            else np.asarray(X_cal)
        )
        y_cal = (
            y_cal.detach().cpu().numpy()
            if isinstance(y_cal, torch.Tensor)
            else np.asarray(y_cal)
        )

        y_cal = y_cal.reshape(-1)

        if len(X_cal) != len(y_cal):
            raise ValueError("X_cal and y_cal have different lengths.")

        for class_id, model in self.models.items():
            # Convert class IDs into binary PU labels for this species.
            y_binary = (y_cal == class_id).astype(np.int64)

            if np.unique(y_binary).size < 2:
                raise ValueError(
                    f"Class {class_id} needs both positive and "
                    "non-positive calibration examples."
                )

            model.fit_calibrator(
                PUCalibrator(method=method),
                X_cal,
                y_binary,
            )

        self.is_calibrated = True
        return self


    def predict_calibrated_proba(self, x):
        import numpy as np

        if not getattr(self, "is_calibrated", False):
            raise ValueError("Fit the calibrators before prediction.")

        is_tensor = isinstance(x, torch.Tensor)
        device = x.device if is_tensor else torch.device("cpu")

        x_np = (
            x.detach().cpu().numpy()
            if is_tensor
            else np.asarray(x)
        )

        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)

        scores = np.zeros(
            (len(x_np), len(self.classes)),
            dtype=np.float32,
        )

        for col, class_id in enumerate(self.classes):
            model = self.models.get(class_id)

            if model is not None:
                scores[:, col] = (
                    model.predict_calibrated_proba(x_np)[:, 1]
                )

        return torch.as_tensor(
            scores,
            dtype=torch.float32,
            device=device,
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


def manifold_mixup_ftm_components(
    logits,
    targets,
    paired_targets,
    mix_coefficients,
    hierarchy,
    gamma=2.0,
):
    """Return coefficient-weighted FTM terms for a batch of mixed examples."""
    if logits.shape != targets.shape or targets.shape != paired_targets.shape:
        raise ValueError(
            "Mixed FTM logits and both target batches must have identical shapes."
        )
    if mix_coefficients.shape != (logits.shape[0],):
        raise ValueError(
            "Mix coefficients must contain one value per batch example."
        )
    if not torch.isfinite(mix_coefficients).all() or (
        (mix_coefficients < 0.0) | (mix_coefficients > 1.0)
    ).any():
        raise ValueError("Mix coefficients must be finite values in [0, 1].")

    primary_components = focal_tree_min_logit_components(
        logits, targets, hierarchy, gamma=gamma
    )
    paired_components = focal_tree_min_logit_components(
        logits, paired_targets, hierarchy, gamma=gamma
    )
    weights = mix_coefficients.unsqueeze(1)
    return weights * primary_components + (1.0 - weights) * paired_components


def train_hierarchy_probe(
    classifier,
    train_dataloader,
    label2index,
    hierarchy,
    species_label_map,
    learning_rate,
    num_epochs,
    ftm_gamma=2.0,
    validation_dataloader=None,
    weight_decay=0.0,
    early_stopping_patience=5,
    validation_min_delta=0.0,
    manifold_mixup=False,
    manifold_mixup_alpha=0.2,
    manifold_mixup_probability=1.0,
    device="cpu",
):
    """Train a hierarchy head using only the Focal Tree-Min objective."""
    device = torch.device(device)
    classifier = classifier.to(device)
    tree = classifier.hierarchy()
    labels_by_index = sorted(label2index, key=label2index.__getitem__)
    if [label2index[label] for label in labels_by_index] != list(
        range(len(labels_by_index))
    ):
        raise ValueError("label2index values must be contiguous from zero.")

    target_lookup = torch.stack(
        [
            hierarchy.targets_for_species(species_label_map[label])
            for label in labels_by_index
        ]
    ).to(device)
    if len(train_dataloader.dataset) == 0:
        raise ValueError("The hierarchy training set is empty.")
    if (
        validation_dataloader is not None
        and len(validation_dataloader.dataset) == 0
    ):
        logger.warning(
            "Validation set is empty; skipping validation and early stopping."
        )
        validation_dataloader = None
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive or None.")
    if validation_min_delta < 0:
        raise ValueError("validation_min_delta must be non-negative.")
    if not math.isfinite(manifold_mixup_alpha) or manifold_mixup_alpha <= 0:
        raise ValueError("manifold_mixup_alpha must be a positive finite number.")
    if (
        not math.isfinite(manifold_mixup_probability)
        or not 0.0 <= manifold_mixup_probability <= 1.0
    ):
        raise ValueError("manifold_mixup_probability must be finite and in [0, 1].")
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    level_nodes = {}
    for node in tree.nodes:
        depth = len(tree.ancestors[node]) - 1
        level_nodes.setdefault(depth, []).append(tree.node2index[node])

    label_indices = torch.arange(len(labels_by_index), device=device)
    leaf_indices = torch.tensor(
        [
            tree.node2index[species_label_map[label]]
            for label in labels_by_index
        ],
        dtype=torch.long,
        device=device,
    )

    def evaluate_validation():
        classifier.eval()
        all_targets = []
        all_node_predictions = []
        all_labels = []
        all_species_predictions = []
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for embeddings, labels in validation_dataloader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                targets = target_lookup[labels]
                logits = classifier.forward_logits(embeddings)
                scores = torch.sigmoid(logits)
                components = focal_tree_min_logit_components(
                    logits, targets, tree, gamma=ftm_gamma
                )
                constrained_scores = tree.inference_probabilities(scores)
                species_scores = constrained_scores[:, leaf_indices]

                total_loss += components.sum(dim=1).sum().item()
                total_samples += embeddings.size(0)
                all_targets.append(targets.cpu())
                all_node_predictions.append(
                    (constrained_scores >= 0.5).cpu()
                )
                all_labels.append(labels.cpu())
                all_species_predictions.append(
                    species_scores.argmax(dim=1).cpu()
                )

        if not total_samples:
            return None

        targets = torch.cat(all_targets).numpy()
        node_predictions = torch.cat(all_node_predictions).numpy()
        labels = torch.cat(all_labels).numpy()
        species_predictions = torch.cat(all_species_predictions).numpy()
        species_macro_f1 = f1_score(
            labels,
            species_predictions,
            labels=label_indices.cpu().numpy(),
            average="macro",
            zero_division=0,
        )
        node_macro_f1 = f1_score(
            targets,
            node_predictions,
            average="macro",
            zero_division=0,
        )
        return total_loss / total_samples, species_macro_f1, node_macro_f1

    best_validation_f1 = float("-inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        samples = 0
        level_totals = {
            depth: {
                "loss": 0.0,
                "scores": 0.0,
                "targets": 0.0,
                "predictions": 0.0,
                "node_values": 0,
            }
            for depth in level_nodes
        }
        logger.info("Hierarchy epoch %d/%d", epoch + 1, num_epochs)
        for embeddings, labels in train_dataloader:
            embeddings = embeddings.to(device)
            targets = target_lookup[labels.to(device)]
            mixup_batch = manifold_mixup and (
                manifold_mixup_probability == 1.0
                or (
                    manifold_mixup_probability > 0.0
                    and torch.rand((), device=device).item()
                    < manifold_mixup_probability
                )
            )
            if mixup_batch:
                hidden = classifier.forward_hidden(embeddings)
                permutation = torch.randperm(
                    embeddings.size(0), device=device
                )
                mix_coefficients = torch.distributions.Beta(
                    manifold_mixup_alpha, manifold_mixup_alpha
                ).sample((embeddings.size(0),)).to(device)
                weights = mix_coefficients.unsqueeze(1)
                mixed_hidden = (
                    weights * hidden
                    + (1.0 - weights) * hidden.index_select(0, permutation)
                )
                logits = classifier.forward_hidden_logits(mixed_hidden)
                paired_targets = targets.index_select(0, permutation)
            else:
                logits = classifier.forward_logits(embeddings)
            scores = torch.sigmoid(logits)
            if mixup_batch:
                components = manifold_mixup_ftm_components(
                    logits,
                    targets,
                    paired_targets,
                    mix_coefficients,
                    tree,
                    gamma=ftm_gamma,
                )
            else:
                components = focal_tree_min_logit_components(
                    logits, targets, tree, gamma=ftm_gamma
                )
            loss = components.sum(dim=1).mean()
            constrained_scores = tree.inference_probabilities(scores.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = embeddings.size(0)
            running_loss += loss.item() * batch_size
            samples += batch_size
            for depth, node_indices in level_nodes.items():
                level_component = components[:, node_indices]
                level_score = scores.detach()[:, node_indices]
                level_target = targets[:, node_indices]
                level_prediction = (
                    constrained_scores[:, node_indices] >= 0.5
                )
                stats = level_totals[depth]
                stats["loss"] += level_component.sum().item()
                stats["scores"] += level_score.sum().item()
                stats["targets"] += level_target.sum().item()
                stats["predictions"] += level_prediction.sum().item()
                stats["node_values"] += batch_size * len(node_indices)

        logger.info(
            "Hierarchy epoch [%d/%d], FTM loss: %.4f",
            epoch + 1,
            num_epochs,
            running_loss / samples,
        )
        for depth, stats in level_totals.items():
            logger.info(
                "Hierarchy depth %d: FTM=%.4f, mean_score=%.4f, "
                "target_prevalence=%.4f, predicted_prevalence=%.4f",
                depth,
                stats["loss"] / samples,
                stats["scores"] / stats["node_values"],
                stats["targets"] / stats["node_values"],
                stats["predictions"] / stats["node_values"],
            )

        if validation_dataloader is not None:
            validation = evaluate_validation()
            validation_loss, species_f1, node_f1 = validation
            logger.info(
                "Validation: FTM=%.4f, species_macro_f1=%.4f, "
                "node_macro_f1=%.4f",
                validation_loss,
                species_f1,
                node_f1,
            )
            if species_f1 > best_validation_f1 + validation_min_delta:
                best_validation_f1 = species_f1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in classifier.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    early_stopping_patience is not None
                    and epochs_without_improvement
                    >= early_stopping_patience
                ):
                    logger.info(
                        "Stopping after %d epochs without validation "
                        "species macro-F1 improvement.",
                        epochs_without_improvement,
                    )
                    break

    if best_state is not None:
        classifier.load_state_dict(best_state)
    return classifier


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
    hierarchy_path=None,
    species_label_map=None,
    ftm_gamma=2.0,
    hidden_dim=1024,
    weight_decay=0.0,
    early_stopping_patience=5,
    validation_min_delta=0.0,
    manifold_mixup=False,
    manifold_mixup_alpha=0.2,
    manifold_mixup_probability=1.0,
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

    elif config in ("hierarchy", "hierarchical"):
        if learning_rate is None:
            learning_rate = bacpipe.settings.probe_configs["config_1"][
                "learning_rate"
            ]
        if num_epochs is None:
            num_epochs = bacpipe.settings.probe_configs["config_1"][
                "num_epochs"
            ]
        hierarchy = HierarchyTree.from_json(
            hierarchy_path or DEFAULT_HIERARCHY_PATH
        )
        resolved_species_map = get_species_label_map(
            label2index, hierarchy, species_label_map
        )
        probe = HierarchyProbe(
            in_dim=embed_size,
            hierarchy=hierarchy,
            label2index=label2index,
            species_label_map=resolved_species_map,
            hidden_dim=hidden_dim,
        )
        validation_gen = probe_dataset_loader(
            "val", df, embeds, label2index, **kwargs
        )
        probe = train_hierarchy_probe(
            probe,
            train_gen,
            label2index=label2index,
            hierarchy=hierarchy,
            species_label_map=resolved_species_map,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            ftm_gamma=ftm_gamma,
            validation_dataloader=validation_gen,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            validation_min_delta=validation_min_delta,
            manifold_mixup=manifold_mixup,
            manifold_mixup_alpha=manifold_mixup_alpha,
            manifold_mixup_probability=manifold_mixup_probability,
            device=kwargs.get("device", "cpu"),
        )

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
            f"Unknown probe config {config!r}. Expected 'linear', 'knn', "
            "'pulearn', or 'hierarchy'."
        )

    return probe
