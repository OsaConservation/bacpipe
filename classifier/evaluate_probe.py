import json
import torch
import torch.nn.functional as F

import sklearn.metrics as metrics
import numpy as np
from pathlib import Path
from .train_probe import probe_dataset_loader
from .hierarchy import focal_tree_min_loss, save_hierarchy_probe

from bacpipe.embedding_evaluation.visualization.visualize_predictions import (
    plot_classification_results,
)

from pulearn.metrics import (
    pu_recall_score,
    pu_precision_score,
    pu_f1_score,
    pu_specificity_score,
)


#  accuracy per class
def accuracy_per_class(y_true, y_pred, label2index, items_per_class):
    """
    Accuracy per class

    Parameters
    ----------
    y_true : list
        ground truth
    y_pred : list
        predictions
    label2index : dict
        link labels to ints
    items_per_class : list
        number of items per class

    Returns
    -------
    dict
        classwise accuracy
    """
    acc_per_cls_idx = {class_idx: 0 for class_idx in label2index.values()}

    for pred_class, true_class in zip(y_pred, y_true):
        if pred_class == true_class:
            acc_per_cls_idx[true_class] += 1

    accuracy_per_class = {
        class_label: acc_per_cls_idx[class_idx] / items_per_class[class_label]
        for class_label, class_idx in label2index.items()
    }

    return accuracy_per_class


def macro_accuracy(y_true, y_pred):
    """
    Compute macro accuracy.

    Parameters
    ----------
    y_true : list
        ground truth
    y_pred : list
        predictions

    Returns
    -------
    float
        balance accuracy score
    """
    return metrics.balanced_accuracy_score(y_true, y_pred)


def micro_accuracy(y_true, y_pred):
    """
    Compute micro accuracy.

    Parameters
    ----------
    y_true : list
        ground truth
    y_pred : list
        predictions

    Returns
    -------
    float
        micro accuracy score
    """
    return metrics.accuracy_score(y_true, y_pred)


#  Area under the ROC curve
# def auc(y_true, probability_scores):
#     """
#     Compute the AUC

#     Parameters
#     ----------
#     y_true : list
#         ground truth
#     probability_scores : np.array
#         probability scores for each class

#     Returns
#     -------
#     float
#         area under the ROC curve
#     """
#     if len(np.unique(y_true)) == 2:
#         probability_scores = np.array(probability_scores)[:, 1]
#     return metrics.roc_auc_score(y_true, probability_scores, multi_class="ovr")

def auc(y_true, probability_scores):
    """Compute macro one-vs-rest ROC-AUC from per-class scores.

    Works with both normalized multiclass probabilities and independent
    PU-learning scores. Classes without both positive and negative
    examples are excluded from the macro average.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(probability_scores, dtype=float)

    if scores.ndim != 2:
        raise ValueError(
            "Expected scores with shape (n_samples, n_classes); "
            f"received {scores.shape}."
        )

    if scores.shape[0] != len(y_true):
        raise ValueError(
            f"Score rows ({scores.shape[0]}) do not match "
            f"the number of labels ({len(y_true)})."
        )

    class_aucs = []

    for class_idx in range(scores.shape[1]):
        binary_true = (y_true == class_idx).astype(int)

        # ROC-AUC is undefined if this split lacks positives or negatives.
        if np.unique(binary_true).size < 2:
            continue

        class_auc = metrics.roc_auc_score(
            binary_true,
            scores[:, class_idx],
        )
        class_aucs.append(class_auc)

    if not class_aucs:
        return None

    return float(np.mean(class_aucs))


#  macro f1 score
def macro_f1(y_true, y_pred):
    """
    Compute the macro f1 score

    Parameters
    ----------
    y_true : list
        ground truth
    y_pred : list
        predictions

    Returns
    -------
    float
        macro f1 score
    """
    return metrics.f1_score(y_true, y_pred, average="macro")


#  micro f1 score
def micro_f1(y_true, y_pred):
    """
    Compute the micro f1 score

    Parameters
    ----------
    y_true : list
        ground truth
    y_pred : list
        predictions

    Returns
    -------
    float
        micro f1 score
    """
    return metrics.f1_score(y_true, y_pred, average="micro")


def compute_task_metrics(y_pred, y_true, probability_scores, label2index):
    """
    Compute the evaluation metrics

    Parameters
    ----------
    y_pred : list
        predictions
    y_true : list
        ground truth
    probability_scores : np.array
        probability scores for each class
    label2index : dict
        link labels to ints

    Returns
    -------
    dict
        dictionary with the overall and per class evaluation metrics
    """

    metrics = dict()

    metrics["overall"] = {
        "macro_accuracy": macro_accuracy(y_true, y_pred),
        "micro_accuracy": micro_accuracy(y_true, y_pred),
        "auc": (
            auc(y_true, probability_scores)
            if np.unique(y_true).size > 1
            else None
        ),
        "macro_f1": macro_f1(y_true, y_pred),
        "micro_f1": micro_f1(y_true, y_pred),
    }
    # if not metrics["overall"]["auc"]:
    #     metrics["overall"].pop("auc")
    if metrics["overall"]["auc"] is None:
        metrics["overall"].pop("auc")
    metrics["items_per_class"] = {
        name: y_true.count(idx) for name, idx in label2index.items()
    }
    metrics["per_class_accuracy"] = accuracy_per_class(
        y_true, y_pred, label2index, metrics["items_per_class"]
    )

    return metrics

def compute_task_metrics_pu(y_pred, y_true, probability_scores, label2index):
    """
    Compute the evaluation metrics

    Parameters
    ----------
    y_pred : list
        predictions
    y_true : list
        ground truth
    probability_scores : np.array
        probability scores for each class
    label2index : dict
        link labels to ints

    Returns
    -------
    dict
        dictionary with the overall and per class evaluation metrics
    """

    metrics = dict()

    metrics["overall"] = {
        "macro_accuracy": macro_accuracy(y_true, y_pred),
        "micro_accuracy": micro_accuracy(y_true, y_pred),
        "auc": (
            auc(y_true, probability_scores)
            if np.unique(y_true).size > 1
            else None
        ),
        "macro_f1": macro_f1(y_true, y_pred),
        "micro_f1": micro_f1(y_true, y_pred),
    }
    # if not metrics["overall"]["auc"]:
    #     metrics["overall"].pop("auc")
    if metrics["overall"]["auc"] is None:
        metrics["overall"].pop("auc")
    metrics["items_per_class"] = {
        name: y_true.count(idx) for name, idx in label2index.items()
    }
    metrics["per_class_accuracy"] = accuracy_per_class(
        y_true, y_pred, label2index, metrics["items_per_class"]
    )

    return metrics


def compute_hierarchical_metrics(y_true, y_pred, hierarchy):
    """Compute micro-averaged ancestor-augmented hP, hR, and hF1.

    The hierarchy has no explicit universal-root node, so all of its
    represented nodes (including top-level families) are included.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("Hierarchical labels must be 2D sample-by-node arrays.")
    expected_shape = (y_true.shape[0], len(hierarchy.nodes))
    if y_true.shape != y_pred.shape or y_true.shape != expected_shape:
        raise ValueError(
            "Hierarchical labels must have matching shapes with one column "
            f"per hierarchy node; received {y_true.shape} and {y_pred.shape}."
        )
    if y_true.shape[0] == 0:
        raise ValueError("Hierarchical metrics require at least one sample.")

    true_count = 0
    predicted_count = 0
    correct_count = 0
    for true_row, predicted_row in zip(y_true, y_pred):
        true_nodes = set()
        for index in np.flatnonzero(true_row):
            true_nodes.update(hierarchy.ancestors[hierarchy.nodes[index]])

        predicted_nodes = set()
        for index in np.flatnonzero(predicted_row):
            predicted_nodes.update(hierarchy.ancestors[hierarchy.nodes[index]])

        true_count += len(true_nodes)
        predicted_count += len(predicted_nodes)
        correct_count += len(true_nodes & predicted_nodes)

    hierarchical_precision = (
        correct_count / predicted_count if predicted_count else 0.0
    )
    hierarchical_recall = correct_count / true_count if true_count else 0.0
    denominator = hierarchical_precision + hierarchical_recall
    hierarchical_f1 = (
        2.0 * hierarchical_precision * hierarchical_recall / denominator
        if denominator
        else 0.0
    )
    return {
        "hP": hierarchical_precision,
        "hR": hierarchical_recall,
        "hF1": hierarchical_f1,
    }


def save_probe_results(paths, config, metrics, **kwargs):
    """
    Save a dict with all performance metrics.

    Parameters
    ----------
    paths : SimpleNamespace object
        dict with attributs of paths for loading and saving
    config : string
        type of classification (linear or knn)
    metrics : dict
        performance
    """

    for k, v in kwargs.items():
        if isinstance(v, Path):
            kwargs[k] = v.as_posix()
        elif isinstance(v, np.ndarray):
            kwargs[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating, np.bool_)):
            kwargs[k] = v.item()

    metrics["config"] = kwargs

    save_path = paths.probe_path.joinpath(f"probe_results_{config}.json")

    with open(save_path, "w") as f:
        # ``default=str`` keeps the dump from crashing on non-serializable
        # values (e.g. a ``CustomModel``/``CustomModels`` class object that is
        # forwarded through kwargs when a custom model is used).
        json.dump(metrics, f, indent=2, default=str)


def eval_probe(
    probe,
    embeds,
    df,
    label2index,
    device="cpu",
    config="linear",
    paths=None,
    save_probe=False,
    **kwargs,
):
    """
    Perform inference using probe.

    Parameters
    ----------
    probe : object
        trained classification object
    embeds : np.array
        embeddings
    df : pandas.DataFrame
        classification dataframe
    label2index : dict
        link labels to ints
    device : str, optional
        'cpu' or 'cuda', by default "cpu"
    config : str, optional
        type of classification, by default "linear"
    paths : SimpleNamespace object, optional
        paths used for saving the probe results, by default None
    save_probe : bool, optional
        whether to save the trained probe and label mapping, by
        default False. The evaluation results, confusion matrix and
        plots are always saved when ``paths`` is provided.

    Returns
    -------
    dict
        performance metrics
    """

    test_dataloader = probe_dataset_loader(
        "test", df, embeds, label2index, **kwargs
    )

    device = torch.device(device)
    probe = probe.to(device)

    probe.eval()
    y_pred = []
    y_true = []
    probabilities = []
    hierarchy_targets = []
    hierarchy_predictions = []
    ftm_loss_sum = 0.0
    evaluated_samples = 0
    hierarchy_tree = None
    target_lookup = None
    label_by_index = None
    if config in ("hierarchy", "hierarchical"):
        hierarchy_tree = probe.hierarchy()
        label_by_index = sorted(label2index, key=label2index.__getitem__)
        target_lookup = torch.stack(
            [
                hierarchy_tree.targets_for_species(
                    probe.species_label_map[label]
                )
                for label in label_by_index
            ]
        ).to(device)

    for embeddings, y in test_dataloader:
        embeddings, y = embeddings.to(device), y.to(device)

        outputs = probe(embeddings)
        if config in ("hierarchy", "hierarchical"):
            targets = target_lookup[y]
            loss = focal_tree_min_loss(
                outputs,
                targets,
                hierarchy_tree,
                gamma=kwargs.get("ftm_gamma", 2.0),
            )
            ftm_loss_sum += loss.item() * embeddings.size(0)
            evaluated_samples += embeddings.size(0)

            node_probs = hierarchy_tree.inference_probabilities(outputs)
            leaf_indices = [
                hierarchy_tree.node2index[probe.species_label_map[label]]
                for label in label_by_index
            ]
            probs = node_probs[:, leaf_indices].detach().cpu().numpy()
            predicted = torch.as_tensor(
                probs.argmax(axis=1),
                dtype=torch.long,
                device=device,
            )
            hierarchy_targets.extend(targets.detach().cpu().numpy())
            hierarchy_predictions.extend(
                (node_probs >= 0.5).detach().cpu().numpy()
            )
        elif config == "linear":
            # Use softmax to get probabilities
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
            _, predicted = torch.max(outputs, 1)

        elif config == "knn":
            # KNN does not require softmax
            predicted, probs = outputs
            probs = probs.cpu().numpy().tolist()

        elif config == "pulearn":
            predicted, probs = outputs
            probs = probs.detach().cpu().numpy().tolist()

        y_pred.extend(predicted.cpu().numpy().tolist())
        y_true.extend(y.cpu().numpy().tolist())
        probabilities.extend(probs)

    results = compute_task_metrics(y_pred, y_true, probabilities, label2index)

    if config in ("hierarchy", "hierarchical"):
        targets_array = np.asarray(hierarchy_targets, dtype=np.int8)
        predictions_array = np.asarray(
            hierarchy_predictions, dtype=np.int8
        )
        results["hierarchy"] = {
            "ftm_loss": ftm_loss_sum / evaluated_samples,
            **compute_hierarchical_metrics(
                targets_array, predictions_array, hierarchy_tree
            ),
            "node_micro_f1": metrics.f1_score(
                targets_array.reshape(-1),
                predictions_array.reshape(-1),
                average="binary",
                zero_division=0,
            ),
            "node_macro_f1": metrics.f1_score(
                targets_array,
                predictions_array,
                average="macro",
                zero_division=0,
            ),
            "node_binary_accuracy": metrics.accuracy_score(
                targets_array.reshape(-1),
                predictions_array.reshape(-1),
            ),
            "per_node_f1": {
                node: metrics.f1_score(
                    targets_array[:, index],
                    predictions_array[:, index],
                    zero_division=0,
                )
                for node, index in hierarchy_tree.node2index.items()
            },
        }

    if (
        save_probe
        and not paths is None
        and config in ("hierarchy", "hierarchical")
    ):
        save_hierarchy_probe(
            probe, paths.probe_path / "hierarchy_probe.pt"
        )
        with open(paths.probe_path / "label2index.json", "w") as f:
            json.dump(label2index, f, indent=1)
    elif save_probe and not paths is None and config == "pulearn":
        import joblib
        joblib.dump(probe, paths.probe_path / f"{config}_probe.joblib")
        with open(paths.probe_path / "label2index.json", "w") as f:
            json.dump(label2index, f, indent=1)

    elif save_probe and not paths is None:
        state_dict = probe.state_dict()
        torch.save(state_dict, paths.probe_path / f"{config}_probe.pt")
        with open(paths.probe_path / "label2index.json", "w") as f:
            json.dump(label2index, f, indent=1)

    if not paths is None:
        save_probe_results(paths, config, results, **kwargs)
        save_confusion_matrix(
            paths, y_true, y_pred, label2index, task_name=config
        )
        plot_classification_results(
            paths=paths, task_name=config, results=results
        )

    return results


def save_confusion_matrix(
    paths, y_true, y_pred, label2index, task_name="linear"
):
    """
    Save the confusion matrix as a plot to the probe path.

    Parameters
    ----------
    paths : SimpleNamespace object
        paths object used for saving the plot
    y_true : list
        ground truth values
    y_pred : list
        prediction values
    label2index : dict
        link labels to ints
    task_name : str, optional
        name of the task used in the plot file name, by default "linear"
    """
    from matplotlib import pyplot as plt

    i2l = {v: k for k, v in label2index.items()}
    labels_list = list(i2l.values())
    num_classes = len(labels_list)
    conf_matrx = metrics.confusion_matrix(
        [str(i2l[v]) for v in y_true],
        [str(i2l[v]) for v in y_pred],
        labels=[str(l) for l in labels_list],
    )

    conf_matrx_plot = metrics.ConfusionMatrixDisplay(
        conf_matrx, display_labels=labels_list
    )

    side_length = max(5, 5 + (num_classes * 0.8))
    fig, ax = plt.subplots(figsize=[side_length, side_length])

    conf_matrx_plot.plot(ax=ax)  # , cmap='Blues', values_format='d')
    ax.grid(False)

    ax.set_xlabel(
        "Predicted Label", fontsize=14, fontweight="bold", labelpad=15
    )
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold", labelpad=15)

    ax.tick_params(axis="both", which="major", labelsize=10)

    # Rotate tick labels if they get crowded
    if num_classes > 5:
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    fig.savefig(
        paths.probe_path / f"confusion_matrix_{task_name}.png",
        bbox_inches="tight",
    )
    plt.close(fig)
