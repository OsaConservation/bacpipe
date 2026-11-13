import json

import numpy as np
import pandas as pd
import pytest
import torch

from classifier.evaluate_probe import (
    compute_hierarchical_metrics,
    eval_probe,
)
from classifier.hierarchy import (
    DEFAULT_HIERARCHY_PATH,
    HierarchyTree,
    focal_tree_min_loss,
    focal_tree_min_logit_components,
    load_hierarchy_probe,
    save_hierarchy_probe,
    get_species_label_map,
)
from classifier.inference_probe import (
    prepare_probe_inference,
    run_probe_inference,
)
from classifier.train_probe import train_probe
from classifier.train_probe import manifold_mixup_ftm_components


def small_tree():
    return HierarchyTree(
        ["family", "genus", "species_a", "species_b"],
        {
            "family": None,
            "genus": "family",
            "species_a": "genus",
            "species_b": "genus",
        },
    )


def test_hierarchy_json_and_species_targets(tmp_path):
    hierarchy_path = tmp_path / "hierachry.json"
    hierarchy_path.write_text(
        json.dumps({"family": {"genus": {"species_a": {}, "species_b": {}}}}),
        encoding="utf-8",
    )

    tree = HierarchyTree.from_json(hierarchy_path)
    target = tree.targets_for_species("species_a")

    assert tree.nodes == ["family", "genus", "species_a", "species_b"]
    assert target.tolist() == [1.0, 1.0, 1.0, 0.0]


def test_ftm_loss_backpropagates_and_inference_is_hierarchy_consistent():
    tree = small_tree()
    scores = torch.tensor(
        [[0.25, 0.75, 0.9, 0.1]], requires_grad=True
    )
    targets = tree.targets_for_species("species_a").unsqueeze(0)

    loss = focal_tree_min_loss(scores, targets, tree, gamma=2.0)
    loss.backward()
    probabilities = tree.inference_probabilities(scores)

    assert torch.isfinite(loss)
    assert torch.isfinite(scores.grad).all()
    assert probabilities[0, 1] <= probabilities[0, 0]
    assert probabilities[0, 2] <= probabilities[0, 1]


def test_hierarchical_metrics_micro_average_ancestor_augmented_sets():
    tree = small_tree()
    targets = np.zeros((2, len(tree.nodes)), dtype=np.int8)
    predictions = np.zeros_like(targets)
    targets[0, tree.node2index["species_a"]] = 1
    targets[1, tree.node2index["species_b"]] = 1
    predictions[0, tree.node2index["species_b"]] = 1
    predictions[1, tree.node2index["family"]] = 1

    scores = compute_hierarchical_metrics(targets, predictions, tree)

    assert scores == {"hP": 0.75, "hR": 0.5, "hF1": 0.6}


def test_ftm_loss_penalizes_all_positive_collapse():
    tree = small_tree()
    targets = tree.targets_for_species("species_a").unsqueeze(0)
    correct_scores = torch.tensor([[0.95, 0.95, 0.95, 0.05]])
    collapsed_scores = torch.ones_like(correct_scores)

    correct_loss = focal_tree_min_loss(correct_scores, targets, tree)
    collapsed_loss = focal_tree_min_loss(collapsed_scores, targets, tree)

    assert correct_loss < collapsed_loss
    assert correct_loss < 0.01
    assert collapsed_loss > 1.0


def test_ftm_absent_node_gradient_reduces_descendant_score():
    tree = small_tree()
    targets = tree.targets_for_species("species_a").unsqueeze(0)
    scores = torch.full((1, len(tree.nodes)), 0.8, requires_grad=True)

    focal_tree_min_loss(scores, targets, tree).backward()

    absent_species_index = tree.node2index["species_b"]
    assert scores.grad[0, absent_species_index] > 0


def test_ftm_loss_keeps_gradient_at_saturated_all_positive_scores():
    tree = small_tree()
    targets = tree.targets_for_species("species_a").unsqueeze(0)
    scores = torch.ones((1, len(tree.nodes)), requires_grad=True)

    loss = focal_tree_min_loss(scores, targets, tree)
    loss.backward()

    assert torch.isfinite(loss)
    assert scores.grad[0, tree.node2index["species_b"]] > 0


def test_logit_ftm_loss_has_gradient_after_sigmoid_saturation():
    tree = small_tree()
    targets = tree.targets_for_species("species_a").unsqueeze(0)
    logits = torch.full((1, len(tree.nodes)), 30.0, requires_grad=True)

    loss = focal_tree_min_logit_components(logits, targets, tree).sum()
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad[0, tree.node2index["species_b"]] > 0


def test_hierarchy_probe_matches_ftm_only_mlp_architecture():
    from classifier.hierarchy import HierarchyProbe

    tree = small_tree()
    probe = HierarchyProbe(in_dim=8, hierarchy=tree)
    output = probe(torch.randn(3, 8))

    assert probe.trunk[0].in_features == 8
    assert probe.trunk[0].out_features == 1024
    assert isinstance(probe.trunk[1], torch.nn.ReLU)
    assert probe.hierarchy_head[0].in_features == 1024
    assert probe.hierarchy_head[0].out_features == len(tree.nodes)
    assert isinstance(probe.hierarchy_head[1], torch.nn.Sigmoid)
    assert output.shape == (3, len(tree.nodes))
    assert torch.all((output >= 0.0) & (output <= 1.0))
    embeddings = torch.randn(3, 8)
    hidden = probe.forward_hidden(embeddings)
    hidden_logits = probe.forward_hidden_logits(hidden)
    torch.testing.assert_close(
        probe.forward_logits(embeddings),
        hidden_logits,
    )


def test_manifold_mixup_ftm_components_use_weighted_original_targets():
    tree = small_tree()
    targets = torch.stack(
        [
            tree.targets_for_species("species_a"),
            tree.targets_for_species("species_b"),
        ]
    )
    paired_targets = targets.flip(0)
    logits = torch.randn_like(targets, requires_grad=True)
    coefficients = torch.tensor([0.25, 0.75])

    components = manifold_mixup_ftm_components(
        logits,
        targets,
        paired_targets,
        coefficients,
        tree,
        gamma=2.0,
    )
    expected = (
        coefficients.unsqueeze(1)
        * focal_tree_min_logit_components(logits, targets, tree)
        + (1.0 - coefficients.unsqueeze(1))
        * focal_tree_min_logit_components(logits, paired_targets, tree)
    )
    torch.testing.assert_close(components, expected)
    components.sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("alpha", [0.0, -0.5, float("inf"), float("nan")])
def test_hierarchy_training_rejects_invalid_manifold_mixup_alpha(alpha):
    labels = ["callidryas"]
    label2index = {"callidryas": 0}
    df = pd.DataFrame(
        {"label": labels, "predefined_set": ["train"]}
    )
    embeds = np.asarray([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="manifold_mixup_alpha"):
        train_probe(
            embeds,
            df,
            label2index,
            config="hierarchy",
            hierarchy_path=DEFAULT_HIERARCHY_PATH,
            num_epochs=1,
            batch_size=1,
            manifold_mixup_alpha=alpha,
        )


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("inf"), float("nan")])
def test_hierarchy_training_rejects_invalid_manifold_mixup_probability(
    probability,
):
    labels = ["callidryas"]
    label2index = {"callidryas": 0}
    df = pd.DataFrame(
        {"label": labels, "predefined_set": ["train"]}
    )
    embeds = np.asarray([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="manifold_mixup_probability"):
        train_probe(
            embeds,
            df,
            label2index,
            config="hierarchy",
            hierarchy_path=DEFAULT_HIERARCHY_PATH,
            num_epochs=1,
            batch_size=1,
            manifold_mixup_probability=probability,
        )


def test_manifold_mixup_disabled_matches_default_training():
    labels = ["callidryas", "spurrelli"]
    label2index = {label: index for index, label in enumerate(labels)}
    df = pd.DataFrame(
        {
            "label": labels * 2,
            "predefined_set": ["train", "train", "test", "test"],
        }
    )
    embeds = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]],
        dtype=np.float32,
    )
    options = {
        "config": "hierarchy",
        "learning_rate": 0.01,
        "num_epochs": 1,
        "batch_size": 2,
        "hierarchy_path": DEFAULT_HIERARCHY_PATH,
        "hidden_dim": 4,
        "device": "cpu",
    }

    torch.manual_seed(42)
    default_probe = train_probe(embeds, df, label2index, **options)
    torch.manual_seed(42)
    disabled_probe = train_probe(
        embeds,
        df,
        label2index,
        manifold_mixup=False,
        **options,
    )

    for key, value in default_probe.state_dict().items():
        torch.testing.assert_close(value, disabled_probe.state_dict()[key])


def test_hierarchy_training_supports_manifold_mixup_without_mixup_defaults():
    labels = ["callidryas", "spurrelli"]
    label2index = {label: index for index, label in enumerate(labels)}
    df = pd.DataFrame(
        {
            "label": labels * 2,
            "predefined_set": ["train", "train", "test", "test"],
        }
    )
    embeds = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]],
        dtype=np.float32,
    )

    probe = train_probe(
        embeds,
        df,
        label2index,
        config="hierarchy",
        learning_rate=0.01,
        num_epochs=1,
        batch_size=2,
        hierarchy_path=DEFAULT_HIERARCHY_PATH,
        hidden_dim=4,
        manifold_mixup=True,
        manifold_mixup_alpha=0.2,
        device="cpu",
    )

    assert probe.hidden_dim == 4
    assert torch.isfinite(probe.forward_logits(torch.as_tensor(embeds))).all()


def test_zero_manifold_mixup_probability_matches_unmixed_training():
    labels = ["callidryas", "spurrelli"]
    label2index = {label: index for index, label in enumerate(labels)}
    df = pd.DataFrame(
        {
            "label": labels * 2,
            "predefined_set": ["train", "train", "test", "test"],
        }
    )
    embeds = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]],
        dtype=np.float32,
    )
    options = {
        "config": "hierarchy",
        "learning_rate": 0.01,
        "num_epochs": 1,
        "batch_size": 2,
        "hierarchy_path": DEFAULT_HIERARCHY_PATH,
        "hidden_dim": 4,
        "device": "cpu",
    }

    torch.manual_seed(42)
    baseline_probe = train_probe(embeds, df, label2index, **options)
    torch.manual_seed(42)
    zero_probability_probe = train_probe(
        embeds,
        df,
        label2index,
        manifold_mixup=True,
        manifold_mixup_probability=0.0,
        **options,
    )

    for key, value in baseline_probe.state_dict().items():
        torch.testing.assert_close(
            value, zero_probability_probe.state_dict()[key]
        )


def test_training_applies_custom_hidden_dim_and_validation_stopping(
    caplog, tmp_path
):
    labels = ["callidryas", "spurrelli"]
    label2index = {label: index for index, label in enumerate(labels)}
    df = pd.DataFrame(
        {
            "label": labels * 3,
            "predefined_set": [
                "train",
                "train",
                "test",
                "test",
                "val",
                "val",
            ],
        }
    )
    embeds = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )

    probe = train_probe(
        embeds,
        df,
        label2index,
        config="hierarchy",
        learning_rate=0.0,
        num_epochs=5,
        batch_size=2,
        hierarchy_path=DEFAULT_HIERARCHY_PATH,
        hidden_dim=4,
        early_stopping_patience=1,
        device="cpu",
    )

    assert probe.hidden_dim == 4
    assert probe.trunk[0].out_features == 4
    assert "species_macro_f1" in caplog.text
    assert "node_macro_f1" in caplog.text
    assert "target_prevalence" in caplog.text
    assert "Stopping after 1 epochs" in caplog.text
    checkpoint_path = tmp_path / "custom_width.pt"
    save_hierarchy_probe(probe, checkpoint_path)
    loaded_probe = load_hierarchy_probe(checkpoint_path)
    assert loaded_probe.hidden_dim == 4


def test_dataset_abbreviations_require_an_explicit_leaf_mapping():
    hierarchy = HierarchyTree.from_json(DEFAULT_HIERARCHY_PATH)
    with pytest.raises(ValueError, match="species_label_map"):
        get_species_label_map({"DENMIC": 0}, hierarchy)

    assert get_species_label_map(
        {"DENMIC": 0}, hierarchy, {"DENMIC": "microcephalus"}
    ) == {"DENMIC": "microcephalus"}


def test_hierarchy_train_evaluate_and_inference_round_trip(tmp_path):
    labels = ["callidryas", "spurrelli"]
    label2index = {label: index for index, label in enumerate(labels)}
    df = pd.DataFrame(
        {
            "label": [
                "callidryas",
                "callidryas",
                "spurrelli",
                "spurrelli",
            ],
            "predefined_set": ["train", "test", "train", "test"],
        }
    )
    embeds = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )
    species_label_map = {
        "callidryas": "callidryas",
        "spurrelli": "spurrelli",
    }

    probe = train_probe(
        embeds,
        df,
        label2index,
        config="hierarchy",
        learning_rate=0.01,
        num_epochs=2,
        batch_size=2,
        hierarchy_path=DEFAULT_HIERARCHY_PATH,
        species_label_map=species_label_map,
        device="cpu",
    )
    metrics = eval_probe(
        probe,
        embeds,
        df,
        label2index,
        config="hierarchy",
        device="cpu",
    )
    checkpoint_path = tmp_path / "hierarchy_probe.pt"
    save_hierarchy_probe(probe, checkpoint_path)
    loaded, loaded_labels = prepare_probe_inference(
        "testmodel",
        name="hierarchy",
        probe_path=str(checkpoint_path),
        device="cpu",
    )
    direct_loaded = load_hierarchy_probe(checkpoint_path, device="cpu")
    predictions = run_probe_inference(
        "testmodel",
        loaded,
        name="hierarchy",
        embeds=embeds,
        return_binary_presence=False,
        device="cpu",
    )

    assert loaded_labels == label2index
    assert predictions.shape == (len(embeds), len(probe.node2index))
    assert predictions.dtype == np.float32
    assert np.allclose(
        predictions,
        run_probe_inference(
            "testmodel",
            direct_loaded,
            name="hierarchy",
            embeds=embeds,
            return_binary_presence=False,
            device="cpu",
        ),
    )
    assert "node_micro_f1" in metrics["hierarchy"]
    assert {"hP", "hR", "hF1"} <= metrics["hierarchy"].keys()
