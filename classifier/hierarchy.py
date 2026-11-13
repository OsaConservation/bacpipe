"""Taxonomy helpers and Focal Tree-Min probing components."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_HIERARCHY_PATH = Path(__file__).resolve().parent.parent / "hierachry.json"


class HierarchyTree:
    """Validated rooted tree whose node labels come from a nested JSON object."""

    def __init__(self, nodes, parents):
        self.nodes = list(nodes)
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Hierarchy node labels must be unique.")
        if not self.nodes:
            raise ValueError("The hierarchy must contain at least one node.")

        self.node2index = {node: index for index, node in enumerate(self.nodes)}
        self.parents = dict(parents)
        if set(self.parents) != set(self.nodes):
            raise ValueError("Every hierarchy node must have exactly one parent entry.")
        if not any(parent is None for parent in self.parents.values()):
            raise ValueError("The hierarchy must have at least one root node.")
        for node, parent in self.parents.items():
            if parent is not None and parent not in self.node2index:
                raise ValueError(
                    f"Parent {parent!r} for hierarchy node {node!r} is missing."
                )

        self.children = {node: [] for node in self.nodes}
        for node, parent in self.parents.items():
            if parent is not None:
                self.children[parent].append(node)

        self.ancestors = {}
        self.descendants = {}
        for node in self.nodes:
            ancestors = []
            parent = self.parents[node]
            visited = {node}
            while parent is not None:
                if parent in visited:
                    raise ValueError("The hierarchy contains a cycle.")
                visited.add(parent)
                ancestors.append(parent)
                parent = self.parents[parent]
            ancestors.reverse()
            self.ancestors[node] = ancestors + [node]

            descendants = []
            pending = list(self.children[node])
            while pending:
                child = pending.pop()
                descendants.append(child)
                pending.extend(self.children[child])
            self.descendants[node] = [node] + descendants

        self.leaves = [
            node for node in self.nodes if not self.children[node]
        ]

    @classmethod
    def from_json(cls, hierarchy_path=None):
        path = Path(hierarchy_path or DEFAULT_HIERARCHY_PATH)
        with path.open(encoding="utf-8") as file:
            hierarchy = json.load(file)

        if not isinstance(hierarchy, dict) or not hierarchy:
            raise ValueError("Hierarchy JSON must contain a non-empty object.")

        nodes = []
        parents = {}

        def visit(branch, parent=None):
            if not isinstance(branch, dict):
                raise ValueError(
                    "Each hierarchy node must map to an object of child nodes."
                )
            for node, children in branch.items():
                if not isinstance(node, str) or not node:
                    raise ValueError("Hierarchy node labels must be non-empty strings.")
                if node in parents:
                    raise ValueError(
                        f"Hierarchy node label {node!r} occurs more than once."
                    )
                nodes.append(node)
                parents[node] = parent
                if not isinstance(children, dict):
                    raise ValueError(
                        f"Children of hierarchy node {node!r} must be an object."
                    )
                visit(children, node)

        visit(hierarchy)
        return cls(nodes, parents)

    def targets_for_species(self, species):
        """Return a binary node-presence vector for a leaf label."""
        if species not in self.node2index:
            raise ValueError(
                f"Species label {species!r} is not present in the hierarchy."
            )
        if species not in self.leaves:
            raise ValueError(
                f"Species label {species!r} maps to a non-leaf hierarchy node."
            )

        targets = torch.zeros(len(self.nodes), dtype=torch.float32)
        for node in self.ancestors[species]:
            targets[self.node2index[node]] = 1.0
        return targets

    def ftm_probabilities(self, scores, targets):
        """Return the hierarchy-conditioned probability of each true label."""
        constrained = []
        for node in self.nodes:
            index = self.node2index[node]
            ancestor_indices = [
                self.node2index[ancestor] for ancestor in self.ancestors[node]
            ]
            descendant_indices = [
                self.node2index[descendant]
                for descendant in self.descendants[node]
            ]
            present_probability = scores[:, ancestor_indices].amin(dim=1)
            absent_probability = 1.0 - scores[:, descendant_indices].amax(dim=1)
            constrained.append(
                torch.where(
                    targets[:, index].bool(),
                    present_probability,
                    absent_probability,
                )
            )
        return torch.stack(constrained, dim=1)

    def ftm_true_label_logits(self, logits, targets):
        """Return logits for the hierarchy-conditioned true label per node."""
        constrained = []
        for node in self.nodes:
            index = self.node2index[node]
            ancestor_indices = [
                self.node2index[ancestor] for ancestor in self.ancestors[node]
            ]
            descendant_indices = [
                self.node2index[descendant]
                for descendant in self.descendants[node]
            ]
            present_logit = logits[:, ancestor_indices].amin(dim=1)
            absent_logit = -logits[:, descendant_indices].amax(dim=1)
            constrained.append(
                torch.where(
                    targets[:, index].bool(),
                    present_logit,
                    absent_logit,
                )
            )
        return torch.stack(constrained, dim=1)

    def inference_probabilities(self, scores):
        """Return node scores constrained so children cannot exceed parents."""
        constrained = []
        for node in self.nodes:
            ancestor_indices = [
                self.node2index[ancestor] for ancestor in self.ancestors[node]
            ]
            constrained.append(scores[:, ancestor_indices].amin(dim=1))
        return torch.stack(constrained, dim=1)


def focal_tree_min_components(
    scores, targets, hierarchy, gamma=2.0, epsilon=1e-7
):
    """Return focal-loss terms for the hierarchy-conditioned true-label scores."""
    if scores.shape != targets.shape:
        raise ValueError(
            "Hierarchy scores and targets must have identical shapes; "
            f"received {tuple(scores.shape)} and {tuple(targets.shape)}."
        )
    if gamma < 0:
        raise ValueError("ftm_gamma must be non-negative.")

    probabilities = hierarchy.ftm_probabilities(scores, targets).clamp(
        min=0.0, max=1.0
    )
    return -(
        (1.0 - probabilities).pow(gamma)
        * torch.log(probabilities + epsilon)
    )


def focal_tree_min_loss(scores, targets, hierarchy, gamma=2.0, epsilon=1e-7):
    """Focal Tree-Min loss, summed over nodes and averaged over samples."""
    components = focal_tree_min_components(
        scores, targets, hierarchy, gamma=gamma, epsilon=epsilon
    )
    return components.sum(dim=1).mean()


def focal_tree_min_logit_components(logits, targets, hierarchy, gamma=2.0):
    """Compute stable FTM node losses directly from hierarchy-head logits."""
    if logits.shape != targets.shape:
        raise ValueError(
            "Hierarchy logits and targets must have identical shapes; "
            f"received {tuple(logits.shape)} and {tuple(targets.shape)}."
        )
    if gamma < 0:
        raise ValueError("ftm_gamma must be non-negative.")

    true_label_logits = hierarchy.ftm_true_label_logits(logits, targets)
    focal_factor = torch.sigmoid(-true_label_logits).pow(gamma)
    return -focal_factor * F.logsigmoid(true_label_logits)


class HierarchyProbe(nn.Module):
    """FTM hierarchy head: Linear -> ReLU -> Linear -> sigmoid.

    The hidden layer has 1024 units, matching the FTM-only architecture in
    Table 1 of "Hierarchy-Aware Representation Learning for Bio-acoustic
    Data". The final output has one presence score per taxonomy node.
    """

    def __init__(
        self,
        in_dim,
        hierarchy,
        label2index=None,
        species_label_map=None,
        hidden_dim=1024,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.hierarchy_head = nn.Sequential(
            nn.Linear(hidden_dim, len(hierarchy.nodes)),
            nn.Sigmoid(),
        )
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.node2index = dict(hierarchy.node2index)
        self.parents = dict(hierarchy.parents)
        self.label2index = dict(label2index or {})
        self.species_label_map = dict(species_label_map or {})

    def forward(self, embeddings):
        return self.hierarchy_head[1](self.forward_logits(embeddings))

    def forward_logits(self, embeddings):
        return self.forward_hidden_logits(self.forward_hidden(embeddings))

    def forward_hidden(self, embeddings):
        """Return the post-activation representation used by the output head."""
        return self.trunk(embeddings)

    def forward_hidden_logits(self, hidden):
        """Return hierarchy logits from hidden representations."""
        return self.hierarchy_head[0](hidden)

    def hierarchy(self):
        return HierarchyTree(self.node2index, self.parents)

    def inference_probabilities(self, embeddings):
        return self.hierarchy().inference_probabilities(self(embeddings))

    def checkpoint(self):
        return {
            "state_dict": self.state_dict(),
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "node2index": self.node2index,
            "parents": self.parents,
            "label2index": self.label2index,
            "species_label_map": self.species_label_map,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        nodes = sorted(
            checkpoint["node2index"],
            key=checkpoint["node2index"].__getitem__,
        )
        hierarchy = HierarchyTree(nodes, checkpoint["parents"])
        state_dict = checkpoint["state_dict"]
        probe = cls(
            in_dim=checkpoint.get(
                "in_dim", state_dict["trunk.0.weight"].shape[1]
            ),
            hierarchy=hierarchy,
            label2index=checkpoint.get("label2index"),
            species_label_map=checkpoint.get("species_label_map"),
            hidden_dim=checkpoint.get(
                "hidden_dim", state_dict["trunk.0.weight"].shape[0]
            ),
        )
        probe.load_state_dict(state_dict)
        return probe.to(device)


def get_species_label_map(label2index, hierarchy, species_label_map=None):
    """Resolve dataset labels to hierarchy leaves or raise with useful detail."""
    configured_map = dict(species_label_map or {})
    resolved = {}
    unknown = []
    for label in label2index:
        species = configured_map.get(label, label)
        if species not in hierarchy.node2index or species not in hierarchy.leaves:
            unknown.append(label)
        else:
            resolved[label] = species
    if unknown:
        raise ValueError(
            "These classifier labels do not match leaf nodes in the hierarchy: "
            f"{unknown}. Pass species_label_map={{dataset_label: hierarchy_leaf}}."
        )
    return resolved


def save_hierarchy_probe(probe, path):
    """Persist model weights and all taxonomy metadata needed at inference."""
    torch.save(probe.checkpoint(), path)


def load_hierarchy_probe(path, device="cpu"):
    """Load a hierarchy probe checkpoint created by this module."""
    checkpoint = torch.load(path, map_location=device)
    return HierarchyProbe.from_checkpoint(checkpoint, device=device)
