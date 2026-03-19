from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from river.tree.nodes.branch import DTBranch
from sklearn.tree._tree import Tree as sklearnTree

if TYPE_CHECKING:
    from river.tree import HoeffdingTreeClassifier, HoeffdingTreeRegressor

    from flowcean.testing.generator.ddtig.user_interface import (
        SystemSpecsHandler,
    )

logger = logging.getLogger(__name__)

def convert_river_tree(river_tree: HoeffdingTreeRegressor | HoeffdingTreeClassifier,
                       feature_dict: dict) -> dict:
    """Extracts the structure of a River Hoeffding tree and stores it in a dictionary.

    Args:
        river_tree : A River Hoeffding tree.
        feature_dict : Dictionary mapping feature names to their indices.

    Returns:
        A dictionary representing the tree structure.
    """
    counter = 0
    # Traverse the tree using depth-first search.
    # Yields: parent index, parent node, child node, child index, branch
    def iterate(node=None) -> Iterator[tuple]:
            if node is None:
                yield None, None, river_tree._root, 0, None
                yield from iterate(river_tree._root)

            nonlocal counter
            parent_no = counter

            if isinstance(node, DTBranch):
                for branch_index, child in enumerate(node.children):
                    counter += 1
                    yield parent_no, node, child, counter, branch_index
                    if isinstance(child, DTBranch):
                        yield from iterate(child)

    tree = {}

    # Build tree structure from traversal
    for parent_no, parent, child, child_no, branch_index in iterate():

        # Store new node in dict
        if tree.get(child_no) is None:
            new_node = Node()
            tree[child_no] = new_node

        # Store feature of node if node is not a leaf
        if isinstance(child, DTBranch):
            tree[child_no].split_feature = child.feature
            tree[child_no].split_feature_idx = feature_dict[child.feature]
        else:
            # Get #samples for leaf node
            tree[child_no].samples = int(child.total_weight)

        # If node has a parent, store node as either child left
        # or child right of parent node
        if parent_no is not None:
            if (branch_index == 0):
                tree[parent_no].child_left = child_no
            else:
                tree[parent_no].child_right = child_no
            tree[parent_no].split_threshold = float(parent.repr_branch(branch_index, shorten=True).split(" ")[-1])
    return tree


def convert_sklearn_tree(sklearn_tree: sklearnTree,
                         feature_dict: dict) -> dict:
    """Extracts the structure of a scikit-learn decision tree and stores it in a dictionary.

    Args:
        sklearn_tree : A scikit-learn decision tree.
        feature_dict : Dictionary mapping feature indices to feature names.

    Returns:
        A dictionary representing the tree structure.
    """
    n_nodes = sklearn_tree.node_count
    tree = {}

    # Traverse each node in the scikit-learn tree
    for node in range(n_nodes):
        split_feature = feature_dict.get(sklearn_tree.feature[node])
        if split_feature is None:
            split_feature = -2
        # Store info of sklearn node in our own Node object
        new_node = Node(child_left=sklearn_tree.children_left[node],
                        child_right=sklearn_tree.children_right[node],
                        split_feature=split_feature,
                        split_feature_idx=sklearn_tree.feature[node],
                        split_threshold=sklearn_tree.threshold[node],
                        )
        # Get #samples for leaf node
        if new_node.child_left == -1 and new_node.child_right == -1:
            new_node.samples = sklearn_tree.n_node_samples[node]
        tree[node] = new_node
    return tree


@dataclass
class Node:
    """Represents a node in a TestTree."""

    def __init__(
        self,
        child_left: int = -1,
        child_right: int = -1,
        split_feature: str | int = -2,
        split_feature_idx: int = -2,
        split_threshold: float = -2.0,
        samples: int = 0,
    ) -> None:
        """Initializes a node.

        Args:
            child_left : Index of the left child node (-1 if leaf).
            child_right : Index of the right child node (-1 if leaf).
            split_feature : Feature used for splitting (-2 if leaf).
            split_feature_idx : Index of the split feature (-2 if leaf).
            split_threshold : Threshold value for splitting (-2.0 if leaf).
            samples : Number of samples reaching this node (0 if not a leaf).
        """
        self.child_left = child_left
        self.child_right = child_right
        self.split_feature = split_feature
        self.split_feature_idx = split_feature_idx
        self.split_threshold = split_threshold
        self.samples = samples

    def __str__(self) -> str:
        return f"\nchild_left: {self.child_left},\nchild_right: {self.child_right},\nsplit_feature: {self.split_feature},\nsplit_feature_idx: {self.split_feature_idx},\nsplit_threshold: {self.split_threshold}, \nsamples: {self.samples}"

@dataclass
class TestTree:
    """Represents a tree structure used for generating test inputs.

    Attributes:
    ----------
    test_tree : dict
        Dictionary representing the structure of a River or scikit-learn tree.

    Methods:
    -------
    get_n_samples()
        Returns the total number of samples used to train the tree.
    """

    def __init__(
        self,
        model_tree: HoeffdingTreeRegressor | HoeffdingTreeClassifier | sklearnTree,
        specs_handler: SystemSpecsHandler,
    ) -> None:
        """Initializes the TestTree from a model tree.

        Args:
            model_tree : A River or scikit-learn decision tree.
            specs_handler : Object for accessing feature specifications.
        """
        if isinstance(model_tree, sklearnTree):
            feature_dict = specs_handler.extract_feature_names_with_idx_reversed()
            self.test_tree = convert_sklearn_tree(model_tree, feature_dict)
            logger.info("Converted a scikit-learn tree to TestTree successfully.")
        else:
            feature_dict = specs_handler.extract_feature_names_with_idx()
            self.test_tree = convert_river_tree(model_tree, feature_dict)
            logger.info("Converted a River tree to TestTree successfully.")

    def get_n_samples(self) -> int:
        """Returns the total number of samples used to train the tree.

        Returns:
            Total number of samples.
        """
        samples = 0
        for key in self.test_tree:
            if self.test_tree[key].samples != 0:
                samples += self.test_tree[key].samples
        return samples


    def __str__(self) -> str:
        tree_str = ""
        for idx, node in self.test_tree.items():
            tree_str += f"\nNode {idx}: {node!s}\n"
        return tree_str
