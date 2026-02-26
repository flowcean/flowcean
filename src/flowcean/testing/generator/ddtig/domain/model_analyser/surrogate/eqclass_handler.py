#!/usr/bin/env python3
from __future__ import annotations
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)

class EquivalenceClassesHandler():
    """
    A class used to extract equivalence classes from a decision tree.

    Attributes
    ----------
    test_tree : TestTree
        The decision tree structure.
    
    minmax_values_specs : dict
        Dictionary storing min/max values for each feature from specifications.
    
    n_samples : int
        Number of samples used to train the tree.
    
    n_features : int
        Number of features in the dataset.
    

    Methods
    -------
    get_equivalence_classes()
        Extracts and formats equivalence classes from the decision tree.
    
    to_str(eqclass)
        Converts a single equivalence class to a string.
    
    to_strs(eqclasses, feature_names)
        Converts a list of equivalence classes to a readable string format.
    """

    ROOT_INDEX = 0      # Root index of decision trees is always 0


    def __init__(
        self,
        test_tree,
        minmax_values_specs: dict,
        n_features: int,
    ) -> None:
        """
        Initializes the EquivalenceClassesHandler.

        Args:
            test_tree : The decision tree used for extracting equivalence classes.
            minmax_values_specs : Dictionary containing min/max values for each feature.
            n_features : Number of features in the dataset.
        """
        self.test_tree = test_tree.test_tree
        self.n_samples = test_tree.get_n_samples()
        self.minmax_values_specs = minmax_values_specs
        self.n_features = n_features
        self.eqclass_prio = []


    # Initializes an empty equivalence class with None bounds.
    # Equivalence class is stored a dictionary with:
    #   keys = feature indices
    #   values = min and max values of corresponding key 
    def _init_equivalence_class(self) -> dict:
        equivalence_class = {}
        for feature in range(self.n_features):
            range_ec = {'min': None, 'max': None}
            equivalence_class.update({feature: range_ec})
        return equivalence_class


    # Updates the min/max bounds of a feature in an equivalence class
    # based on the parent node's split.
    def _update_equivalence_class(self,
                                   equivalence_class: dict,
                                   parent: int,
                                   node: int) -> None:
        split_feature_idx = self.test_tree[parent].split_feature_idx
        split_threshold = self.test_tree[parent].split_threshold
        # Left child is reached when threshold is not exceeded -> Update max value of feature
        if (self.test_tree[parent].child_left == node):
            equivalence_class[split_feature_idx]['max'] = split_threshold 
        # Right child is reached when threshold is exceeded -> Update min value of feature    
        else:
            equivalence_class[split_feature_idx]['min'] = split_threshold


    # Recursively collects all paths from the root to leaf nodes.
    def _collect_paths(self, 
                        node: int, 
                        path: list,
                        paths: list) -> None:
        
        child_left = self.test_tree[node].child_left
        child_right = self.test_tree[node].child_right
        path.append(node)
        
        # If it's a leaf node, store the path
        if child_left == -1 and child_right == -1:
            paths.append(list(path))
        else:
            # Otherwise, explore both subtrees recursively
            self._collect_paths(child_left, path, paths)
            self._collect_paths(child_right, path, paths)

        # Backtrack: remove the last element from the path
        path.pop()


    # Collects all paths from the root to leaf nodes.
    def _collect_all_paths(self, root) -> list:
        path, paths = [], []
        self._collect_paths(root, path, paths)
        return paths

    # Extracts raw equivalence classes from tree paths.
    # Each leaf in the tree corresponds to one individual equivalence class.
    def _extract_equivalence_classes(self, paths) -> list:
        equivalence_classes = []

        # Traverse all paths from root to a leaf
        for path in paths:
            parent = path[0]
            equivalence_class = self._init_equivalence_class()
            for node in path[1:]:
                # At each node, update the ranges of features based on the
                # information stored in parent node
                self._update_equivalence_class(equivalence_class, parent, node)
                parent = node
            equivalence_classes.append(deepcopy(equivalence_class))
            if len(self.eqclass_prio) < len(paths):
                # Compute importance of each eqclass depending on number of training samples reached that class
                self.eqclass_prio.append(self.test_tree[parent].samples/self.n_samples)
            
        # Returns e.g., [{0: {'min': 0, 'max': 10}, 1: {'min': 3, 'max': None}}, {...}]
        return equivalence_classes
    

    # Formats raw equivalence classes into Interval objects.
    def _format_equivalence_classes(self, equivalence_classes) -> list:
        from flowcean.testing.generator.ddtig.domain import Interval, IntervalEndpoint
        equivalence_classes_formatted = []

        for eqclass in equivalence_classes:
            eqclass_formatted = ()
            for feature in range(len(eqclass)):
                min_value = eqclass[feature]['min']
                max_value = eqclass[feature]['max']
                left_endpoint = IntervalEndpoint.LEFT_OPEN
                right_endpoint = IntervalEndpoint.RIGHT_OPEN
                # If min is NULL, interval is left closed and min is from specifications
                if min_value == None:
                    left_endpoint = IntervalEndpoint.LEFT_CLOSED
                    min_value = self.minmax_values_specs[feature]['min']
                # If max is NULL, interval is right closed and max is from specifications
                if max_value == None:
                    right_endpoint = IntervalEndpoint.RIGHT_CLOSED
                    max_value = self.minmax_values_specs[feature]['max']
                interval = (Interval(feature, left_endpoint, right_endpoint, min_value, max_value),)

                # Each equivalence class is represented as a tuple with n elements,
                # where n is the number of features
                eqclass_formatted = eqclass_formatted + interval
            equivalence_classes_formatted.append(eqclass_formatted)
        # Returns: List of formatted equivalence classes as tuples of Intervals.
        return equivalence_classes_formatted
    
    
    def get_equivalence_classes(self) -> list:
        """
        Extracts and formats equivalence classes from the decision tree.

        Returns:
            List of formatted equivalence classes.
        """
        self.eqclass_prio = []
        paths = self._collect_all_paths(self.ROOT_INDEX)
        equivalence_classes = self._extract_equivalence_classes(paths)
        equivalence_classes_formatted = self._format_equivalence_classes(equivalence_classes)
        logger.info("Extracted equivalence classes successfully.")
        return equivalence_classes_formatted


    @staticmethod
    def to_str(eqclass: tuple) -> str:
        """
        Converts a single equivalence class to a string.

        Args:
            eqclass : A tuple of Interval objects.

        Returns:
            String representation of the equivalence class.
        """
        eqclass_str = "("
        eqclass_str += eqclass[0].__str__()
        for interval in eqclass[1:]:
            eqclass_str += ', '
            eqclass_str += interval.__str__()
        eqclass_str += ')'
        return eqclass_str


    @staticmethod
    def to_strs(eqclasses: list, feature_names: list) -> str:
        """
        Converts a list of equivalence classes to a readable string format.

        Args:
            eqclasses : List of equivalence classes.
            feature_names : List of feature names.

        Returns:
            Formatted string of all equivalence classes.
        """
        eqclasses_str = ""
        for i in range(len(eqclasses)):
            feature_idx = 0
            eqclasses_str += f"Equivalence class {i}:\n"
            eqclasses_str += "{"
            eqclasses_str += f"{feature_names[feature_idx]}: "
            eqclasses_str += eqclasses[i][feature_idx].__str__()
            for interval in eqclasses[i][1:]:
                feature_idx += 1
                eqclasses_str += ", "
                eqclasses_str += f"{feature_names[feature_idx]}: "
                eqclasses_str += interval.__str__()
            eqclasses_str += '}\n'
        return eqclasses_str   
        

    @staticmethod
    def is_subset(eqclass1: tuple,
                  eqclass2: tuple) -> tuple | None:
        """
        Compares the interval ranges of all features between two equivalence classes
        and determines which one is a subset of the other.

        An equivalence class is considered a subset only if all its intervals are
        strictly contained within the corresponding intervals of the other class.
        The method returns the superset equivalence class if such a relationship exists.

        Args:
            eqclass1 : First equivalence class (tuple of Interval objects).
            eqclass2 : Second equivalence class (tuple of Interval objects).

        Returns:
            The equivalence class that is the superset,
            or None if neither is a subset of the other.
        """
        from flowcean.testing.generator.ddtig.domain import Interval

        # Compare the first interval to determine initial superset
        intervalA = eqclass1[0]
        intervalB = eqclass2[0]
        interval_res = Interval.is_subset(intervalA, intervalB)

        if interval_res is None:
            return None
        
        # Identify which equivalence class contains the superset interval
        if (interval_res == intervalA):
            eqclass_large = eqclass1
        else:
            eqclass_large = eqclass2
        
        # Check consistency across all remaining intervals
        for idx in range(1, len(eqclass1)):
            intervalA = eqclass1[idx]
            intervalB = eqclass2[idx]
            interval_res = Interval.is_subset(intervalA, intervalB)

            if interval_res is None:
                return None
            
            # If any interval contradicts the initial superset assumption, return None
            elif ((interval_res == intervalA) and (eqclass_large != eqclass1) or
                  (interval_res == intervalB) and (eqclass_large != eqclass2)):
                return None
            
        return eqclass_large
