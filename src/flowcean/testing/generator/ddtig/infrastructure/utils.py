from typing import Any
from itertools import product


def get_all_combinations(first_val: Any,
                         second_val: Any,
                         n_vars: int) -> list:
    """
    Generate all distinct combinations of values for n variables,
    where each variable can take one of two specified values.

    Args:
        first_val: First possible value for each variable.
        second_val: Second possible value for each variable.
        n_vars: Number of variables.

    Returns:
        List of all possible combinations.
        Example: get_all_combinations(0, 1, 3) -> 
                 [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                  (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    """

    vals = []
    # Create a list of [first_val, second_val] for each variable
    for _ in range(n_vars):
        vals.append([first_val, second_val])
    # Generate all possible combinations using Cartesian product
    return list(product(*vals)) 


def split_tuple(tuple_to_split: tuple,
                subtuple_lens: list) -> tuple:
    """
    Split a tuple into subtuples of specified lengths.

    Args:
        tuple_to_split: The tuple to be split.
        subtuple_lens: List of lengths for each subtuple.

    Returns:
        A tuple of subtuples with the specified lengths.
        Example: split_tuple((1, 2, 3, 4, 5, 6), [3, 2, 1]) -> 
                 [(1, 2, 3), (4, 5), (6,)]
    """
    subtuples = []
    start_idx = 0
    end_idx = 0
    for length in subtuple_lens:
        end_idx += length
        subtuples.append(tuple_to_split[start_idx:end_idx])
        start_idx += length
    return subtuples


def reverse_list_by_value(numbers_list: list) -> list:
    """
    Reverse the values in a list based on their magnitude,
    preserving the original positions.

    Args:
        numbers_list: List of numeric values to reverse.

    Returns:
        A list where each value is replaced by its reversed counterpart
        based on magnitude.
        Example: reverse_list_by_value([1, 4, 2, 3]) -> [4, 1, 3, 2]
    """
    # Sort values in ascending order and reverse them
    sorted_probs = sorted(numbers_list)
    reversed_probs = sorted_probs[::-1]
    
    value_to_reversed = {}
    # Map each original value to its reversed counterpart
    for orig, rev in zip(sorted_probs, reversed_probs):
        if not orig in value_to_reversed:
            value_to_reversed[orig] = [rev]
        else:
            value_to_reversed[orig].append(rev)
    
    # Replace each value in the original list with its reversed version
    return [value_to_reversed[p].pop(0) for p in numbers_list]
