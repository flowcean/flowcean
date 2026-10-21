---
icon: lucide/funnel
---

# Transforms

Pre-processing of data, feature engineering, or augmentation, are fundamental processes in machine learning.
Flowcean generalizes these processes under the term _transforms_.
This page will guide through the concept of transforms and demonstrate how to use them within Flowcean.

## Nomenclature

Transforms are a set of operations that modify data.
They can include operations such as data normalization, dimensionality reduction, data augmentation, and much more.
These transformations are essential for preparing data for machine learning tasks and improving model performance.

In Flowcean, we use the generalized term _transform_ for all types of _pre-processing of data_, _feature engineering_, and _data augmentation_, as they all involve the same fundamental concept of transforming data to get a modified dataset.

Flowcean provides a flexible and unified interface to apply transforms to data.
The framework allows to combine these transforming steps as needed.

## Types and Composition

`Transform[Input, Output]` preserves the relationship between a transform's input and result.
Use `StatelessTransform[Input, Output]` for custom transforms that need no fitting; otherwise implement `Transform` and the fitting operations the transform supports.
For example, Polars transforms usually accept and return `pl.LazyFrame`.
Column names, dimensions, and valid values remain runtime requirements, not guarantees of the type annotation.

The `|` operator applies transforms from left to right and creates a new composition:

```python
from flowcean.core import Lambda


def format_value(value: int) -> str:
    return str(value)


pipeline = Lambda(format_value) | Lambda(len)
assert pipeline(123) == 3
```

## Optional Capabilities

`Invertible` promises that a transform supports inversion, although it may need fitting first.
Use `InvertibleLambda(func, inverse_func)` rather than `Lambda` when wrapping an invertible function.
For a fitted transform or a composition, `inverse(transform)` checks support and constructs the inverse in reverse order:

```python
from flowcean.core import inverse
from flowcean.polars import Standardize

scaler = Standardize().fit(data)
restored = inverse(scaler)(scaler(data))
```

Incremental fitting is a separate capability, `IncrementallyFittable`.
Stateless transforms support it as a no-op; batch-only transforms such as `Standardize` do not.
Use `fit_incremental(transform, batch)` for individual transforms or compositions.
It checks every component's support before updating any of them.
A general composition does not itself advertise `Invertible` or `IncrementallyFittable`; the checked operations inspect its components.
