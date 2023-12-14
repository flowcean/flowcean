# Transforms

Transforms, also known as pre-processing of data, feature engineering, or augmentation, are fundamental processes in the AGenC framework.
These techniques enable users to manipulate and transform input datasets to produce modified output datasets.
This page will guide through the concept of transforms and demonstrate how to use them effectively within AGenC.

## Nomenclature

Transforms are a set of operations that modify the data in the dataset.
They can include operations such as data normalization, dimensionality reduction, data augmentation, and much more.
These transformations are essential for preparing data for machine learning tasks and improving model performance.

In AGenC, we use the generalized term _transform_ for all types of _pre-processing of data_, _feature engineering_, and _data augmentation_, as they all involve the same fundamental concept of transforming data to obtain a modified dataset.

AGenC provides a flexible and unified interface to apply transforms to data.
The framework allows to combine these transforming steps steps as needed.

## Using Transforms

Here's a basic example:

```python
from agenc.transforms import Select, Standardize

# Load the dataset
dataset: pl.DataFrame = ...

# Define transforms
selection = Select(features=["reference", "temperature"])
standardization = Standardize(
    mean={
        "reference": 0.0,
        "temperature": 0.0,
    },
    std={
        "reference": 1.0,
        "temperature": 1.0,
    },
)

# Apply the transforms to data
transformed_data = standardization(selection(dataset))
```

## Using the YAML experiment configuration

AGenC allows to compose multiple transforms and apply them in a specific order.
The desired order of transforms is specified in the experiment configuration.
This flexibility enables you to experiment with different combinations and orders to find the best pre-processing pipeline for the respective machine learning tasks.

This is an excerpt from an experiment configuration specifying two transforms:

```yaml
data:
  # ...
  transforms:
    - class_path: agenc.transforms.Select
      arguments:
        features:
          - reference
          - temperature
    - class_path: agenc.transforms.Standardize
      arguments:
        mean:
          reference: 0.0
          temperature: 0.0
        std:
          reference: 1.0
          temperature: 1.0
# ...
```
