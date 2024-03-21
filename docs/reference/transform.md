# Transform

Transforms are all types of manipulations on data and, thus include methods known or labeled as "feature engineering", "preprocessing of data", "transformation of a data structure", and "data augmentation". Even though their purpose might be different, they share the same concept of receiving a data set as an input and returning a (possibly altered) data set as an output.

## Abstract Base Class

```{eval-rst}
.. autoclass:: agenc.core.Transform
   :members:
   :special-members: __call__
```

## Explode

```{eval-rst}
.. autoclass:: agenc.transforms.Explode
   :members:
```

## Select

```{eval-rst}
.. autoclass:: agenc.transforms.Select
   :members:
```

## Sliding Window

```{eval-rst}
.. autoclass:: agenc.transforms.SlidingWindow
   :members:
```

## Standardize

```{eval-rst}
.. autoclass:: agenc.transforms.Standardize
   :members:
```
