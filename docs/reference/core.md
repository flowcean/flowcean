---
icon: lucide/boxes
---

# Core

`flowcean.core` provides the shared interfaces and learning strategies used across Flowcean.
See [learning strategies](../user_guide/learning_strategies.md) for how environments, learners, and models fit together.

## Implementing Typed Components

Use concrete representations where an operation requires them, and generic parameters where the relationship between inputs and results matters.
`Model[Input, Output]`, `Transform[Input, Output]`, and `Environment[Observation]` are implementation bases; `Predictor` and the optional capability protocols describe structural requirements.
Learners parameterize their training inputs, training outputs, and concrete returned model independently.
The built-in offline and incremental learning strategies operate on lazy Polars frames, not arbitrary data representations.

Library function signatures are explicit and checked with basedpyright through `just check-types`; Ruff checks annotation hygiene through `just check`.
Tests and examples are checked as consumers without requiring exhaustive annotations.
Unknown local inference from dependencies is permitted, so checking does not prove the absence of `Any`.
Keep dynamic values at narrow integration boundaries rather than using them in Flowcean-owned contracts.
Use rule-qualified `# pyright: ignore[...]` comments only for justified exceptions.

Types describe representations and capabilities, not data-dependent schemas, array dimensions, fitted state, or the safety of deserialized objects.
Validate those requirements at the operation that depends on them.
See [transforms](../user_guide/transform.md), [models](../user_guide/model.md), and [environments](../user_guide/environment.md) for extension and composition behavior.

## API

::: flowcean.core
