# Addressing the Abstraction and Reasoning Corpus via Procedural Example Generation

This repository presents code to procedurally generate examples for the [ARC](https://github.com/fchollet/ARC) training tasks. For each of the 400 tasks, an example generator is provided.

See the [demo notebook](demo.ipynb) for example usage of the code and a visualization of the data. The primary entry point is the `generate_dataset` function defined in [main.py](main.py). The only major dependency is the [ARC-DSL](https://github.com/michaelhodel/arc-dsl), which is however included as a single file in [dsl.py](dsl.py). Other relevant files are [generators.py](generators.py), which contains the task-specific example generators, and [verifiers.py](verifiers.py), which contains the corresponding task solver programs used for keeping only generated examples that are valid.


## Example usage:

```python 
demo_generator('00d62c1b')
```

### 00d62c1b (original)

![00d62c1b (original)](00d62c1b_original.png "00d62c1b (original)")


### 00d62c1b (generated)

![00d62c1b (generated)](00d62c1b_generated.png "00d62c1b (generated)")

