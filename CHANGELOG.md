# Changelog

## [Unreleased](https://github.com/ParaConUK/cohobj/tree/HEAD)

[Full Changelog](https://github.com/ParaConUK/cohobj/compare/v0.3.0...HEAD)

## [v0.3.0](https://github.com/ParaConUK/cohobj/tree/v0.3.0)

[Full Changelog](https://github.com/ParaConUK/cohobj/compare/v0.2.0...v0.3.0)

*new features*

- Use of scikit-image library tools for object labelling and bounds.

- Greatly speeded up code to find and merge objects spanning the domain boundary.

- Hence overall speedup of `label_3D_cyclic` and so `get_object_labels` of about 2 orders of magnitude.

- Use of loguru.logger. 

To enable logging in a script using cohobj:

```python
from loguru import logger
logger.enable("cohobj")
```

## [v0.2.0](https://github.com/ParaConUK/cohobj/tree/v0.2.0)

[Full Changelog](https://github.com/ParaConUK/cohobj/compare/v0.1.0...v0.2.0)
