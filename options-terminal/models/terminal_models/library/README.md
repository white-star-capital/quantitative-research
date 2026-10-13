# library/

One package per implementation (MODELS.md §2):

```
<implementation_id>/
├── __init__.py
├── manifest.yaml     # MODELS.md §3.3
├── model.py          # subclass of core.base.Model
├── README.md         # methodology, assumptions, known limits
└── tests/
    ├── fixtures/     # recorded real inputs + expected outputs; never invented prices
    └── test_model.py
```

Implementations import only from `terminal_models.core` and the scientific stack (M-2, enforced by `lint-imports`).
