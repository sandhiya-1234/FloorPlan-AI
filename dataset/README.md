# dataset/

This directory is intentionally empty. No verified dataset has been added yet.

```
dataset/
  raw/          real CubiCasa5K subset (Phase 2), git-ignored
  processed/    converted annotations + DATASET_MANIFEST.json (Phase 2)
  train/ validation/ test/    leakage-checked splits (Phase 2)
```

- The old 15-file synthetic set was moved to `docs/audit/invalid_artifacts/synthetic_fixture_dataset/`. It is not real data.
- `preprocessing/download_dataset.py` only generates synthetic fixtures and is deprecated. Do not use it to create data.
- Target data: CubiCasa5K, CC BY-NC-SA 4.0. See `docs/dataset.md` for the acceptance criteria that must hold before training.
