# BOTIOT Dataset Files

Place the BOTIOT JSON files in this directory:

- `statistics.json`
- `train.json`
- `test.json`

Expected sample schema for `train.json` and `test.json`:

```json
[
  {
    "label": 0,
    "len_seq": [60, 52, 1514],
    "ts_seq": [0.000000, 0.000300, 0.002100]
  }
]
```

Notes:
- `label` is an integer class id.
- `len_seq` and `ts_seq` must have the same length.
- `ts_seq` should be monotonically non-decreasing per sample.
