# QuASIM-Own Benchmark Results

Generated: 2026-01-15T03:24:43.123275

Total runs: 25

## Summary by Task and Model

### tabular-cls

| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |
|-------|---------------|------------------|--------------|-----------|---------------|
| logreg | 0.6600 ± 0.0000 | 0.6610 ± 0.0000 | 0.09 | 1.000 | ✅ |
| mlp | 0.8770 ± 0.0160 | 0.8773 ± 0.0161 | 0.14 | 0.982 | ❌ |
| rf | 0.7850 ± 0.0138 | 0.7850 ± 0.0140 | 3.52 | 0.982 | ❌ |
| slt | 0.7780 ± 0.0186 | 0.7779 ± 0.0185 | 5.68 | 0.976 | ❌ |

### text-cls

| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |
|-------|---------------|------------------|--------------|-----------|---------------|
| slt | 0.9910 ± 0.0020 | 0.9910 ± 0.0020 | 13.92 | 0.998 | ❌ |

## Reliability-per-Watt Ranking

Computed as: `(stability × primary_metric) / energy_proxy`

| Rank | Task | Model | Reliability-per-Watt |
|------|------|-------|---------------------|
| 1 | tabular-cls | logreg | 113.292437 |
| 2 | tabular-cls | mlp | 96.050515 |
| 3 | tabular-cls | rf | 3.369061 |
| 4 | tabular-cls | slt | 2.055988 |
| 5 | text-cls | slt | 1.092951 |
