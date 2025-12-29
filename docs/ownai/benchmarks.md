# QuASIM-Own Benchmark Results

Generated: 2025-12-29T03:42:35.405490

Total runs: 25

## Summary by Task and Model

### tabular-cls

| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |
|-------|---------------|------------------|--------------|-----------|---------------|
| logreg | 0.6600 ± 0.0000 | 0.6610 ± 0.0000 | 0.09 | 1.000 | ✅ |
| mlp | 0.8770 ± 0.0160 | 0.8773 ± 0.0161 | 0.17 | 0.982 | ❌ |
| rf | 0.7850 ± 0.0138 | 0.7850 ± 0.0140 | 3.89 | 0.982 | ❌ |
| slt | 0.7780 ± 0.0186 | 0.7779 ± 0.0185 | 6.22 | 0.976 | ❌ |

### text-cls

| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |
|-------|---------------|------------------|--------------|-----------|---------------|
| slt | 0.9910 ± 0.0020 | 0.9910 ± 0.0020 | 14.78 | 0.998 | ❌ |

## Reliability-per-Watt Ranking

Computed as: `(stability × primary_metric) / energy_proxy`

| Rank | Task | Model | Reliability-per-Watt |
|------|------|-------|---------------------|
| 1 | tabular-cls | logreg | 113.587098 |
| 2 | tabular-cls | mlp | 78.307787 |
| 3 | tabular-cls | rf | 3.053323 |
| 4 | tabular-cls | slt | 1.878803 |
| 5 | text-cls | slt | 1.029240 |
