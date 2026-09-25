# Fixed-hidden A-LoRA rank sweep

All endpoints use hidden seed 42 and functional scale 16. `train1000` is the predeclared first 1,000 training records.

| rank | alpha | boundary params | train1000 CE | dev CE | test CE | dev−train |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 6,144 | 1.117070340 | 1.112973366 | 1.132496473 | -0.004096974 |
| 4 | 64 | 12,288 | 1.116460634 | 1.112711213 | 1.132136936 | -0.003749422 |
| 8 | 128 | 24,576 | 1.115131792 | 1.112059818 | 1.131511148 | -0.003071973 |
| 16 | 256 | 49,152 | 1.113038024 | 1.111045905 | 1.130232467 | -0.001992119 |
| 32 | 512 | 98,304 | 1.109090478 | 1.108875296 | 1.128488106 | -0.000215182 |
| 50 | 800 | 153,600 | 1.104395854 | 1.107084474 | 1.126872773 | +0.002688620 |

- Best dev rank: `50` (CE `1.107084474`).
- Descriptive best test rank: `50` (CE `1.126872773`).

This single-seed diagnostic distinguishes rank behavior within the fixed-hidden output-only protocol; it is not a cross-seed claim.
