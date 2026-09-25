# Model-size extension: dev

|Task|Model|Base|LoRA|Budget LoRA|LoRA+A-LoRA|
|---|---|---:|---:|---:|---:|
|cluener|qwen25_3b_base|28.317|65.814|65.845|66.804|
|cluener|qwen25_7b_base|42.516|70.601|70.781|71.734|
|wikisql|qwen25_3b_base|6.641|81.328|80.938|82.188|
|wikisql|qwen25_7b_base|51.953|84.766|84.375|84.531|

cluener / qwen25_3b_base
- stack − hidden: +0.9895, corrected CI [-1.7940, +3.7730], seeds [-0.3414714179108387, 2.186624372191389, 0.6122384772691447, 2.2748687575716815, 0.2153526639507959]
- stack − hidden_budget: +0.9589, corrected CI [-2.1383, +4.0561], seeds [-0.963233075941929, 2.1133493019869434, 1.6993138384561775, 1.7957943654538155, 0.14930082083367324]

cluener / qwen25_7b_base
- stack − hidden: +1.1328, corrected CI [-2.2261, +4.4917], seeds [2.003231689259053, 2.974811409930865, -0.2760340421308598, 1.2449974045111247, -0.28290957868422595]
- stack − hidden_budget: +0.9526, corrected CI [-2.2413, +4.1465], seeds [1.8098450598585316, 2.8319135588463524, -0.34896428394019097, 0.6715588918436737, -0.20120724346077168]

wikisql / qwen25_3b_base
- stack − hidden: +0.8594, corrected CI [-1.5196, +3.2383], seeds [1.953125, 1.5625, -0.390625, 1.171875, 0.0]
- stack − hidden_budget: +1.2500, corrected CI [-0.8458, +3.3458], seeds [2.34375, 1.953125, 0.390625, 1.171875, 0.390625]

wikisql / qwen25_7b_base
- stack − hidden: -0.2344, corrected CI [-4.6229, +4.1541], seeds [1.953125, 1.171875, -0.390625, -1.171875, -2.734375]
- stack − hidden_budget: +0.1562, corrected CI [-4.0854, +4.3979], seeds [2.34375, 1.5625, 0.0, -1.171875, -1.953125]

3B: exact budget;7B: control has512 MORE trainable parameters. Selected previously positive tasks; no causal size-only claim.
