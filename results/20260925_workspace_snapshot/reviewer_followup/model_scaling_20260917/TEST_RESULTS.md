# Model-size extension: test

|Task|Model|Base|LoRA|Budget LoRA|LoRA+A-LoRA|
|---|---|---:|---:|---:|---:|
|cluener|qwen25_3b_base|27.009|64.435|64.504|66.180|
|cluener|qwen25_7b_base|35.823|69.189|69.219|70.466|
|wikisql|qwen25_3b_base|6.641|77.168|76.875|78.027|
|wikisql|qwen25_7b_base|50.293|81.484|81.445|82.129|

cluener / qwen25_3b_base
- stack − hidden: +1.7443, corrected CI [-0.2093, +3.6980], seeds [2.9474383365213157, 1.2889249130621465, 2.151402002034459, 1.5348636884809395, 0.7990932847412893]
- stack − hidden_budget: +1.6753, corrected CI [-0.1211, +3.4718], seeds [2.7302280392517844, 1.1903727681264087, 2.1916452560385125, 1.384590155054937, 0.8798048328726651]

cluener / qwen25_7b_base
- stack − hidden: +1.2773, corrected CI [-0.0890, +2.6437], seeds [2.2014232894847225, 1.1579724577551502, 0.8474181162113723, 0.753940735281887, 1.4259798505068773]
- stack − hidden_budget: +1.2473, corrected CI [-0.1729, +2.6675], seeds [2.1161784323387565, 1.2976384958497533, 0.9743890857174193, 0.4677235228645742, 1.3803774191075746]

wikisql / qwen25_3b_base
- stack − hidden: +0.8594, corrected CI [-0.2825, +2.0013], seeds [1.66015625, 0.87890625, 0.390625, 0.5859375, 0.78125]
- stack − hidden_budget: +1.1523, corrected CI [+0.2419, +2.0628], seeds [1.66015625, 1.26953125, 1.171875, 0.5859375, 1.07421875]

wikisql / qwen25_7b_base
- stack − hidden: +0.6445, corrected CI [-0.4997, +1.7888], seeds [1.171875, 0.9765625, 0.5859375, 0.5859375, -0.09765625]
- stack − hidden_budget: +0.6836, corrected CI [-0.8405, +2.2077], seeds [1.5625, 0.9765625, 0.78125, 0.09765625, 0.0]

3B: exact budget;7B: control has512 MORE trainable parameters. Selected previously positive tasks; no causal size-only claim.

Historical1.5B anchors (verified reuse; outside new statistical family):
cluener: {"base":0.0,"hidden":59.17261475217774,"hidden_both":62.565222304454025,"hidden_budget":59.2402085211657}
wikisql: {"base":7.12890625,"hidden":70.25390625,"hidden_both":72.65625,"hidden_budget":70.44921875}
