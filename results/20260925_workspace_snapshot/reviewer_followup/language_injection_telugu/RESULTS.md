# Telugu cross-script task-injection pilot

## Scope and protocol

This is a deliberately narrow first language-injection test, **not** a claim
of general Telugu instruction tuning. Direct Hugging Face access for the
planned 67k Telugu Alpaca/Dolly corpus was blocked at the upstream Xet object
store, so the pilot uses the accessible official Telugu (`tel_Telu`) Belebele
multiple-choice reading-comprehension data. It tests whether the final
affine/energy mechanism improves learning of a new-script target task, not
whether it lowers held-out CE.

The 900 Telugu rows were shuffled with split seed `20260724`; 720 rows were
reserved for training and 180 for test. Each rendered task contains the Telugu
passage, question, and four Telugu answers, and the assistant target is exactly
the correct option number. Qwen2.5 tokenization required a 2048-token context:
at 1024 it retained only 220 train / 47 test examples, whereas the fixed 2048
filter retained 717 train / 176 test rows. The four options are scored by their
next-token log likelihood, so the metric is direct accuracy rather than
generation-format or held-out CE.

All runs use Qwen2.5-1.5B-Base, the tied-transpose affine input/LM-head
adapter (rank 16, alpha 128), hidden LoRA (rank 8, alpha 16), the same English
explicit-constraint auxiliary anchor CE=.1, and 32 epochs / 1440 optimizer
steps. The matched control has no energy or reference KL. The candidate adds
the final configuration unchanged: energy=100 over steps [0,400) and total
teacher KL=.05 averaged across the Telugu main batch and English anchor batch.
Eight seeds (59--66) ran in parallel for each arm.

## Target-task result

| seed | matched control | early-energy + dual-KL candidate | candidate - control |
| ---: | ---: | ---: | ---: |
| 59 | 0.335227 | 0.306818 | -0.028409 |
| 60 | 0.289773 | 0.340909 | +0.051136 |
| 61 | 0.318182 | 0.323864 | +0.005682 |
| 62 | 0.318182 | 0.284091 | -0.034091 |
| 63 | 0.306818 | 0.306818 | +0.000000 |
| 64 | 0.301136 | 0.357955 | +0.056818 |
| 65 | 0.301136 | 0.306818 | +0.005682 |
| 66 | 0.312500 | 0.318182 | +0.005682 |

| Metric | control mean | candidate mean | paired delta (95% paired-t CI) |
| --- | ---: | ---: | ---: |
| Telugu Belebele accuracy (176 held-out rows) | 0.310369 | 0.318182 | +0.007813 [-0.019408, +0.035033] |

For context, the pre-existing Qwen2.5 corrected-SFT run before this Telugu
task adaptation scored 0.255682 on the identical 176 rows. Thus the Telugu
task injection itself has a material effect, but the tested energy+dual-KL
mechanism is only directionally positive relative to its matched adaptation
control: 5 positive, 2 negative, and 1 tied seed (two-sided sign-test p=.453).
The confidence interval is broad and crosses zero.

The pre-specified decision is therefore to **not** spend another long full
English IFEval sweep on this arm: there is no reliable target-task effect to
trade against retention. The useful conclusion is negative but informative:
the final mechanism's CE and Qwen3 strict-IFE benefits do not automatically
transfer to this low-data, long-context, cross-script reading-comprehension
injection task. A next multilingual experiment should use the intended broad
Telugu instruction corpus once it is retrievable, and keep the 2048-token
filtering/token-budget issue explicit.
