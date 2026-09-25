# Output-only equal-budget IFEval diagnostics

Scope: `final_exact_eight_runs`. Validation passed for 8 runs × 541 canonical paired prompts, including both strict and loose official score files.

## Response length and cutoff signals

The current response artifacts do **not** contain generated-token counts or finish reasons. Empty responses are directly observable; nonterminal endings and long nonterminal endings are only heuristics, not proof that the 512-token cap was reached.

| Seed | Method | Mean chars | Median chars | P95 chars | Mean whitespace units | Empty | Nonterminal | Long-P90 nonterminal | Replacement-char responses |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | `aff_r50` | 1651.2 | 1666.0 | 2561.0 | 239.2 | 0 | 451 | 52 | 18 |
| 42 | `vocab_r1` | 1716.7 | 1854.0 | 2563.0 | 282.1 | 0 | 405 | 51 | 2 |
| 43 | `aff_r50` | 1621.8 | 1699.0 | 2547.0 | 241.2 | 0 | 448 | 49 | 62 |
| 43 | `vocab_r1` | 1441.6 | 1637.0 | 2563.0 | 234.3 | 0 | 277 | 50 | 4 |
| 44 | `aff_r50` | 1780.8 | 1774.0 | 3071.0 | 220.0 | 0 | 468 | 52 | 44 |
| 44 | `vocab_r1` | 1313.1 | 1321.0 | 2453.0 | 215.2 | 0 | 295 | 47 | 9 |
| 45 | `aff_r50` | 1696.2 | 1739.0 | 2656.0 | 239.1 | 0 | 435 | 53 | 64 |
| 45 | `vocab_r1` | 993.2 | 753.0 | 2313.0 | 156.9 | 0 | 165 | 41 | 5 |

### Paired response-length differences

| Seed | Mean char A−V | Median char A−V | A longer / Vocab longer / equal | Mean whitespace-unit A−V |
|---:|---:|---:|---:|---:|
| 42 | -65.5 | -61.0 | 246 / 293 / 2 | -42.9 |
| 43 | +180.2 | +124.0 | 300 / 241 / 0 | +6.9 |
| 44 | +467.7 | +396.0 | 376 / 165 / 0 | +4.8 |
| 45 | +702.9 | +735.0 | 432 / 109 / 0 | +82.2 |

## Paired prompt directions

These are prompt-level pairs within each trained seed. Seed 42 is selected-seed description; seeds 43–45 are the independent training-seed group.

| Seed | Mode | A correct | Vocab correct | A−V (pp) | Both / A-only / Vocab-only / neither |
|---:|---|---:|---:|---:|---:|
| 42 | strict | 123 | 112 | +2.03 | 75 / 48 / 37 / 381 |
| 42 | loose | 138 | 128 | +1.85 | 82 / 56 / 46 / 357 |
| 43 | strict | 126 | 117 | +1.66 | 75 / 51 / 42 / 373 |
| 43 | loose | 136 | 133 | +0.55 | 81 / 55 / 52 / 353 |
| 44 | strict | 113 | 113 | +0.00 | 72 / 41 / 41 / 387 |
| 44 | loose | 126 | 124 | +0.37 | 77 / 49 / 47 / 368 |
| 45 | strict | 117 | 124 | -1.29 | 71 / 46 / 53 / 371 |
| 45 | loose | 134 | 141 | -1.29 | 85 / 49 / 56 / 351 |

Independent-seed descriptive mean directions:

- strict: mean A−V +0.12 pp; A better / tie / Vocab better seeds = 1 / 1 / 1.
- loose: mean A−V -0.12 pp; A better / tie / Vocab better seeds = 2 / 0 / 1.

## Strict instruction-family results

Family means the prefix before `:` in each IFEval instruction id. Counts below are paired instruction checks nested within prompts and seeds. They are descriptive diagnostics—not independent-seed significance tests.

### Available independently trained seed(s): 43, 44, 45

| Family | Unique slots | Obs. | A success | Vocab success | A−V (pp) | A-only / V-only | Per-seed A−V (pp) |
|---|---:|---:|---:|---:|---:|---:|---|
| `change_case` | 89 | 267 | 39 (14.61%) | 41 (15.36%) | -0.75 | 13 / 15 | 43:-1.12, 44:-2.25, 45:+1.12 |
| `combination` | 65 | 195 | 28 (14.36%) | 7 (3.59%) | +10.77 | 25 / 4 | 43:+9.23, 44:+10.77, 45:+12.31 |
| `detectable_content` | 53 | 159 | 100 (62.89%) | 98 (61.64%) | +1.26 | 28 / 26 | 43:-5.66, 44:+18.87, 45:-9.43 |
| `detectable_format` | 157 | 471 | 245 (52.02%) | 237 (50.32%) | +1.70 | 60 / 52 | 43:+5.73, 44:+1.27, 45:-1.91 |
| `keywords` | 163 | 489 | 225 (46.01%) | 239 (48.88%) | -2.86 | 62 / 76 | 43:-7.98, 44:+3.68, 45:-4.29 |
| `language` | 31 | 93 | 52 (55.91%) | 45 (48.39%) | +7.53 | 22 / 15 | 43:+25.81, 44:-6.45, 45:+3.23 |
| `length_constraints` | 143 | 429 | 119 (27.74%) | 134 (31.24%) | -3.50 | 47 / 62 | 43:-3.50, 44:-4.90, 45:-2.10 |
| `punctuation` | 66 | 198 | 39 (19.70%) | 29 (14.65%) | +5.05 | 23 / 13 | 43:+6.06, 44:+10.61, 45:-1.52 |
| `startend` | 67 | 201 | 1 (0.50%) | 26 (12.94%) | -12.44 | 1 / 26 | 43:-4.48, 44:-8.96, 45:-23.88 |

### Selected seed 42 (reported separately)

| Family | Unique slots | Obs. | A success | Vocab success | A−V (pp) | A-only / V-only | Per-seed A−V (pp) |
|---|---:|---:|---:|---:|---:|---:|---|
| `change_case` | 89 | 89 | 13 (14.61%) | 11 (12.36%) | +2.25 | 6 / 4 | 42:+2.25 |
| `combination` | 65 | 65 | 4 (6.15%) | 5 (7.69%) | -1.54 | 2 / 3 | 42:-1.54 |
| `detectable_content` | 53 | 53 | 38 (71.70%) | 36 (67.92%) | +3.77 | 12 / 10 | 42:+3.77 |
| `detectable_format` | 157 | 157 | 88 (56.05%) | 78 (49.68%) | +6.37 | 27 / 17 | 42:+6.37 |
| `keywords` | 163 | 163 | 74 (45.40%) | 80 (49.08%) | -3.68 | 19 / 25 | 42:-3.68 |
| `language` | 31 | 31 | 16 (51.61%) | 18 (58.06%) | -6.45 | 4 / 6 | 42:-6.45 |
| `length_constraints` | 143 | 143 | 43 (30.07%) | 44 (30.77%) | -0.70 | 15 / 16 | 42:-0.70 |
| `punctuation` | 66 | 66 | 13 (19.70%) | 5 (7.58%) | +12.12 | 10 / 2 | 42:+12.12 |
| `startend` | 67 | 67 | 0 (0.00%) | 1 (1.49%) | -1.49 | 0 / 1 | 42:-1.49 |

## Strict directional prompt examples

### Available independently trained seed(s): 43, 44, 45

Prompt examples use strict `follow_all_instructions`. Excerpts are capped at 160 characters; responses are not copied.

#### A-only representative prompts

| Key | Directional seeds | Opposite seeds | Families | Prompt excerpt |
|---:|---|---|---|---|
| 1098 | 43,44,45 | — | `combination` | What is a name that people call God? Please give exactly two different responses. Separate the responses with 6 asterisk symbols: ******. |
| 2667 | 43,44,45 | — | `detectable_format` | Please elaborate on the following text: "It's not a bug, it's a feature!" Write exactly 2 bullet points in markdown format. Use "*" to indicate a bullet point.… |
| 3241 | 43,44,45 | — | `language` | Rewrite and expand the following in Arabic. You can hallucinate a lot of details. "The company is looking to expand its operations into new markets. It will cr… |
| 1477 | 43,45 | — | `language` | Write a weird poem about yoda being transported into a different universe in the Persian language, no other language is allowed. |
| 1518 | 43,44 | — | `combination` | I am a software engineer with 7 years of experience, and I am looking for a new job. Can you create a resume for me and explain each section? First repeat the… |
| 1793 | 44,45 | — | `combination`, `length_constraints` | Who won the defamation case between Amber Heard and Johnny Depp? Write your answer as if you are writing to a group of elderly people. First, write in the pers… |

#### Vocab-only representative prompts

| Key | Directional seeds | Opposite seeds | Families | Prompt excerpt |
|---:|---|---|---|---|
| 1094 | 43,44,45 | — | `detectable_format` | Please provide the names of 5 famous moms in JSON format. Please, use any interesting or weird tone. Your entire output should just contain a JSON block, nothi… |
| 2243 | 43,44,45 | — | `length_constraints` | Who built the first artificial ice rink? Please include the keys (1) Name (2) Location and (3) Year. Use less than 150 words. |
| 1162 | 44,45 | — | `punctuation` | Write a riddle for kids about auspices but make sure you don't use any commas. |
| 1217 | 44,45 | — | `length_constraints` | Write an ad copy for a new product, a digital photo frame that connects to your social media accounts and displays your photos. Respond with at most 150 words. |
| 127 | 43,44 | — | `keywords` | Take the text below as a starting point, and make it a complete article: "You may have to meet with a helper to work out a parenting plan. The first would be t… |
| 1379 | 44,45 | — | `detectable_format`, `keywords` | Write a limerick about a woman named Sarah who lives in a town where it's always 90°F. Highlight at least 6 sections in your answer with markdown, example: *hi… |

Directional prompt-event family counts are multi-label (a multi-constraint prompt may contribute to several families): `{"aff_only": {"change_case": 9, "combination": 14, "detectable_content": 12, "detectable_format": 44, "keywords": 32, "language": 16, "length_constraints": 27, "punctuation": 10, "startend": 1}, "vocab_only": {"change_case": 9, "combination": 3, "detectable_content": 19, "detectable_format": 38, "keywords": 38, "language": 8, "length_constraints": 29, "punctuation": 6, "startend": 18}}`.

### Selected seed 42

Prompt examples use strict `follow_all_instructions`. Excerpts are capped at 160 characters; responses are not copied.

#### A-only representative prompts

| Key | Directional seeds | Opposite seeds | Families | Prompt excerpt |
|---:|---|---|---|---|
| 1187 | 42 | — | `punctuation` | Can you give me two different formal alternatives to "What's up? I'm going to the beach today" and do not use any commas in your response. |
| 1268 | 42 | — | `length_constraints` | Can you give me an example for a journal entry about stress management? Tell me how you come up with the example. Your entire response should contain less than… |
| 1322 | 42 | — | `detectable_format` | What are the pros and cons of kotlin vs java? Your answer must have a title contained in double angular brackets, such as <<kotlin vs java>>. |
| 1367 | 42 | — | `detectable_content` | Write an extravagant session plan to learn about java. Make sure to include a postscript starting with P.P.S |
| 1379 | 42 | — | `detectable_format`, `keywords` | Write a limerick about a woman named Sarah who lives in a town where it's always 90°F. Highlight at least 6 sections in your answer with markdown, example: *hi… |
| 1381 | 42 | — | `length_constraints` | Write me a funny song with less than 10 sentences for a proposal to build a new playground at my local elementary school. |

#### Vocab-only representative prompts

| Key | Directional seeds | Opposite seeds | Families | Prompt excerpt |
|---:|---|---|---|---|
| 102 | 42 | — | `detectable_format` | Write a dialogue between two people, one is dressed up in a ball gown and the other is dressed down in sweats. The two are going to a nightly event. Your answe… |
| 1072 | 42 | — | `length_constraints` | Write a blog post with 400 or more words about the benefits of sleeping in a hammock. |
| 1203 | 42 | — | `keywords` | What happened when the Tang dynasty of China was in power? Make sure to use the word war at least 8 times, and the word peace at least 10 times. |
| 1217 | 42 | — | `length_constraints` | Write an ad copy for a new product, a digital photo frame that connects to your social media accounts and displays your photos. Respond with at most 150 words. |
| 1251 | 42 | — | `length_constraints` | Write a poem that's at least 350 words about the beauty of eucalyptus trees and their many uses. |
| 1262 | 42 | — | `detectable_format`, `length_constraints` | Can you elaborate on the following text: "Increased axle weight increased the damage to the road"? Your response must contain a title wrapped in double angular… |

Directional prompt-event family counts are multi-label (a multi-constraint prompt may contribute to several families): `{"aff_only": {"change_case": 4, "combination": 1, "detectable_content": 8, "detectable_format": 15, "keywords": 10, "language": 4, "length_constraints": 8, "punctuation": 4}, "vocab_only": {"change_case": 2, "combination": 2, "detectable_content": 5, "detectable_format": 13, "keywords": 7, "language": 5, "length_constraints": 9, "punctuation": 2, "startend": 1}}`.
