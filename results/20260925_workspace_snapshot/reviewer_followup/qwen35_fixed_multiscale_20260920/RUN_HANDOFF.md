ALL FOUR SIZES READY. Preparation57502 finished0. Active scheduler39069,idle closeout13968,storage guard87623. Seven eligible GPUs in use. Four technical gates passed16 reference batches and6 independent-device cases each. Old partial and full studies untouched.

LATEST CHECK2026-09-20T16:33:50+08:00: 271 passed(231 formal+40 smoke),zero failures. Passed counts={'08b': 82, '2b': 78, '4b': 66, '9b': 45};phases={'08b': 'confirmation', '2b': 'search', '4b': 'search', '9b': 'search'}. The0.8B completed72 search fits at16:20:57;its first confirmation WikiSQL H seed7700 is now running. The2B and4B are on second development seed7601;9B has just its final TREC50 HEU fit remaining for seed7600. All three sessions39069/13968/87623 polled alive at16:33. Storage guard watching,admission_paused=False;free118.80GiB. Home filesystem288.44GiB free at15:05;no cleanup or relocation. No frozen study changes. IMPORTANT new user steering:inspect and explain results at completed stages,not just job counts. Stage review helper is active in each manual polling cycle;initial11 complete-grid milestones verified,0.8B SQL positive/TREC negative developmental results reported. See stage_review/LATEST_ZH.md and appended steering section. No scope cancellation or protocol change. Updated18-22h remaining ETA given16:22;continue full queue unless user changes scope.

Operational check at2026-09-20 04:00+08:00: all four sizes in search,89 jobs passed(40 smokes+49 formal fits),zero failures. Both live sessions polled and still running. Shared filesystem free space observed377GiB at02:59,368GiB at03:30,359GiB at04:00. Four study roots together used about7.8GiB at03:30; most shared-volume growth is outside these roots. Continue monitoring capacity; no cleanup performed and currently ample space for this study's remaining adapters. User previously permits deleting Gemma model files only if space is insufficient; do not delete archived/sealed evidence or other workloads.

Timing-only estimate at04:48:59:101 jobs passed(61 formal+40 smoke); all four sizes have completed all three first-LR WikiSQL arms, and9B has its first full TREC50 result. Median development fit minutes(WikiSQL/TREC50):0.8B22.8/16.0,2B23.6/18.4,4B32.1/34.0,9B35.9/51.7. Remaining estimate206.5 GPU-hours,29.5 hours at ideal seven-worker utilization, scaling only evaluation duration by confirmation/dev example counts and deducting active elapsed time. Excludes GPU eligibility/queue constraints and final audit overhead;9B TREC estimate uses only one completed fit, output length/selected LR/other workloads can change duration. User was told roughly30 more hours and continued monitoring. No task-accuracy values inspected or used. Shared disk at04:30 had350GiB free;home and/tmp are a separate filesystem with289GiB free. No storage intervention performed.

05:00 check:102 jobs passed,zero failures,all four still in search. Both sessions39069/13968 polled live again. Shared disk341GiB free;continue half-hour capacity checks. Read-only inventory found models/Gemma-2-2B-Base9.8GiB,Gemma-2-9B-Base35GiB,Gemma-3-1B-Base1.9GiB,all ordinary directories(no symlinks). Nothing deleted or relocated. No new source/protocol changes;only this operational handoff updated while monitoring.

# Active four-size corrected-policy run

Do not end monitoring while the user's authorized experiments are active.
Do not edit a size root after CODE_FROZEN.json exists. Never edit sealed studies.
Always set PYTHONDONTWRITEBYTECODE=1 before importing archived/sealed code.

Active tool sessions: scheduler39069 (replaces empty waiting session95605 before any freeze/fit), idle closeout13968. Preparation57502 completed successfully and was polled.
Dispatcher: ../schedule_qwen35_fixed_20260920.py, SHA pinned in each size root.
Technical preparation: ../prepare_qwen35_fixed_20260920.py.
Status: ../status_qwen35_fixed_20260920.py (read only; no partial effect claims).
Old sessions82482/16704 terminated only after all old workers finished normally.
Old partial archive session91026 finished successfully. All41 old runs audited.

Four roots: qwen35_fixed_08b_20260920, qwen35_fixed_2b_20260920,
qwen35_fixed_4b_20260920, qwen35_fixed_9b_20260920.
114 jobs/size,456 total,16 primary contrasts, same six LRs,2 development seeds,
5 confirmation seeds as old design. Independent fresh reruns; old results retained.

Preparation GPUs7/1/2/3 correspond to0.8B/2B/4B/9B. GPU4 is a shared serialized
verification device. Scheduler reserves4 until all sizes are READY, and reserves
each preparation GPU until its size READY. Formal one worker/GPU requires32/40/56/68GiB free for0.8B/2B/4B/9B. Pre-admission resource revision archived in PRE_ADMISSION_RESOURCE_REVISION.
Other workloads are not ours. Do not terminate or change them.

Numerical finding: single native FLA l2norm forward config change exactly reproduced
all16 microbatch losses of the old0.8B outlier. See ../qwen35_numerics_20260920.
Original selected config was not recorded; no causal claim about final accuracy.
Prototype fixed policy passed8-step full gradient/parameter equality across GPU5/6
for both exact-budget H and HEU on0.8B WikiSQL. All four model technical gates passed; formal training and per-job audits are ongoing.

Policy: process-local12 pinned official FLA configs, deterministic Torch/cuDNN,
CUBLAS_WORKSPACE_CONFIG=:4096:8, TF32 and reduced-precision GEMM reductions disabled.
Each new root has frozen fixed_runtime.py; no installed package mutations.
16 training-only reference batches per size (8 seeds x2 tasks), independent process
and device verification of all arms atseed7600, each worker exact16-micro-loss
check before any optimizer update plus exact local gradient repetition.
Does not prove complete cross-device training determinism.

If a gate fails, preserve artifacts. No score-based retries, no selective replacement.
Pre-freeze technical edits must be documented and their affected gates rerun;
post-freeze policy changes require new roots, never mutate frozen sources.
CONTROL.json pause_admission pauses admissions while draining workers normally.

AFTER all closeout seals: run ../compare_qwen35_policy_20260920.py for a source-verified descriptive old/new4B comparison (no new tests or selection). It requires both studies sealed and checks all matched initializations/orders; outputs outside sealed roots. Investigate any identity mismatch before interpreting differences.

Timing-only update at2026-09-20T16:22:14+08:00:268 passed,all search observations used (no accuracy values). Remaining estimates in GPU-hours:0.8B14.15,2B17.07,4B34.02,9B62.49,total127.73,ideal seven-worker walltime18.25h. Median development wall/evaluation minutes by size and task WikiSQL/TREC50:0.8B22.60/16.06 wall,11.25/4.66 eval;2B23.37/18.20 wall,11.91/6.86 eval;4B31.41/34.00 wall,16.29/16.07 eval;9B36.23/51.78 wall,20.43/26.84 eval. Confirmation prediction retains training/load and multiplies only dev evaluation by2 for WikiSQL and500/256 for TREC50;15 fits/task/model plus one Base/task. Base estimate subtracts median TRAINING.seconds from dev wall then scales evaluation. Active-search remaining uses evaluation fraction if available,otherwise bounded elapsed subtraction;auditing allowances0.25min. No confirmation worker was active at the snapshot. Prediction does not model selected-LR output lengths, precise fair-queue/GPU eligibility, foreign loads, storage intervention or final audit runtime. User given approximate18-22 more hours (Sept21 roughly10:00-14:00 Asia/Shanghai),not a deadline or guarantee. Continue full frozen queue without truncation. Only timing fields were used;one training-progress loss was incidentally printed while inspecting schema,not used for selection or effect inference.

USER STEERING2026-09-20 around16:26:User questioned blind large-scale running,then clarified they want results reviewed and reported at each completed validation stage. They did NOT cancel or authorize cutting the matrix;no pause/control,stopping,rescheduling,protocol or scientific-source changes made. Continue existing queue AND inspect/report phase effects. Do not keep only reporting run counts. At16:30,created parent review_qwen35_stages_20260920.py (outside frozen roots). Invoke it each monitoring cycle;it verifies passed-job audit status,SUMMARY SHA,example count,finite primary,full-selection input hashes;generates master stage_review/LATEST.json,LATEST_ZH.md,and one immutable milestone JSON per completed task/seed full grid and paired confirmation-prefix1/3/5. No intermediate significance tests or result-dependent changes. Uses all six candidates in fixed tie order;single-seed maxima explicitly developmental/descriptive. No installed/frozen code edits. Initial11 milestones generated successfully. Stage results are NOW inspected at explicit user request (earlier no-score statements are historical).

Stage findings reported to user:0.8B full72 search fits,selected two-dev-seed means WikiSQL H77.34375/HB77.05078125/HEU78.173828125 (+0.830078/+1.123047);TREC H83.3984375/HB83.59375/HEU81.4453125 (-1.953125/-2.148438). WikiSQL all select8e-4;TREC H/HB8e-4,HEU4e-4. All144 SELECTION input hashes verified. Single-dev-seed7600 complete-grid descriptive best-per-arm deltas HEU-H/HEU-HB:2B WikiSQL-0.976563/+0.292969,TREC-1.5625/+1.5625;4B WikiSQL+0.78125/+0.390625,TREC+1.953125/0;9B WikiSQL-0.976563/-0.976563;9B TREC seed7600 final HEU still running. These are NOT confirmation outcomes and not causal numerical-policy attribution. Older strongest controlled positive and negative evidence remains unchanged. User told previously blanket four-size expansion lacked timely stage interpretation;correct monitoring now,not effect-driven cherry-picking.

Status16:30:45:0.8B confirmation82passed,first WikiSQL H seed7700 active GPU7 step17;2B78passed,4B66passed,9B45passed;all zero failures. Total271=231formal+40smoke. All three controller/watcher/guard sessions still expected live(last polled15:34);capacity last125GiB16:04 then120.21GiB16:23,unpaused. No cleanup. Read effect milestones after each phase;retain ongoing monitoring until required queue completion unless user changes scope.

USER DIAGNOSTIC REVIEW2026-09-20T16:54+08:00:User explicitly asked to re-examine results and whether equal-budget regression indicates training/configuration problems. New outside-frozen helper analyze_qwen35_budget_review_20260920.py produces qwen35_budget_review_20260920/{ANALYSIS.json,REPORT_ZH.md,training_instability.png,pdf};figure visually checked. This is a posthoc developmental diagnosis,not a new fit or protocol change. At snapshot30 complete two-seed/same-LR/all-three-arm groups:HB>H13,of whichHEU>HB8;HB=H3,HEU>HB3;HB<H14,HEU>HB10 butHEU>bothonly7. Groups reuse seeds/data;no correlation/significance/causal inference.
Key2BWikiSQL high-LR8e-4 results seed7600 H82.32421875/HB79.1015625/HEU80.2734375;seed7601 H82.2265625/HB82.03125/HEU31.34765625. Thus HB regression and HEU severe instability concentrate in DIFFERENT seeds. HEU7601 step4 loss4.771725,gradnorm464.553;step5 loss7.336282,64/64clipped;4e-4sameHEUseed81.4453125. H/HB/HEU firstlosses exactlysamewithinseed;4e-4vs8e-4samearm fullinitialization/orderverifiedequal;sharedHinitidenticalacrossarms;basefrozen,optimizerwhitelistpassed. No proven implementationbug or full-trajectory determinism claim;highLRconfiguration sensitivity supported,not isolated causal mechanism. Selectedtwo-seed2BWikiSQL H8e-4 mean82.27539,HB/HEU4e-4both81.39648;do NOT attribute selected4e-4no-gain directlyto unselected8e-4collapse. HBmean tail8loss at4e-4 .068363 vsH.068079;8e-4 .064502 vs.063696—small,notgeneraloptimizationfailureproof. Online changing-minibatch losses cannot establish train/dev gap. Formatting doesnotexplainHB7600drop(queryvalidimproves)orHEU7601largecollapse(94.82%queryvalid,98.14%JSON,no truncation).
2Bbudgetallocationread:133120extraactiveparams,14fusedGDNQKV +3fullattentionQ projections rank8->9,commonscaling2/initpreserved. Exactcorrectbutnotvalidatedbestallocation. Proposed future bounded diagnostic directions:alternativeexact-budgetallocation,independentboundaryLR/warmup,andfixedcheckpointcommontraining-subsetevaluation. These new diagnostics were NOT launched;current frozen queue unchanged and admissionsunpaused. Lastguard16:52 free111.39GiB,watching;checkfrequentlynear100GiBreserve. NoGemmadeletionormodelrelocationyet. All3sessionslastpolled16:33. Stagehelper continues13milestones;first0.8confirmHmayfinishsoon,needcompletepairedarmsbeforeeffectreport.
