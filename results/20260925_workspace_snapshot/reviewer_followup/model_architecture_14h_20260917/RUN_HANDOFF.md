# Completed scientific run — chronological handoff history

**Final readiness, 2026-09-18:** Scheduler47140 exited0 after all admitted workers/auditors drained at09:50:18. No experiment process remains active. FINAL_AUDIT passed; 377 new fits,18 Base evaluations,32 smokes,104 reused rows;125 formal jobs not admitted. Coverage: core main16/16, core nine-arm three-seed14/16, additional untied five-seed0/4, new-task full three-seed9/16. Final report, diagnostics, CSVs, four visually reviewed scientific figures and three global evidence documents are complete. Seal/verification provenance is in SEAL.json and the external verification output; once SEAL exists, this directory must stay read-only. Earlier “active/continue” instructions below are historical records, not instructions to restart jobs or exceed the original deadline.

## Historical record begins: 2026-09-17 21:41 Asia/Shanghai

User explicitly requested as many model-series, downstream-task and architecture experiments as possible within14 hours, including untied H+E/H+U and standalone E/U/E+U. Continue autonomously; do not end after launch. Deadline **2026-09-18 11:07:10 Asia/Shanghai (03:07:10 UTC)**. No goals created; no subagents authorized or spawned. Seven eligible GPUs are used; do not touch other jobs. All old archives remain immutable.

Workspace /commondocument/wz/cross_encoder_workspace/im_exp. Python /home/wz/anaconda3/envs/torch24/bin/python. rg is NOW available on PATH (unlike earlier summary). No AGENTS found. Unrestricted filesystem, approval never; never set sandbox_permissions. Chinese concise progress updates; aim <=60 seconds between updates, sleeps <=45 seconds. User prefers continued polling.

## Active process

**Scheduler unified exec session16335**, started~21:21, stdout redirected to scheduler.log. Its source and the initial resource policy are hashed in SCHEDULER_SOURCE.json. It registers ready frozen phases core/ and new_tasks/, maintains STATE.json, runs at most one own worker/GPU, uses pool1–7 with >=48GiB free at launch; GPU0 is occupied by another large job. Each successful worker releases GPU then undergoes independent CPU audit (two simultaneous auditors). New-phase registration is supported and occurred at21:34:57. No failures or halted phases so far.

Latest state21:41: **17 passed (16 core smoke+1 formal),7 active,528 pending,halted[]**. All16 core smokes passed. Current 7B CLUENER seed6100 newly fitted arms H+E, H+U, E, U, E+U have trained64steps and test480–704/1343. Qwen3-8B CLUENER Base test32/1343. Qwen2.5-0.5B Base just evaluated1343/1343. First new0.5B E+U seed6100 finished/audited; its score has NOT been inspected for scheduling. New task smokes are queued at priority0 and will take freed cards.

functions.store keys: v6dir=this absolute root; v6m=exec_command args for Python monitor.py (read-only). Usage text((await tools.exec_command(load('v6m'))).output). Store may survive compaction. Read tail scheduler.log for events/errors. Do not rerun scheduler.py: it asserts no existing STATE. No automatic retry/resume; inspect failures. No need to wait on already completed sessions listed below.

WINDOW.json: stop new admission at09:37:10 (12.5h after start), stop active workers by10:07:10, reserve final hour for auditing/reports/seal. Fixed per-job admission allowances require enough time before worker cutoff (large core NER/QA75min, medium50, small30; SQL/class40/30/20;smoke20). No performance-based selection. Per-job allowance is admission-only; hard kill happens only at worker deadline. Scheduler emits all failures/incomplete records. It may finish earlier if all552 planned new jobs complete.

## Scientific phases (FROZEN, DO NOT EDIT)

core/ CODE_FROZEN SHA a4328b4097bf16e4161a7e294659194b41e396151d874c43abe9f11cecf32a6e
core/ DATA_FROZEN SHA 0711f2dc518def451937e6aa1f6e71b4eb7a695c0909f6c68a193ecd400fb7ea
new_tasks/ CODE_FROZEN SHA 71f362508f384e95986e5bd271cad6b038b1fe1d47e2cdee5da620cdf8f176ae
new_tasks/ DATA_FROZEN SHA 8fadf5cbc5c8c9dc9efa698adf88e250c3901e3c080a5bce84f07f7916a66d80

Read PLAN.md, core/PROTOCOL.md, new_tasks/PROTOCOL.md. Scientific fitting/scoring/statistics scripts, jobs, model configs and data are frozen. Reports/plots/additional audits can be added separately. Do not mutate old V5 or earlier archives.

Models: Qwen2.5 Base0.5B/1.5B/3B/7B; Qwen3 Base0.6B/1.7B/4B/8B. All official revisions/files verified. New .5B/1.7B/4B/8B raw configs/tokenizers fetched at pinned revisions; local weights matched official LFS hashes then independently copied/rehashed. Older4 model configs and weights rehashed before reuse. models.json has exact identities. Actual Qwen2.5-7B and Qwen3-8B untied; all other6 tied. All nativeEOS151643, noChatML. Other architecture differences prevent pure-size causality.

Core tasks exact old CLUENER2048train/200dev/1343test, micro2, generation512, eval8dev/32test; WikiSQL2048/256/1024,micro4,cap256,eval16. Full9 cells: Base,H,budgetH,H+E,H+U,H+E+U,E-only,U-only,E+U(noH). H:r8alpha16drop.05 allq/k/v/o/up/down/gate. E/U:r16alpha128scale8drop0;Ehasbias,Udoesnot; separate branches, no tied-merge claim. Parameter counts E33*d,U32*d,both65*d. Budget expands q/k ranks with original r8 init preserved and scaling2. Q25-7B budget has+512; all exact/excess cases in BUDGET_PLAN, not assumed exact.

3 paired seeds for breadth (6100..6102 NER,7100..7102SQL). Untied2 models have pre-planned extension to5 seeds. Report common3 seeds, then5 only if all required arms available. No comparing5-seed averages against3. Core intended464 formal cells,104 verified reused in planned matrix,**360 new jobs (352fits+8Base)**,16smokes. prepare.py accepted148 historical runs overall, re-decoded/re-scored178348 responses; selected104 for new matrix. Reuse audits in provenance/ and ANCHOR_INDEX. Initial/final adapters rehashed but never initialize a new run. Old standalone E/U existed on small CLUENER models and are reused if matched.

Core analyze.py frozen:11 contrasts per16 conditions=176 family. H+EU-H,H+EU-budget,H+E-H,H+U-H,H+EU-HE,H+EU-HU,E-Base,U-Base,EU-Base,EU-E,EU-U. Reports means/SD/raw differences/directions, descriptive unadjusted95% t and Bonf176;df2 for3seeds,df4 for5. Family denominator stays176 even if budget leaves missing rows. Only complete9-cell common-seed blocks summarized; no survivor selection. This is exploratory, not new broad confirmation.

New tasks: **TREC50** author train5452/test500 (latin1). Full500 test kept; normalized duplicate/test-overlap training removed; dev256/train2048 selected via deterministic stratified round-robin, all50 labels in train. Prompt lists fixed descriptions+two-digit codes00–49. Primary exact joint likelihood of both code tokens; candidate prefix-cache verified with5 independent full-prefix forwards at shortest/longest evaluated inputs for each checkpoint. Eval8,micro2,seeds9100..9102; not free generation.

**SQuAD2** official author train/dev + official evaluation script downloaded. Eligibility before outputs: reference prompt<=768 tokens, allgoldserializedtarget+EOS<=128; selected rows checked withall8tokenizers, no truncation. Only38train/32officialdevrows excluded for prompt length; no target cap excludes. Internaldev bywholearticlehash; selected train2048/dev256/test1024 each exact50/50 answerable/unanswerable; article/context disjoint (405/29/35 articles). Test is fixed public-dev subset, nothiddenbenchmarktest. Prompt JSON{answer:string}, emptystring only forunanswerable, usecontextonly. Primary official normalized multi-ref tokenF1;EM/answerable-only/unanswerable-only/schema/strictJSON/rawemptyanswerability/extractiveness separate. InvalidJSON always0, never rewarded asabstention. GreedyFP32cap128eval8,micro2,seeds10100..10102. Constant validempty scores50overall, so reportanswerable-only.

New tasks all8models, Base/H/budgetH/H+EU,3seeds:160formaljobs (144fits+16Base),16smokes. No reused fitted predictions fornewtasks. Total newroot queue552jobs=520formal+32smoke. Testscorer4575goldroundtrips+16640officialperturbations passed. Independentauthorrecord6132 and49056tokenrecord re-encoding auditpassed. Core55352tokenrecords prepared. Newtaskcodecompiles. Newtaskfreeze/registrationcomplete.

Allfits2048/64steps/effectivebatch32/LR2e-4/AdamW(.9,.999)eps1e-8WD0clip1/warmup2cosine. FrozenbaseBF16;trainables/optimizerFP32;BF16autocast;FP32evalTF32offSDPA. Prompt/padmasked answer+EOS tokenmean. Finalcheckpointonly. Eachrunwhitelist/zeroresidual/independentHFmaskedloss/frozenbasegradanddigest/everyrequestedgroupchanged/optimizerFP32/savezeroreloadexact checked. Independent audit_one.py afterGPUexit rescansrawoutputs/checkpoints/order andofficialSQLorQA functions. TREC candidateauditpassesexpectedargmaxcache/fullforward. Smokes longesttrainmicro forward/backward underfork_rng +2steps; no scoregate.

Newtask analyze.py frozen32contrasts=16conditions×2controls,3pairedseeddf2; unadjusted descriptive andBonf32,allplanneddenominatorretained,dev/testseparate. Primary QA F1 andTREC50accuracy.

## Completed extra audits

Parent audit_pairs.py added outside frozen phase sources. It independently checks fullinitialadapterdigests, rank8 sharedH, E/U branchinitializations, sampleorders and cross-arm/taskoriginalbaseweightdigest equality. Firstrun **112 available runs,29groups,104 reused checkpoints rehashed passed** at~21:41. Output PAIR_AUDIT_PROGRESS.json. Re-run atmeaningfulnewcoverage/final (not everyminute). Session88659 completeexit0. Readysessionsbootstrap24779/coreprepare12820/coretest30103/corefreeze14940/newfetch56834/newprepare38381/newtest56535/newauditdata41916/newfreeze71362 allcompleteexit0.

Direct checkfirst7B seed6100: H+E andH+U HinitidenticaltoreusedH; H+E changesH/Eonly,H+U changesH/Uonly;E,U,EUstandalonenointernalLoRA;allbasefrozentensorsunchanged;actualuntiedfalsebinding verified.

## Remaining work

1. Keep monitoring scheduler16335, active progress and per-run auditors through the window. Handle any factual failures safely, preserve evidence, no silent reruns or frozen-code edits. CPU prep may continue alongside runs. Do not end the turn after launch.
2. Finish parent reporting/finalization infrastructure: global end audit should verify phase CODE/DATA, allmodelhashes, allsuccessfulrun auditcounts/summaryhashes, planned-vs-completed coverage, pair audit, historicalfilemanifest links. Distinguishfailed/deadline/neverstarted. core analyze.py only produces test; auxiliary dev canbeadded as clearlydescriptive presentation without alteringfrozenprimary code. New analyze producesboth.
3. Prepare plots/tables for8modelseries, corefullarchitecture9arms inclstandalone, untied HE/HU/HEU effects and newtasks. StaticmatplotlibPNG/PDF/SVG, visuallyinspectPNGs, recordreviewhashes. Don't infercausalplacement or universalbenefit fromheterogeneousparametercounts. Keepnull/negativecases.
4. At end runfrozenanalyses, descriptivediagnostics, report fullthree/five seedtables withmissingcellsandactualcoverage. Rawstandaloneparams andBase critical to user. Showlikelihoodclassificationversusgeneratedcontentdistinction;QAanswerable/unanswerablebreakdown. Keepweakoldresults.
5. Update lora/README.md,lora/docs/RESULTS_SO_FAR.md and emnlp/iclr2027/submissions/paper-2/EXPERIMENT_EVIDENCE_STATUS.md withfinaldata only. RootREADME currentlyinprogress. PriorV5docscompleteunchanged, no newglobalentryyet.
6. Before11:07 deadline complete finalaudit/report/handoff, sealthisentirenewrootexcluding__pycache__,selfmanifest,checknoactiveprocessesorlogsbeingwritten, verifyfilemanifest independentlyread-only. Do notwriteinsideafterseal. No commit/PR/publicationrequested. Do not claimall520done iftimeomitsblocks.

## Prior sealed studies

V5 model_scaling_20260917 complete1978files,manifest adb6c2f3b26e26208b9bb3b71ecd445cfd525e01bcc7a02ad0f2168594e06ae9. 3B/7B CLUENERmeansH/budget/stack64.435/64.504/66.180 and69.189/69.219/70.466; SQL77.168/76.875/78.027 and81.484/81.445/82.129. Only3BSQLvsbudgetcorrectedCIpositive(+1.152[.242,2.063]);other7cross0. Last7BSQLseedstack−H−.09765625,tiesbudget. All90432newresponses+37872anchorsverified. Thisstudyisimmutable.
V4 downstream_nlg_intent_20260917 manifestb1618aa49342401ac96585eaf49733a88e5bd91c26b745f605f856f11648afdf;Bank/E2Epositive meansbutall8correctedCIscross0.
V3 downstream_transfer_20260917 manifestddfcf8b6c43e39303d9900efeac54d68899bf23b51bfb59426a59637ffd1fdc6;SQL.6/1.5bothcorrectedpositive,ANLIuncertain.
V2 posttraining_tasks_grounded_20260916 manifestc717690a663b6d58c70224b4b4f798e34343f1e0de49b8e8bf1df7e91d37ab5f;strong1.5BCLUENER59.17/59.24/62.57. Exact64hexV2hashabove;anearliertranscriptiontypoextradfwasfixedBEFOREcorefreezeandpreparepassed. No scientificrunfailure.

## 22:03 新增汇总工具（尚未运行最终汇总）
- plot_results.py：四张PNG/PDF/SVG图；--preview只写figure_previews，最终版要求FINAL_AUDIT。完成后必须逐图查看，再写FIGURE_REVIEW.json，字段status=passed与reviewed_pngs（四个相对PNG路径→SHA256）。
- diagnose_results.py：要求FINAL_AUDIT，写评分分解、事后算术交互，不增加统计检验。
- ARCHITECTURE_READOUT.md：实际训练对象、参数化、tied/untied解释与证据范围。
- seal_results.py：全部汇总/诊断/图审阅完成后封存；verify_seal.py独立只读核对。**两者输出不要重定向到本档案内**。封存后不得再写任何文件（pycache排除）。
- 最终顺序：scheduler结束 → finalize_audit.py → diagnose_results.py、plot_results.py、write_report.py → 审阅图和实际结论、更新README/handoff与全局文档 → seal_results.py → verify_seal.py。
- 写报告前核对实际统计字段；这些父目录工具只经py_compile和图预览，须在最终真实结果上核验。
- 22:00：39通过=32短程+7正式，7运行，506待排，无失败。7B CLUENER首个seed6100完整，H68.2656、HE69.0654、HU70.3853、HEU70.4670、E51.1432、U54.0660、EU59.9967、Base35.8227。其余种子未齐，不作普遍结论。

22:06：新增 audit_boundary_algebra.py，只取固定CLUENER seed6100已审计方案，实际模型抽取128词表行/8隐藏向量验证E/U合并公式（double与float32误差）。首轮14方案通过，max double 4.9e-15、FP32 1.9e-6；非全词表/非端到端。finalize_audit.py现已包含最后再运行；seal_results.py要求其passed。初始GPU训练代码、评分与预定统计保持未改。

22:14监控：39 passed（32smoke+7formal）/7running/506pending；所有smoke通过，均无halt。8B Base test1248/1343、HEU1024/1343，其他8B也在test。GPU0另有约66GB他人作业，1–7为我方槽位，继续轮询不要结束。
全局 lora/docs/RESULTS_SO_FAR.md 与 paper-2/EXPERIMENT_EVIDENCE_STATUS.md 顶部已加入明确“进行中”段落，最终需替换实际结论。
配对审计最新118个可用初始化/顺序，30groups，104复用checkpoint重hash，通过。
全部4张预览PNG已看过，布局可用；最终仍须真实数据重画后看4图并写FIGURE_REVIEW。摘要字段兼容性已检查111个正式可用SUMMARY，所有诊断所需字段齐全。
报告父目录工具现在均已编写，最终流程按上文执行；尚没有FINAL_AUDIT/REPORT_STATUS/SEAL，不要误认为已收尾。

## 22:39 重要：调度器已接力到V2，实验不中断
- **当前统一exec会话31032**，命令python -u handoff_scheduler.py > scheduler_v2.log；它os.exec到resource_scheduler_v2.py。原16335已按计划终止并poll关闭(exit143)，不要再poll它。
- 共享显存使GPU7只剩约45GiB，原固定48GiB线导致仅6项并行。V2按尺寸准入：.5/.6B需14GiB、1.5/1.7B需24GiB、3/4B需36GiB、7/8B保持48GiB，GPU0–7均可用，每卡我方最多1项；优先级只在符合显存的任务中保持原排序。
- 六个在跑worker保持原PID/同GPU/原stdout，无重跑无停止。旧监管进程SIGSTOP核查STATE与子PID完全对应，再SIGTERM+SIGCONT只结束监管。新监管接管。
- 接管worker不是新监管的子进程，最终exit_code保留null，completion_basis明确说明通过COMPLETE.json与新独立auditor(exit0)确认。没有把未知退出码伪造为0。新启动worker正常捕获退出码。
- RESOURCE_AMENDMENT_V2.json、STATE_HANDOFF_PRE_V2.json、HANDOFF_PROCESS_PROOF.json保存依据；SCHEDULER_V2_SOURCE.json冻结这三文件及resource_scheduler_v2.py、handoff_scheduler.py。**这些V2代码/记录现在不可改**，原SCHEDULER_SOURCE及scheduler.py/PLAN/WINDOW也未改。
- finalize_audit.py与seal_results.py已补充V2哈希核验；最终还核对全部job specs/priority/stage/phase/allowance与接力前相同、接管6个PID未重跑。README与write_report包含该资源修订说明。科学phase代码/数据/评分/统计始终未改。
- 最新46passed（32smoke+14formal）、8running、498pending，无halt。GPU0 .5B HEU seed6100、GPU7 .5B预算H seed6100，均训练至44/64；旧6项继续评测。
- 8B CLUENER seed6100已审计：Base44.19956，H70.52564，预算H71.11039，HE71.58233，HU72.49432，HEU72.35890，EU无H62.76068。E-only/U-only测试尚未完。已向用户说明HEU优于H和预算H但略低于HU，不能把“叠加有收益”写成“双侧必然最好”。

22:47：52passed（32smoke+20formal）、8running、492pending，无halt。接力的6个worker全部完成并经独立auditor exit0通过；worker_exit_code均null，completion_basis保存了真实依据。8B完整首种子E53.36183/U58.20591/EU62.76068；其余结果见22:39段。7B第二种子仅U-only还在test，8B第二种子多个架构在训练，.5B两组在test。
finalize_audit/seal进一步要求当前READY精确等于最初STATE注册的phase.ready，已检查两phase仍一致。plot_results模型系列两列现共享同任务y轴（最终重画即可）。SQuAD诊断补充说明raw-empty指标和nonempty_extractive分母。最终务必查看新任务DEV_ANALYSIS并保留任何开发/测试不一致，不用其选点。

23:09：58passed（32smoke+26formal）、8running、486pending，无halt。7B seed6101 U-only test1280/1343；8B seed6101 HEU已审计、H在训练、E刚启动，预算H/HE/HU接近评测结束。0.5B seed6100 HE/HU/HEU/预算H已通过，H在dev、E训练、U尚待。两冻结analyze.py均已在当前真实部分结果上运行成功，生成的TEST_ANALYSIS/DEV_ANALYSIS目前complete=0（还没有完整3种子块），最终需重跑。verify_seal.py现在返回verified_at及within_requested_deadline，最终以实际只读复核结束时间判定是否在14小时内。

23:11：60passed（32smoke+28formal）、8running、484pending，无halt。7B CLUENER已有完整seed6100和6101；6101分数Base35.82268/H69.14176/预算H69.00209/HE69.36231/HU70.74417/HEU70.29973/E52.24230/U53.74241/EU60.19041。第二种子HEU低于HU约0.444分，已向用户明确说明，并继续预定种子，不因该结果改变方案。8B第二种子H/E/U仍在拟合或评测，其他已有；7B第三种子已启动。此时尚无完整三种子块。
档案无symlink（已一次检查），可用现有sealer。

## 23:56 最新状态与新增诊断
- 76passed=32smoke+44formal，8running，468pending，无halt。当前exec仍31032、resource_scheduler_v2.py，输出scheduler_v2.log。7B CLUENER seed6102只剩U-only test1024/1343，首个完整3seed块即将出现；8B seed6102 H刚启动、HU训练末尾，HE/Hbudget/HEU在评测，E/U尚待。0.5B在seed6101。尚无完整3seed块。
- 8B seed6101已完整：Base44.19956/H71.09312/预算H71.03560/HE70.07861/HU71.41479/HEU71.82712/E52.10588/U58.98867/EU61.87142。HE比H低1.0145，但开发HE71.79487高于H70.75963，已坦诚告诉用户开发/测试反向，不能当作对照错配；HEU此次高于HU，与seed6100排序相反。
- 0.5B seed6100九格齐：Base0.67183/H55.96838/预算H56.14534/HE57.34019/HU56.72924/HEU57.44784/E28.01245/U21.63277/EU37.36298。小模型此seed E侧叠加贡献更大，不能概括所有模型都是U最佳。
- audit_pairs.py最近再次通过：152可用运行、33配对组、104复用checkpoint重哈希。CPU任务session60934已poll完，勿再poll。
- diagnose_results.py新增TRAINING_DIAGNOSTICS.csv：所有已审计拟合的前/后8步loss、前8步裁剪前全局gradnorm、超过1的步数比例、开发/测试分数。8步窗口是查看8B前两seed后追加的事后诊断，不能因果归因或冒充预注册检验。历史训练metadata使用本档案已冻结副本；96份副本均已再次核对与原封存文件SHA相同。已有134个拟合history字段64步已核对。历史dev分数只取存在的旧封存dev摘要，CSV记录路径/SHA，明确未在此附录重评旧dev。
- 父目录汇总工具仍只经语法/数据字段兼容检查，FINAL_AUDIT/正式报告/SEAL尚未生成。务必等scheduler完成，依次finalize_audit → diagnose_results/plot_results/write_report → 审阅四PNG及实际统计/开发反向案例、更新全局三个文档 → seal_results → verify_seal。

## 2026-09-18 00:24：首个三种子全架构块已齐
- 90 passed = 32 smoke + 58 新正式结果，8 running，454 pending，无 halt；当前监管仍为 exec session 31032。只读巡检可使用 functions.store 的 v6watchcmd，v6lastpassed / v6lastwatch 保存上次快照；分批巡检的 exec cell 505 已完成。
- 7B CLUENER 首三个种子完整，冻结 core/analyze.py 已汇总：H 68.950885、预算 H 68.890421、HE 69.399638、HU 70.265093、HEU 70.353156、E 51.542955、U 53.739802、EU 59.785854、Base 35.822684。HEU−H +1.402271、HEU−预算 +1.462735，三个种子均正，但 Bonf176 区间均跨零。HEU−HU 仅 +0.088064，逐种子方向混合。H/预算/HEU 是已核验历史复用，不能当作新独立重复。
- 当前 8B CLUENER 第三个种子只剩 U-only 在 test 480/1343 左右。H/HE/HU/HEU 的 seed6102 分数 70.242492 / 70.872224 / 71.045447 / 71.740527；最终完整统计等待 U-only。此卡还有其他工作负载，不能由墙钟时间推断模型生成质量。
- 7B WikiSQL seed7100 已通过的 H/预算/HE/HU/HEU 分数为 81.445313 / 81.054688 / 81.250000 / 81.835938 / 82.617188；EU(noH) 71.484375，Base 50.292969。E、U仍在评测，不能当作完整块。单加E略低于H的负例已向用户说明。
- 00:05 的 plot_results.py --preview 已完成，PLOT_DATA_STATUS 有49行；00:23已看更新后的 untied_side_effects.png，布局正常，最终四图仍须重新生成、逐图审阅并记录哈希。
- audit_boundary_algebra.py 增加去除共同 logit 平移后的 tied 两侧差异，ARCHITECTURE_READOUT.md 同步解释 softmax 平移不变性；这是事后局部诊断。00:24启动第一次执行更新版本（CPU session58181，须poll），最终finalizer仍会再执行。冻结科学代码/数据没有修改。

00:39更新：session58181已完成并poll关闭，25个矩阵检查通过，maxFP64 6.22e−15、maxFP32 3.70e−6，23个固定单元暂缺。
8B CLUENER三种子全部通过，core/analyze.py已重跑（2/20条件完整）。均值Base44.199559、H70.620416、预算70.866479、HE70.844389、HU71.651521、HEU71.975517、E52.986642、U58.414697、EU62.630707。HEU−H +1.355102，raw[1.833263,.734007,1.498034]，Bonf176[-17.93927,20.64948]；HEU−预算 +1.109038，raw[1.248506,.791526,1.287081]，CI[-8.33102,10.54910]。两项均3/3正但校正跨零。HE−H有seed6101负值；HEU−HU有seed6100负值，均保留。无H的EU有266240参数，Base冻结；EU−E校正CI[2.37118,16.91695]，其余本8B块校正CI均跨零。原Base schema97.99%，E/U/EU均值95.09/96.33/97.29%，不能把内容涨幅仅解释成格式率改善。U seed6102截断1.191%、平均42.10token，最终F1 58.04951，已通过审计。
预览图00:38重画成功，CPU session91635已poll关闭；最终仍须重新画四图和审阅。watcher exec cells511/513均结束。监管31032持续，最新95passed=32smoke+63新正式、8running、449pending，无halt。7B WikiSQL首种子九格已齐，standalone E62.792969、U52.929688、EU71.484375；E/U顺序与该模型CLUENER不同，不作普遍侧别优势结论。

## 01:12 重要：已采用覆盖排队修订V3
- **当前监管统一exec会话47140**，实际进程coverage_scheduler_v3.py，日志scheduler_v3.log。原V2会话31032已按计划终止并poll关闭（exit143），不要再poll或启动它。V1会话16335也早已关闭。当前functions.store v6session=47140，v6m/v6dir保持可用。
- 4小时容量复核发现，中等尺寸模型之前尚有158组待排，其中52组是untied第4/5种子。为避免窗口耗尽而没有8型号主对照，**只将40个尚未启动的Qwen3-1.7B/4B Base/H/预算H/HEU任务有效优先级25→18，将52个额外untied种子22→35**。其他排序不变，原seed/task/model/name平局顺序保留。当前顺序0smoke→10untied前三全架构→15Q25-.5全架构→18中等主对照→20首选3型号新任务→25中等其余架构→30旧tied缺失架构→35untied额外两seed→40其余新任务型号。
- 这是排队顺序的公开修订，不能继续声称原优先级始终未变。修订时已经看过部分核心结果，但规则依据模型/架构覆盖和运行耗时，按整类调整、无分数排序。所有552科学specs、原priority字段、seed、样本/训练/解码/评分/统计族、显存门槛与硬截止均保持不变，未删除准备矩阵任何单元，最终报告遗漏。新启动任务另记admission_effective_priority，不覆盖原priority。
- **八个在跑worker全部保留PID/GPU/stdout与进度，不停止、不重跑**。独立CPU审计队列为空时仅替换监管；接管退出码仍null，COMPLETE+新审计确认。第一个接管7B SQL E7101已完成并审计通过，其余继续。
- 新冻结SCHEDULER_V3_SOURCE.json覆盖6文件：coverage_scheduler_v3.py、handoff_coverage_v3.py、COVERAGE_AMENDMENT_V3.json、STATE_COVERAGE_REVIEW_V3.json、STATE_HANDOFF_PRE_V3.json、HANDOFF_PROCESS_PROOF_V3.json。**这6文件及清单不可修改**，V1/V2冻结文件均保持原样。V3manifestSHA=27fb03edda1ec644fe37f395974477d3d8e1394b37938040167a8f4bd0081475。
- 父目录新增scheduler_lineage.py（未冻结科学代码）：读校验三个版本哈希、两个接力快照、全部552 specs/原priority/phase/stage/allowance、14次接管PID记录、92改序单元在review和handoff时均pending、新启动admissionpriority符合规则。当前真实STATE验证通过。finalize_audit.py调用并把结果写FINAL_AUDIT.scheduler_lineage；seal_results.py再次校验一致。两者、helper和write_report均py_compile通过。
- README与write_report.py已经披露V3。原PLAN/WINDOW保持冻结，仍须在最终报告与全局三个文档说明实际完成覆盖。最终监管完成后正常走finalize_audit→diagnose/plot/report→审图/补真实解释/全局文档→seal→verify。
- 当前109passed（32smoke+77新正式）、8running、435pending，无halt。8B WikiSQL首种子九格已齐（尚未读取该组数值），7B第二seed尚余HU/U等。0.5B第三seed HE/HU评测中，H/E/U仍待。只读watcher exec cell518已完成；之后可按45秒间隔运行v6watchcmd，新增结果/失败/15分钟进度不动时检查。继续监控，不要结束对话。

01:37：V3接管8个worker全部完成且审计exit0，worker_exit_code均null；V2的6个也再次核验，同PID无重跑。scheduler_lineage验证全部通过；两个科学phase的READY/CODE/DATA全哈希再次通过。只读watcher532/534均已完成，当前监管仍47140。最新124passed（32smoke+92正式）、8running、420pending，无halt，磁盘余607GB。
0.5B CLUENER三种子九组全部通过，core/analyze.py已重跑（3/20完整）。均值Base0.671835、H56.598049、预算56.510715、HE57.767766、HU57.541118、HEU57.860717、E28.586885、U21.837276、EU36.930869。HEU−H +1.262669（raw1.479461/1.392927/.915618，Bonf176[-9.13644,11.66178]）；HEU−预算 +1.350003（raw1.302501/1.354850/1.392658，CI[-.200454,2.900459]）。均三正但校正跨零；本0.5块全部校正区间均跨零。HEU相对HE在seed6102为−.06629，相对HU在seed6101为−.28057，不是每seed双侧最优。E29568/U28672/EU58240参数，预算H比HEU多128，非严格等参。
8B WikiSQLseed7100完整：Base46.582031/H82.910156/预算82.8125/HE82.8125/HU82.226562/HEU83.105469/E62.109375/U61.132812/EU67.675781。HU相对H负值已向用户明确报告，不因结果改序。7B SQLseed7101完整：H81.835938/预算81.835938/HE82.910156/HU82.324219/HEU82.8125/E64.0625/U51.074219/EU70.117188，Base50.292969。此seed双侧低于HE约.09766。8Bseed7101仍未全齐；0.5 SQLseed7100 Base0、EU6.738281，其余在跑，不能外推所有standalone任务提升幅度都大。

02:03：7B WikiSQL前三种子九格齐，core/analyze已更新（4/20完整）。均值Base50.292969/H81.510417/预算81.315104/HE81.868490/HU81.868490/HEU82.421875/E62.858073/U51.399740/EU70.442708。HEU−H +.911458（raw1.171875/.976563/.585938，CI[-9.30585,11.12877]）；HEU−预算+1.106771（raw1.5625/.976563/.78125，CI[-12.81708,15.03062]）。主增量均3/3正但Bonf176跨零；H/预算/HEU是历史复用，不是新独立三次。U-only第三seed50.195313低于Base.097656，已向用户说明。EU−U均值19.042969、校正CI[2.32097,35.76497]；本SQL7B其余校正区间均跨零。
8B SQLseed7101完整：Base46.582031/H82.910156/预算82.519531/HE82.910156/HU83.984375/HEU83.59375/E59.863281/U59.765625/EU67.773438。HU高于双侧，与首seed排序相反，已公开报告。0.5 SQLseed7100 H35.9375/预算36.035156/HE37.988281/HU37.304688/HEU39.648438/E7.226563/EU6.738281，U现在已通过但暂未读取；不能外推双侧standalone总优于E。
监管仍47140（V3），watcher537/539已结束。最新141passed=32smoke+109正式、8running、403pending，无halt。8B SQL第三seed各组在训练/评测，0.5 SQL第二seed推进；Q3-1.7/4主对照尚待priority18，新的TREC/SQuAD正式任务尚待priority20。最终仍需全部审计、报告、四图审阅、三个全局文档、封存和11:07前独立复核，不能现在结束。

02:23：0.5B WikiSQL三种子九格齐，core/analyze更新（5/20完整）。均值Base0/H36.360677/预算36.490885/HE38.802083/HU36.653646/HEU38.541667/E7.03125/U1.432292/EU7.486979。HEU−H +2.180990（raw3.710938/.976563/1.855469，Bonf176[-45.62560,49.98758]）；HEU−预算 +2.050781（raw3.613281/.488281/2.050781，CI[-51.45962,55.56118]）。两主对照3/3正，但校正跨零。**HEU−HE均值−.260417，raw1.660156/−.585938/−1.855469；本组单加E平均高于双侧**，已向用户明确报告。H+U−H三seed中两个负。独立E−Base校正CI[.34245,13.72005]，其他本组校正CI均跨零；无H绝对分数很低，不能称其所有任务高质量适配。
1.7B正式Base已于02:16:56启动，H/预算/HEU首种子随后启动，admission_effective_priority=18而原priority=25，scheduler_lineage全核验通过。4B即将接续。GPU0剩余显存仅能小模型，0.5所有正式任务完成后自动按内存合格次序跑0.6B缺失架构（当前CLUENER EU seed6100），非额外改序。
最新159passed=32smoke+127正式、8running、385pending，无halt；8B WikiSQL还余第三seed H/E/U尾项。watcher542/544已结束。当前监管47140、scheduler_v3.log；functions.store另有v6schedulerlog。仍需监控到固定收尾窗并执行全部最终工作。

02:30：8B WikiSQL前三种子九格齐，core/analyze更新（6/20完整，即6/16三种子条件，五种子尚未开始）。均值Base46.582031/H83.138021/预算83.040365/HE83.072917/HU83.561198/HEU83.561198/E60.709635/U60.416667/EU67.708333。HEU−H +.423177（raw.195313/.683594/.390625，Bonf176[-7.99338,8.83973]）；HEU−预算 +.520833（raw.292969/1.074219/.195313，CI[-15.97670,17.01837]）。主增量3/3正但校正跨零。**HE−H均值−.065104，raw−.097656/0/−.097656；HEU−HU均值正好0，raw+.878906/−.390625/−.488281**，已经向用户说明在HU上再加E并无平均额外收益，不能称双侧普遍最优。
独立EU−Base +21.126302，raw21.09375/21.191406/21.09375，Bonf176 CI[19.195412,23.057192]；本8BSQL其余校正CI均跨零。这里Base查询结构有效率99.023438%，EU99.934896%，LF匹配38.476563→59.147135，执行正确率46.582031→67.708333，因此不能把全部21.13pp增量解释为修复无效查询格式。诊断脚本已包含这些冻结次指标，不需新检验。无H双侧266240参数且全base冻结，此特定配置低参数适配证据较清晰，但没有新standalone等预算内部LoRA对照。
4B Base和首个HEU已启动；1.7B首种子H/预算/HEU在dev/test，Base也在test。当前162passed=32smoke+130正式、8running、382pending，无halt，exec watcher547结束。先前只读watcher544/542均结束，监管仍V3 session47140。下一步继续监控中等模型与后续new_tasks，不能收尾。

03:00：175passed=32smoke+143正式，8running，369pending，无halt。当前V3监管47140不变；watcher551已结束。4B Base仍在test800/1343，进度持续更新，生成较慢但无报错；1.7B第三seed H/预算/HEU训练末尾或评测，4B第二seed H/预算/HEU在评测。GPU0跑0.6B NER EU seed6101（其他模型因14GiB准入不合格，按内存合格次序补priority30）。所有新任务正式任务仍待priority20，准备/32smoke已通过，勿误说已完成新任务。
1.7B NER Base22.844383；seed6100 H64.083601/预算64.289172/HEU66.505246，seed6101 H63.804071/预算64.021584/HEU66.718897；第三seed未齐。4B NER首seed H68.761076/预算69.018850/HEU69.901660（Base未审计）。两型号的正向部分结果已明确按部分结果告知，未声称显著或改变排队。0.6B NER首seed九格齐：Base.591169/H60.476190/预算60.761113/HE60.636463/HU60.947584/HEU60.887223/E28.895265/U19.255803/EU36.311322。HEU略低HU，是新增单seed负例，最终保留。
audit_pairs.py最新再次通过：**255份可用初始化/顺序记录、45组、104复用检查点重新hash**。255包含8个在跑初始化，不能作完成数。CPU session90288已poll完成退出0。磁盘余604GB。下一阶段仍需继续轮询中等尺寸主对照、new_tasks、其余架构并按固定窗口最终审计/报告/封存；结束目标时间11:07，不能现在结束。

03:13：1.7B NER前三种子主对照齐：Base22.844383/H64.113765/预算64.302343/HEU66.740263；HEU−H raw[2.421645,2.914826,2.543026]，均值+2.626499；HEU−预算raw[2.216074,2.697313,2.400374]，均值+2.437920。只是主对照的描述性均值与逐seed差值，**完整九方案尚缺分侧和standalone，冻结全块统计尚不输出该条件**；没有增加事后检验。用户已收到此范围说明。core/TEST_ANALYSIS仍最近02:27生成的6/20完整块，最终需重跑。
最新181passed=32smoke+149正式（另有1个auditing）、8running，362pending，无halt；0.6B NER第二seed EU刚完成正在审计。4B第三seed H/预算/HEU在dev/test，Base test1120/1343；1.7B WikiSQL Base与首种子主对照已开始。当前监管47140 / scheduler_v3.log，watcher556已结束。

03:28：**八型号CLUENER的Base/H/预算/HEU统一前三种子主对照均已齐**，各型号三个seed均同时高于H与预算（历史四型号主对照复用已核验，不是24个新独立复现）。全九架构并未均齐，core完整块统计仍6/20，不能混淆主对照完成和完整消融完成。按H/预算/HEU三种子均值：Q25-.5 56.598049/56.510715/57.860717；1.5 59.001686/59.054387/62.247485；3B64.484732/64.576571/66.613987；7B68.950885/68.890421/70.353156；Q3-.6 60.974759/61.143775/62.033688；1.7 64.113765/64.302343/66.740263；4B68.606527/68.911843/69.666776；8B70.620416/70.866479/71.975517。
4B Base22.478576，schema42.665674%、strictJSON93.521966%、capped6.031273%、平均68.898734tokens。Base运行较久已完成通过，未因慢而停止/重跑；不能把零样本分数简单解释为模型知识量。4B HEU−H raw[1.140584,1.425913,.614251]，平均+1.060249；HEU−预算raw[.882811,.712267,.669722]，平均+.754933。此处仍是主对照描述性结果，无新事后检验；侧别消融待priority25。
当前190passed=32smoke+158正式，另有1个auditing，8running，353pending，无halt。中等尺寸WikiSQL运行中；new_tasks正式尚未开始。watcher559结束，监管47140/V3持续。新版只读watcher用55秒间隔通知、functions.wait55000ms（仍<60秒），可以继续使用v6watchcmd/v6lastpassed/v6lastwatch。最终11:07前封存复核目标不变，不要结束。

03:51：**new_tasks正式已启动**，TREC50先跑1.5B，7B随后进入，8B待接续；SQuAD2仍按原task/seed顺序在TREC之后。1.5B TREC Base已完成、独立审计756条dev/test通过：accuracy12.6%、all50 macroF1 11.868279、coarse37.2%；缓存候选联合概率与独立5前缀完整前向在dev/test各最短最长2条检查中argmax均一致，maxerr2.10e−5/1.81e−5。不要把2条前向比对说成全756条都重做模型前向；756条是独立评分/记录审计。当前1.5B三个适配方案在训练或评测，7B Base和HEU在跑。
1.7B WikiSQL前三种子主对照齐：Base5.273438/H73.372396/预算73.404948/HEU74.902344；HEU−H raw[2.734375,.488281,1.367188]，mean+1.529948；HEU−预算raw[2.34375,.390625,1.757813]，mean+1.497396。仍只是主对照描述性结果，完整分侧/standalone尚缺，原九方案冻结检验没有改变。
父目录write_report.py新增main_arm_descriptions：最终另写CORE_MAIN_ARMS_DESCRIPTIVE.json，记录16个预定主对照条件的三seed均值/原始差值/旧拟合复用数，不加新置信区间或检验。REPORT_STATUS多字段core_main_three_seed_blocks，报告分别给主对照覆盖与九方案覆盖。当前函数在真实数据上核对15/16主对照完整（仅4B SQL尚缺第三seedH/预算）；历史四型号各任务主对照均9拟合复用，新四型号0复用。sealer要求该JSON并核对它、core3/core5、新任务覆盖数与报告一致。两脚本语法检查通过，但最终主函数须等FINAL_AUDIT后运行。
监管仍47140/V3、scheduler_v3.log；watcher562已结束。最近208passed=32smoke+176正式、8running、336pending，无halt。所有新任务正式结果不复用旧预测。继续监控；完整最终审计/绘图审阅/报告和三个全局文档更新/封存/11:07前独立验证全部尚待。

04:24：核心16/16主对照三种子齐。4B WikiSQL Base39.0625/H81.022135/预算81.445313/HEU82.324219；HEU−H raw[.683594,.390625,2.832031]，−预算[.878906,.097656,1.660156]，全新拟合。CLUENER与WikiSQL各8模型、各24个配对seed，双侧相对H和预算均为正；其中四旧模型主对照复用，不能写成全新48次独立验证，也没有新增推断检验。
Q3-0.6 NER九格三种子已齐，冻结core/analyze更新7/20完整（7/16三种子，0/4五种子）。均值HE61.383413/HU62.264025/HEU62.033688/E30.289615/U18.782818/EU37.225554。HEU−HU raw[-.060361,-.166350,-.464301]，三个均负，平均−.230337，Bonf176区间[-7.402759,6.942085]；EU−Base均值36.634384，校正区间[7.280445,65.988324]，本组其余10个校正区间均跨零。H/预算/HEU/E/U是既有结果复用，HE/HU/EU是本轮新跑。
TREC正式进展：1.5B Base12.6；seed9100 H64.4/预算62.2/HEU65.2；9101 H62.4/预算62.6/HEU72.6；9102 HEU73.8，其余待齐。7B Base30.6，9100 H87.6/预算88.4/HEU89.0。8B Base38.4，9100预算86.4/HEU85.4，H尚在评测。已向用户明确8B首seed相对预算−1pp，不能普遍化收益；其余种子仍按冻结顺序运行，SQuAD2尚未正式启动。最新227passed=32smoke+195正式、8running、317pending，无halt。
watcher574已结束；监管47140继续。最终sealer新覆盖一致性检查已确认补丁存在，两个报告脚本py_compile通过；FINAL_AUDIT及最终报表尚未执行，继续固定窗口监控。

04:38：1.5B TREC50三种子四方案齐，冻结new_tasks/analyze已输出dev/test各1/16完整。测试Base12.6/H66.133333/预算65.733333/HEU70.533333；相对H差值[.8,10.2,2.2]均值4.4，相对预算[3.0,10.0,1.4]均值4.8，两组各3/3正，但未校正及Bonf32区间均跨零（校正约[-69.587,78.387]、[-61.927,71.527]）。dev相对H[-1.171875,5.078125,5.46875]，并非全正；均值仍正。测试macroF1 H53.868884/预算52.978348/HEU55.697838，coarse81.8/81.266667/89.4。8B TREC首seed H87.2/预算86.4/HEU85.4，比两种对照均低，不能忽略该负例。
SQuAD2正式已启动，GPU7先评1.5B Base；未调整排程。watcher578已结束，监管47140继续。plot_results --preview已重画，CPU41198正常结束，当前core主对照全覆盖预览和新任务预览已目视检查（不代替最终四张图审阅）。audit_pairs.py新一轮在CPU93590执行，需确认其结果；其他最终收尾任务仍待固定窗口。

04:59：CPU93590已正常结束，跨方案初始化/样本序列配对审计314个可用运行记录、60个分组通过，104个复用checkpoint再次哈希通过（含运行中初始化记录，勿当作完成数）。watcher584已结束；最新247passed=32smoke+215正式、另1auditing、8running、296pending，无halt。早前手写04:24和04:38段落标题是近似观察时段，准确时间以STATE/调度日志/审计JSON为准。
7B TREC三种子齐，冻结new_tasks分析现2/16完整。测试Base30.6/H87.8/预算88.066667/HEU89.266667；HEU−H raw[1.4,2.0,1.0]平均1.466667，Bonf32[-5.876208,8.809541]；−预算[.6,.6,2.4]平均1.2，CI[-13.961139,16.361139]；全正但校正跨零。dev相对预算第三seed−.78125，另两seed正。8B TREC前两seed相对H均−1.8pp，相对预算−1.0/−.6，已向用户报告负例，第三seed仍跑。
1.5B SQuAD2 Base审计通过：F1 7.076870、answerable F1 14.153739、unanswerable0、schema22.65625%、raw empty0%。首seedHEU F1 74.713031、answerable70.715125、unanswerable78.710938；预算F1 73.624098、answerable70.490384、unanswerable76.757813。两适配模型schema/strictJSON均100%，整体差值1.088933来自已解析答案评分，而非格式有效率。H在测试尾部，尚无完整三种子配对。Base低分包含未学会输出格式，不能解释成没有阅读知识。

05:21：8B TREC三种子齐，new_tasks冻结分析现3/16完整。测试Base38.4/H87.733333/预算87.133333/HEU86.333333；相对H raw[-1.8,-1.8,-.6]平均−1.4，Bonf32[-11.507426,8.707426]；相对预算[-1.0,-.6,-.8]平均−.8，CI[-3.717763,2.117763]。各3/3负；未校正对预算区间[-1.296828,-.303172]，校正仍跨零。dev均值H79.296875/预算79.166667/HEU80.598958，平均反向，主/宏F1/粗类测试均下降。已向用户明确反例，不能主张双侧总有收益。
新增父级trec_error_analysis.py（不触及冻结科学代码），在观察8B开发/测试反向之后做事后逐例描述诊断：所有已完整三seed的模型、dev/test、两对照、50金标准类别全部保留，核对id/gold/原评分hash与准确率差=改善−退步；计数跨seed重复相同问题，不新增检验。当前--preview跑12组通过，生成TREC_ERROR_ANALYSIS_PREVIEW.json/.md/_CLASSES.csv。8B test相对H三个seed recovered7/6/6、regressed16/15/9；相对预算8/8/7与13/11/11。两对照均没有某个同一问题在三个seed全退步或全改善。退步跨多类别，相对H合计最大animal−7、count−4，相对预算animal−5、LOC:other−5；这些是重复评测计数，不是独立样本或因果机制证据。已告知用户结果及边界。
diagnose_results.py最终运行会调用新helper生成TREC_ERROR_ANALYSIS.json/.md/_CLASSES.csv，并记录组数；write_report附链接和事后声明；sealer要求该JSON status passed及两展示文件。4脚本py_compile通过，完整最终函数仍待FINAL_AUDIT。watcher588已结束；CPU93590早已关闭无待轮询。最新261passed=32smoke+229正式、8running、283pending，无halt；磁盘596GB剩余。监管仍47140/V3，后续继续固定时间窗。

05:45：SQuAD2仍在补三种子，暂无完整新QA块。1.5B首两seed H73.721755/73.831944，预算73.624098/74.021338，HEU74.713031/75.197975，双侧均正；第三HEU75.617368先完成。7B首两seed H84.379418/85.230362，预算84.464268/85.021331，HEU83.878711/85.067139，相对H两次均负、相对预算先负后正。8B首seed H87.355067、预算87.034415、HEU87.233154，相对H负、相对预算正，不能合并成一致收益。
8B QA Base整体F1 50.154772（schema100%，并非恒空），answerable86.832981/unanswerable13.476563；H首seed83.108571/91.601563，HEU80.911621/93.554688，说明总F1提升伴随可回答题下降、无答案题改善。7B Base25.602428，answerable48.470480/unanswerable2.734375/schema70.3125%。用户已获知1.5/7差异及8B Base分解，最终必须保留回答/无答案分开结果，不能只报整体涨幅。
watcher598已结束；监管47140/V3继续，最新274passed=32smoke+242正式、8running、270pending，无halt。GPU0在跑0.6B WikiSQL第三seed U-only最后一格；完成后其两核心九格三seed均齐。其余卡推进7/8B QA与1.5QA尾项；1.7/4分侧架构尚在priority25等待。最终审计/四图最终审阅/报告/全局三文档/封存及独立验证未执行，仍需11:07:10前完成。

06:05：冻结core分析8/20完整（8/16九架构三seed：.5/.6/7/8×两核心；五seed0/4）。0.6B WikiSQL新补standalone E14.322917/U3.776042/EU19.108073，Base5.371094；U-only三个seed均低于Base，差值[-1.660156,-1.074219,-2.050781]，平均−1.595052，Bonf176[-18.428162,15.238058]。HE71.419271/HU71.712240/HEU72.395833；H70.963542/预算71.223958/HEU主结果为旧复用前三seed。该块全部11校正区间跨零，不能概括每个独立分支都有效。
new_tasks分析4/16完整，新增1.5B SQuAD2。均值Base7.076870/H74.054530/预算74.101385/HEU75.176125；相对H raw[.991277,1.366031,1.007477]均值1.121595，Bonf32[-1.968945,4.212134]；相对预算[1.088933,1.176637,.958648]均值1.074740，CI[-.525440,2.674919]。未校正区间两者均正，但校正跨零，已向用户说明。回答子集F1 H71.025727/预算70.989229/HEU72.943395；无答案77.083333/77.213542/77.408854；所有适配schema/strict100%，本组提升主要在回答子集，不能仅说是格式修复或恒空刷分。
1.7B分侧/standalone正式已启动（EU首seed已过），4B随后；GPU0按内存合格次序已进入.5B TREC，非排程修改。watcher601已结束，最新286passed=32smoke+254正式、8running、258pending，暂无halt。监管47140继续；所有最终收尾程序尚未执行。

06:27：优先新任务1.5/7/8两任务三seed全部齐，冻结new_tasks/analyze dev/test各6/16完整。7B QA测试均值H84.763514/预算84.678717/HEU84.489529；相对H raw[-.500707,-.163223,-.158025] mean−.273985，Bonf32[-3.138709,2.590739]；相对预算[-.585557,+.045808,-.027816] mean−.189188，CI[-5.225737,4.847360]。HEU可回答F1 78.874891低于H79.618173/预算79.578788，无答案90.104167略高于89.908854/89.778646；三方案schema100%。7B dev平均双侧比H+.338077、比预算+.077660，测试反向。
8B QA测试均值H87.483349/预算87.429158/HEU86.929314；相对H raw[-.121913,-.433661,-1.106531] mean−.554035，Bonf32[-7.895496,6.787426]；相对预算[+.198739,-.216354,-1.481917] mean−.499844，CI[-13.271735,12.272047]。HEU回答F1 81.671128低于H83.300031/预算83.517171，无答案92.1875高于91.666667/91.341146，raw empty51.855469高于50.390625/50.260417。Base可回答86.832981/无答案13.476563，不能只报整体总分抹掉答题能力与拒答权衡。8B dev相对H平均+.007797、相对预算−.175025，同样不是稳定正收益。
用户已获知完整6块中：1.5两个正、7 TREC正QA负、8两任务负；全部Bonf32跨零，不能因区间跨零而隐去负均值。core主对照16条件全正的证据受任务选择与部分历史复用限制，不能主张普遍增益。
watcher604已结束；最新298passed=32smoke+266正式、8running、246pending，无halt。GPU1–7主要推进1.7/4核心分侧，GPU0推进.5/.6 TREC（按seed/model次序交错）。监管47140仍在，全部最终收尾未执行。

06:46：watcher607已结束；当前310passed=32smoke+278正式、8running、234pending，原priority待排40:87、30:54、22:52、25:41（22已按V3延后为35，别按原priority误判实际顺序）。1.7NER首seed九格齐：Base22.844383/H64.083601/预算64.289172/HE65.678853/HU66.356639/HEU66.505246/E41.442562/U37.771454/EU50.015768；第二seedHE65.694575/HU66.446125/HEU66.718897/EU49.405879，E/U还跑。4B首seed还缺U-only（test1024/1343在推进）；其余已过Base22.478576/H68.761076/预算69.018850/HE68.881399/HU69.541432/HEU69.901660/E46.510089/EU56.103060。暂不混入完整三seed统计。
scheduler_lineage于06:46再核验通过：三版源码manifest及两次共14个接管PID、92条仅pending的排队调整、所有原job spec与priority字段保留；科学配置无修改。无新调度修订。用户已获知当前完整块覆盖与时间预算内未完成项会明确保留。监管47140继续；最终收尾全部待执行。

07:09：watcher611已结束；最新322passed=32smoke+290正式、8running、222pending，无halt。1.7/4 NER第三seed推进；GPU0 .5B TREC第三seedH快结束，.6第三seed随后。4B U-only seed6100已通过：F1 48.882961、schema95.457930%、strictJSON97.617275%、capped2.382725%、平均47.830975tokens；耗时长但非停滞/失败，第二seed也接近尾部。0.5 TREC Base1.8，前两seed H52.4/51.0、预算53.4/51.8、HEU49.0/61.0，第三预算64.8/HEU67.4、H待齐；首seed双侧比两对照负，已向用户说明种子波动，不能只报平均涨幅。.6 TREC Base18.2，前两seedH82.4/83.6、预算82.4/82.8、HEU84.2/86.0。
CPU99008已完成再次plot_results --preview；已查看new_task_effects最新6块图，正负原始点与均值布局正常。它仍是预览，不能替代最终四PNG审阅与FIGURE_REVIEW。当前无任何CPU exec待轮询，只需监管47140与下一个watcher。最后完整审计/诊断/报告/最终图/全局三文档/封存和11:07前独立验证仍未执行。

07:30：冻结core/analyze现9/20完整（9/16三seed、0/4五seed），新增1.7 NER九格：HE65.741446/HU66.475676/HEU66.740263/E42.204956/U37.542772/EU50.378479，Base22.844383/H64.113765/预算64.302343。U−Base均值14.698389，Bonf176[6.042860,23.353918]为正；其余10校正区间跨零，包括双侧相对H与预算、独立EU−Base。HEU−HU三个seed[.148607,.272772,.372382]均正但CI跨零；不能推广双侧总最优。所有此1.7结果为新跑。
new_tasks分析现8/16完整：新增.5/.6 TREC。.5 Base1.8/H55.533333/预算56.666667/HEU59.133333，相对H[-3.4,10.0,4.2]平均3.6，相对预算[-4.4,9.2,2.6]平均2.466667，均2正1负且Bonf32跨零。.6 Base18.2/H83.2/预算82.733333/HEU84.8，相对H[1.8,2.4,.6]平均1.6，相对预算[1.8,3.2,1.2]平均2.066667，均3正但Bonf32跨零。用户已获知这两组波动/正均值及1.7 U独立适配证据。
watcher616已结束，最新334passed=32smoke+302正式、8running、210pending。4B NER第三seed HU/E/U在test，1.7 SQL首seed各分侧在test；GPU0进入.5 SQuAD2 Base。监管47140继续，无其他CPU会话待轮询；全部最终收尾工作尚待。

07:46：watcher619已结束，最新348passed=32smoke+316正式、1auditing、8running、195pending，无halt。4B NER第三U-only仍在test1024/1343持续推进；1.7 SQL第二seed、4 SQL第二seed已接续。0.5 QA Base通过F1 9.711858、回答16.494029/无答案2.929688/schema33.007813%；首seedHEU64.984588、回答51.062926/无答案78.90625/schema99.804688%，预算审计中，H刚启动，不能当完整配对。
针对新增1.7/4边界adapter覆盖，父级audit_boundary_algebra.py再次启动于CPU exec 66837，需轮询其结果；这是固定首seed、采样词表行/随机hidden的数值恒等式核对，仍非全词表或端到端合并推断评测。监管仍47140；最终所有收尾工作未执行，09:37:10停准入、10:07:10停本轮worker、11:07:10前交付与独立验证期限不变。

08:06：CPU66837已正常结束，BOUNDARY_ALGEBRA_AUDIT更新40个固定可用首seed单元全部通过、另8单元缺失（1.5B EU/HE/HU，3B E/U/EU/HE/HU尚未完成）；最终审计需再次运行以纳入新完成项。矩阵报告记录字段是records，避免误用cells造成输出整份大JSON。
4B NER第三seed U已过，冻结core分析10/20完整（10/16三seed，0/4五seed）。均值HE68.867822/HU69.863020/HEU69.666776/E47.700155/U49.238456/EU56.587265；Base22.478576/H68.606527/预算68.911843。U−Base +26.759879，Bonf176[8.882627,44.637131]；EU−Base +34.108689，CI[18.762488,49.454890]，两者为正，其余9个校正区间跨零。HEU−HU raw[+.360228,−.272470,−.676489] mean−.196244，已向用户明确双侧并非普遍最佳。全部此4B结果新跑。
watcher623已结束；最新364passed=32smoke+332正式、8running、180pending，无halt。1.7 SQL第三seed正在尾部，4SQL第三seed在训；GPU0 .6QA首seed预算在训。无CPU exec待轮询，仅监管47140与后续watcher。完整最终审计、诊断、四图最终审阅、报告及全局三文档、封存与11:07前独立验证仍全部未执行。

08:28：冻结core分析12/20完整（12/16三seed，0/4五seed），1.7/4 WikiSQL九格补齐。1.7均值HE74.218750/HU74.348958/HEU74.902344/E43.196615/U12.337240/EU54.101563，Base5.273438/H73.372396/预算73.404948；仅EU−Base +48.828125的Bonf176[6.922915,90.733335]为正，其他10区间跨零，HEU−HE seed7101−.195313。4B均值HE81.347656/HU81.966146/HEU82.324219/E54.752604/U48.730469/EU65.983073，Base39.0625/H81.022135/预算81.445313；仅U−Base +9.667969的Bonf176[.819518,18.516419]为正，其他10跨零；HEU−HU seed7101−.488281、HEU−HE同seed−.195313。主叠加三seed均高于H与预算但该族校正仍跨零，所有新尺寸主对照结果没有改变。
Q25-1.5/3缺失分侧架构已进入队列：1.5首seedHE/HU/EU通过，3首seed五臂在评测，1.5第二seed开始。GPU0 .5QA第二seed预算在训。watcher626已结束；最新378passed=32smoke+346正式、8running、166pending，无halt。监管47140继续，无CPU会话待轮询。
最终需更新的三全局文件已再次读取：lora/README.md首段、lora/docs/RESULTS_SO_FAR.md首段、emnlp/iclr2027/submissions/paper-2/EXPERIMENT_EVIDENCE_STATUS.md首段/证据表及新增节；均在当前workspace。/home/wz/projects/mypro/im_exp为同一realpath别名，不要双写。lora README旧“当前主线/三条主张”仍是历史loss工作流，可最终改标题为历史工作流与研究假设并加边界说明，保留原loss账本与旧结果。当前尚未做这些最终文档更改，根README仍In progress。仍需所有最终审计、报告、图、封存与独立验证，不能在启动后结束。

08:56：watcher631已结束，最新391passed=32smoke+359正式、8running、153pending，无halt。Q25-3B NER第二seed多数在test尾部，1.5第三seedHE/HU/EU已启动；GPU0 .6QA第二seedH在训。截止准入09:37:10不足一小时，仍按原窗口运行，未修改allowance/优先级/科学设置。
新QA小模型仍不完整：.5前两seedH65.928660/63.103072，预算65.825170/61.461989，HEU64.984588/61.800090；相对H两次负、相对预算先负后正。第二seedHEU回答F1 69.693930高于H61.557706/预算64.134915，但无答案53.90625低于64.648438/58.789063，不能把答题子集提高等同整体提高。.6 BaseF1 .715014（回答1.430027、无答案0、schema2.734375%，包含严重格式失败）；首seedH74.114856/预算74.310168/HEU74.500648，第二预算74.189849/HEU73.801696、H待齐。用户已获知正负混合，不宣称未齐三seed条件有效。
监管仍47140；无CPU会话待轮询，全部最终收尾工作未开始。下一watcher建议在09:25左右回到主循环检查接近截止的运行，再于09:37停准入后持续等所有本轮worker/auditor退出，最多10:07。一旦SCHEDULER_COMPLETE且监管退出就开始finalize_audit.py，不必等到10:07整点。

09:20：watcher634已结束；最新404passed=32smoke+372正式、8running、140pending，无halt。磁盘588GB剩余。1.5 NER九格三seed齐，冻结core分析13/20完整（13/16三seed、0/4五seed）；3B NER第三seed各臂评测尾部、U-only较慢仍test256/1343。1.5SQL首seed各单侧在评测，GPU0 .5QA最后H种子在评测。
1.5 NER均值Base0/H59.001686/预算59.054387/HE61.276797/HU60.340212/HEU62.247485/E36.051611/U30.903917/EU45.656974。新HE/HU/EU，其他旧复用。EU−Base +45.656974 Bonf176[19.605555,71.708393]为正，其他10区间跨零；双侧相对HE/HU各3/3正，平均+.970688/+1.907273，不能把此排名扩展到其他型号。原五seed八比较的既有正证据不被本轮三seed/176族覆盖或否定。用户已获知这组分侧结果。
下一watcher可直接监控到09:37:10停准入（注意UTC 01:37:10）。仍无CPU会话待轮询，监管47140继续。最后一个小时的完整审计/报告/四图最终审阅/全局文档/封存与独立验证均尚未执行；不要结束或拖过11:07:10。

09:39：**已于09:37:10按计划停止准入**，watcher637已结束。STATE09:37:45为420passed=32smoke+388正式、7running、125not_admitted_time_budget、0pending，无halt。未启动分布：core7B20、8B32（合计52额外untied种子）；core1.5B SQL5、3B SQL7；new_tasks3B/1.7B/4B各20（合计60），.6B QA第三seedH1。未启动125不能写成失败或结果；.6QA即便另外方案三seed齐，H仅2seed，不生成四臂完整块均值/检验。
剩余7worker：3B NER第三U test1248/1343；1.5SQL seed7101 HU/E/U评测；3B SQL seed7101 HE/HU/EU刚训。最多10:07:10停本轮worker，预计可能更早自然结束；所有worker/auditor退出、SCHEDULER_COMPLETE出现且监管47140退出后立即finalize_audit，别机械等10:07。
冻结core当前13/20完整，3B NER尾项过后应14/16三seed（按实际再核实）；1.5/3SQL九格三seed不齐，主对照16/16依旧齐。冻结new_tasks现9/16完整，新增.5QA：Base9.711858/H63.590076/预算62.896418/HEU63.390123；相对H[-.944072,-1.302982,+1.647194] mean−.199953，Bonf32[-23.683723,23.283817]；相对预算[-.840582,+.338101,+1.983594] mean+.493704，CI[-20.200573,21.187982]，1正2负与2正1负。回答F1 H61.620256/预算61.925649/HEU62.847954；无答案65.559896/63.867188/63.932292；schema99.707031/99.707031/99.837240，不能只挑回答子集称整体胜出。
用户已获知准入停止和125未启动。所有最终收尾仍未执行；无CPU会话待轮询，监管47140仍活动。注意新watcher无pending后应以running/auditing/awaiting_audit全为0退出并启动最终工作。

10:04最终封存准备：watcher640已自然结束；监管47140已poll确认exit0，SCHEDULER_COMPLETE时间09:50:18，STATE427passed/125not_admitted，无其他状态。finalize_audit.py会话86951成功退出，FINAL_AUDIT passed：core296新正式+16smoke、418188输出、185330官方SQL检查；new_tasks99正式+16smoke、100648输出、59278作者QA检查。合计377fits+18Base+32smoke，518836新输出；55352+49056=104408token记录重新编码，499可用正式运行/82组配对、104复用checkpoint再次哈希；48固定边界矩阵单元全部通过无缺失。
diagnose_results/plot_results/write_report均成功完成（65564/57540/79462全部exit0）。报告已加入人工科学解释：14完整核心块28叠加主CI全部跨零；9新任务块5均值双正、3双负、1混合，18新任务CI全部跨零；具体standalone正校正证据与单侧/叠加负例保留。诊断4291次指标行、473训练历史行、14交互条件、20 TREC错误核对组。FINAL_INTERPRETATION_ZH.md、两份CONTRASTS、ALL_AVAILABLE_TEST_RUNS.csv(499行、104reuse)、CORE_MAIN_ARMS_DESCRIPTIVE.json与REPORT_STATUS生成完整。
四张最终PNG已目视审阅；缺失标注从数据点旁移到空白处，untied差分的复合对照加括号，仅展示改变。FIGURE_REVIEW.json记录实际四PNG哈希，core PNG与首次审阅字节相同。plots支持PNG/PDF/SVG和plotted_values.csv。检查本目录无.tmp残留、无symlink，主要报告全部本地链接可解析。三全局文档和根README已更新实际覆盖/正负结果/遗漏；旧封存研究与loss账本保留。接下来只需seal_results.py，然后verify_seal.py独立只读复核，验证日志放本目录外；不要在封存后写本目录。最终仍须11:07:10前完成复核，才能给用户最终答复。
