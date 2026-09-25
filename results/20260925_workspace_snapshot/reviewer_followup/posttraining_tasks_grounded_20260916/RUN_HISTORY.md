# Active-run handoff

User authorized trying better post-training tasks, asks continued polling and
verified reuse. Do not terminate the active experiment prematurely. No subagents.

Workspace `/commondocument/wz/cross_encoder_workspace/im_exp`, Python
`/home/wz/anaconda3/envs/torch24/bin/python`. GPUs1..6 reserved for this queue;
GPUs0/7 have unrelated processes. Never stop unrelated jobs.

## Completed

- V1 `../posttraining_tasks_20260916` SEALED: 215 files,
  manifest SHA256 `ef25da8c2f7bc4ae8c20e06dc6f8df15576f06375196afff99d09f5e5c28f467`.
  Four smoke and four hidden training jobs, 2,400 responses audited.
  No V1 A-LoRA task scores. Do not rerun its writers or modify its archive.
- V1 ToolACE included unanchored relative-date gold; direct character offset
  generation made entity-span F1 about2 despite text F1 about51. Complete record
  retained; V2 prepared before observing any A-LoRA task score.
- V2 data: grounded positive ToolACE calls only, each leaf argument appears
  literally in the user request. API-disjoint split preserved, no abstention or
  general/BFCL claim. Train3548/dev696/reserved-test710. NER type/text/occurrence
  mechanically maps to unchanged original spans without gold access. Entity
  train10148/internal-dev600/reserved-public-dev1343. Pilot uses2048/200.
- Both official Base model file hashes rechecked. All V2 adapters fresh.
  Four smoke checks passed, two Base/hidden task pilots complete, two fixed-demo
  Base NER diagnostics complete. Code/data frozen before V2 evaluations.
- V2 seed6100 final hidden content: ToolACE Q3 66.5 vs Base29.5;
  Q25 66.0 vs29.5. CLUENER span F1 Q3 59.753593 vs Base0.488400;
  Q25 64.876033 vs0. Two-training-demo unadapted Base NER scores18.155620/
  25.169147, JSON100/96%, caps0/9.5%. These diagnostics have different prompts;
  do not use them as matched adapter controls.
- TASK_GATE passed both tasks. Main tensor-scope audit verifies identical
  original parameter digests across arms, pure input/output have zero hidden
  trainables, native EOS once/target, no ChatML tokens. Rerun at completion.

## Running

`adapter_pipeline.py` supervisor, exec session75101, log adapter_scheduler.log.
96 NEW jobs +4 reused V2 hidden seed6100 =100 fitted configurations:
2tasks x2models x5arms x5seeds6100..6104.
Arms hidden, input, output, hidden_both, hidden_budget. The initial hidden seed
6100 is reused only after all source/config/checkpoint/order/input checks; see
PILOT_REUSE.json. StageA included step16 generation but no stochastic sampling.

Each2048 rows/64steps, LR2e-4, batch32/micro2; only one common LR (pilot, not
full tuned confirmation). FP32 trainables/Adam, BF16 autocast training, FP32 eval
of BF16 base values, SDPA/TF32off, greedy512tokens/batch8/nativeEOS151643.
Queue pool GPU_POOL.json now1..6; no other current task scheduler remains active.
Do not restart the full pipeline: checkpoint directories intentionally refuse
overwrites. Preserve any failure and use a new attempt if necessary.

Read ADAPTER_PILOT_STATE.json for counts/PIDs. `monitor.py` is read-only but
verbose after many completions; use a short Python reader as needed. Poll about
45–50seconds and send concise Chinese commentary at least once/minute. Do not
infer stable effects from individual seeds. Final public test partitions remain
unscored throughout this bounded task-selection/architecture pilot.

## Completion work

1. Ensure ADAPTER_PILOT_COMPLETE status passed (96 jobs), no owned jobs remain.
2. Verify FEWSHOT_FROZEN files too (they were added independently after the main
   DATA_FROZEN manifest). Run audit_parameters.py (all model files, trainable
   scopes and frozen tensor identities,43,482 token records including fewshot).
3. Run audit_results.py to re-decode raw tokens/rescore every complete result;
   expected110 evaluation configs x200 =22,000 V2 responses,104 training runs
   including4 smoke+4StageA+96new. Avoid rerunning writers after final sealing.
4. Run analyze.py: paired initialization/order and exact budget checks; primary
   comparisons both−hidden and both−hidden_budget (2models x2tasks x2 =8),
   Bonferroni seed-t intervals, separate nominal conditional cluster bootstrap.
   Main tasks' all outcomes retained. ToolACE cluster mapping currently uses
   full V2 dev connected components; entity clusters are text/document rows.
5. Run plot_results.py; inspect exported scientific plot (PNG/PDF/SVG). Write
   final Chinese interpretation, separating learning-task suitability from
   standalone effects and additive effects. Include two-shot Base diagnostic,
   literal-copy subset scope, single-LR/small-dev limitations and V1 revisions.
6. Update current README and workspace evidence docs with factual pilot results
   and links; do not modify manuscript tables or any prior sealed archive.
7. Save completion and final immutable manifest, check it read-only once. V2
   includes analyses/report scripts added after the original code freeze, which
   is okay; never modify the frozen training/scoring/data files.

Do not claim BFCL results, general new-knowledge learning, parameter efficiency
versus every possible LoRA placement, or global absence of gains from null CIs.
Equal-budget counts: Q3both and budget5,112,832; Q25both and budget9,332,224.
Standalone input33,792/50,688 and output32,768/49,152. Both original models tied;
the independent input/output maps here are not a tied-mergeability proof.

## IMPORTANT: held-out phase added and authorized scope continues

User was told that after five-seed development, BOTH tasks and ALL controls will
be evaluated on the reserved data. Do NOT finalize at development completion.
`heldout_pipeline.py` is running, exec session6401, output heldout_scheduler.log.
It prepared/froze HELDOUT_PROTOCOL.md, HELDOUT_FROZEN.json and code/data before
any reserved-test predictions (and before remaining development seeds finished).
It waits for ADAPTER_PILOT_COMPLETE, then runs 8 development numerical probes
(first32 rows of four Base + four hidden seed6100 checkpoints). Batch32 raw
tokens are compared to saved batch8 outputs. ANY mismatch globally selects
batch8; otherwise use32. Choice saved in HELDOUT_BATCH.json before test inference.

Then106 held-out configurations run on GPUs1..6:
4 unadapted Base +100 fitted configurations +2 two-shot Base NER diagnostics.
Total expected109,442 held-out responses =52*710 +54*1343. No new training.
Every loaded original-weight tensor digest and adapter tensor/file hash must
match the source training checkpoint. Same fixed prompts/scorer/greedy512/EOS;
only the globally checked batch size may differ from development. No test CE
is computed; this phase confirms content and output behavior. This is still
small-training-data, one-common-LR evidence, not tuned/full-data/BFCL results.

Once complete:
- Finish the development audit/analysis/plots as above (22,000 responses).
- Verify HELDOUT_COMPLETE passed106 jobs. Run audit_heldout.py (109,442 responses),
  analyze_heldout.py -> HELDOUT_ANALYSIS.json / HELDOUT_RESULTS_ZH.md, then
  plot_heldout.py -> figures/heldout_additive_effects.{png,pdf,svg}.
- Interpret held-out main results with the same paired seed family8 and separate
  conditional item/cluster intervals. Report standalone means as secondary.
- Include both few-shot Base NER references as prompt-sensitivity diagnostics;
  never treat their different prompts as matched adapter contrasts.
- Re-run audit_parameters.py at completion; two new fewshot_test token files
  bring token record count from43,482 to46,168.
- Seal V2 only after ALL schedulers, analyses, documentation, and visual review
  finish; final manifest must include both development and held-out phases.

V2 current initial seed6100: tool hidden/both/budget Q3=66.5/67.0/67.5;
Q25=66.0/65.5/66.5. NER Q3=59.7536/62.0553/60.0619;
Q25=64.8760/66.3218/65.0778. These are single-seed development observations only.
About22/96 new development jobs finished at the last update, no failures.

## Audit supervisor added (18:02 local)

`finish_analysis_pipeline.py`, exec session8634, output `analysis_scheduler.log`,
is waiting for the development phase. It will automatically run parameter/data
identity checks, development audit/analysis/plots, then wait for held-out
completion and run its audit/analysis/plots. Read `ANALYSIS_STATE.json`,
`DEVELOPMENT_ANALYSIS_COMPLETE.json`, `ANALYSIS_COMPLETE.json`. Do not start
duplicate audit writers while it is running. It asserts110 checkpoint scopes,
46,168 token records,22,000 dev responses,104 training runs,109,442 held-out
responses. Final interpretation, visual inspection, docs and sealing are still
manual completion work after this supervisor finishes.

A factual report-template typo was found: both frozen analysis scripts inherited
“包含无调用情形” from V1, although V2 data correctly excludes no-call examples.
Frozen sources/data/scores remain unchanged. `correct_report_scope.py` corrects
ONLY this sentence in both generated Markdown reports after analysis, saving
exact before/after hashes in REPORTING_CORRECTIONS.json. Supervisor runs it once.
FEWSHOT_FROZEN's five extra files were independently hash-checked successfully.

## Later monitoring notes (18:32 local)

69/96 development jobs recorded complete, zero failures, fourth seed ongoing.
Three complete paired seeds: ToolACE both-minus-hidden Q3=-0.667/Q25=-0.500 pp;
NER Q3=+1.656/Q25=+2.226 F1, both-minus-budget=+1.141/+2.136. These remain
DEVELOPMENT interim observations, not final effects or held-out conclusions.
Functions store key `posttraining_poll_cmd` contains a short read-only status
command for all three stage states and analysis status. Wait40 seconds between
brief polls/updates when otherwise idle, less if doing useful work.

`seal_results.py` is prepared but NOT run. It requires all phases/audits/reports
complete. Finish interpretation, docs, visual inspection and replace this
handoff's active-state text BEFORE sealing. Do not redirect seal stdout to a
file inside this archive (that file would change after hashing). Capture exec
stdout, then no more writers. Manifest excludes only itself and __pycache__.

Added ongoing-experiment link paragraphs to lora/README.md and
lora/docs/RESULTS_SO_FAR.md. Replace those paragraphs with final evidence links
after completion; paper-2/EXPERIMENT_EVIDENCE_STATUS.md still needs final update.
All frozen core/data/protocol files hash-checked again successfully; no changes.
Disk had690GB free, current archive2.7GB. GPU0/7 still unrelated, use1..6.

## Development completed; held-out running (18:59 local)

All96 new training/development jobs passed. Exec session75101 closed exit0.
`finish_analysis_pipeline.py` (session8634) completed development audits/analysis/
plot and now waits for HELDOUT_COMPLETE. Do NOT rerun those writers.
PARAMETER_AUDIT passed110scopes,14model files,46,168token records; RESULT_AUDIT
passed22,000responses and104training runs. ANALYSIS pairing/budget checks passed.
Development plot figures/additive_effects.png was visually inspected: labels,
intervals and footer clear. Exported PNG/PDF/SVG exist.

Development five-seed primary means (hidden / both / equal-budget):
ToolQ3 66.00/66.00/66.40; ToolQ25 67.50/66.80/67.70.
NERQ3 61.18/63.58/61.43; NERQ25 63.02/65.13/62.88.
NERQ25 both−hidden +2.107 simultaneous-family8CI[+0.237,+3.976];
both−budget +2.253CI[+0.646,+3.859]. NERQ3 means positive but seed CIs cross0.
All tool main CIs cross0. These are DEVELOPMENT only, not finalheldoutresults.

Eight pre-test numerical probes passed:256rawtoken sequences plusEOS match
batch32vsbatch8 exactly, so HELDOUT_BATCH selects32. Six actual held-outjobs
started18:56:34, currently0/106done,0failed. Larger710/1343rows means several
minutes/config; keepmonitoring106and109,442responses throughcompletion.

`correct_report_scope.py` now supports incremental correction: development
RESULTS_ZH.md was already corrected for the inherited no-call wording, with
before/afterhashes in REPORTING_CORRECTIONS.json. At the end, analysis supervisor
will call it again, verify unchanged corrected developmentreport, and correct
new HELDOUT_RESULTS_ZH.md. No frozen sources or metrics changed.

## Held-out update (19:36 local)

31/106 configs done,0failed,about30,854/109,442responses generated. Continue
monitoring until all106 complete; no source/protocol changes. GPUs1..6 busy.
`posttraining_poll_cmd` store now focuses on HELDOUT_STATE, generated-response
progress and (only after heldout completion) analysis state. The audit supervisor
8634 is WAITING, although ANALYSIS_STATE retains its previous plot_results.py
label; DEVELOPMENT_ANALYSIS_COMPLETE passed, nothing is stuck. Heldout session6401
still active. Closed training session75101, do not poll it again.

All first-seed held-out results are available (still NOT five-seed conclusion):
ToolQ3 Base31.1268,hidden57.7465,input39.8592,output28.7324,both56.9014,budget57.7465.
ToolQ25 Base23.6620,hidden61.6901,input43.9437,output35.2113,both61.4085,budget62.1127.
NERQ3 Base0.5912,hidden60.4762,input28.8953,output19.2558,both60.8872,budget60.7611.
NERQ25 Base0,hidden59.1594,input35.3158,output31.2407,both62.1054,budget59.1934.
Thus retain the negative Q3 tool output-only finding too; do not claim that all
standalone placements work on everytask. Await fullfive seeds and finalaudits.

Additional un-frozen analysis/docs added:
- diagnose_lengths_and_loss.py run successfully -> TOKEN_LENGTH_AND_LOSS_DIAGNOSTIC.json.
  All heldout goldtargets fit512tokens (Toolmax330,NERmax138 includingEOS).
  OrdinaryLoRA mean-vs-Base CE reduction EOS shares:ToolQ3 .063875%,ToolQ25 .333996%,
  NERQ3 .283958%,NERQ25 .032669%. This is DEVELOPMENT teacher-forceddiagnostic,
  not a content score; answer CE includes syntax and copiedtokens.
- METHOD_AND_LIMITS_ZH.md explains V1->V2revision, controls, provenance, token
  consistency checks and bounded claims in Chinese. Use as finalreportmethodlink.
  Includes smallseed t assumptions, language/dataset confounding, cannot infer
  which boundaryside caused stackedgains from both-only comparison.
- verify_seal.py is prepared, read-only fullmanifest and extras/missingfiles
  checker. Not run before sealing (manifestnotyetexists). Include command in
  finalREADME; seal_results.py remains prepared but MUST waitforallreports.

Final remaining work is unchanged: let heldout + analysis supervisors finish,
inspectheldoutplot, write FINAL_INTERPRETATION_ZH.md andfinalREADME, update the
three workspaceevidence docs, replacehandoffactive notes, closeownedcompleted
sessions, seal, read-onlyverification. Keep all resultsandallnegative signs.

## One additional final analysis (19:40 local)

Prepared `describe_heldout_content.py` (NOT run; waits on HELDOUT_AUDIT passed).
Run it manually after the automatic heldout analysis finishes, before final
interpretation/sealing. It retains all26groups (24main+2fewshot), summarizes
seed-mean NER precision/recall, tool function-name accuracy, JSON/schema and
termination, and saves HELDOUT_CONTENT_DIAGNOSTIC.json with106sourcehashes.
These are explicitly post-hoc descriptive auxiliary metrics, never replace
main family8contrasts or claim a causal mechanism. They help check whether
NER gains occur with already-near-perfect JSON validity.
Currentheldout35/106,0failed,about35,568responses. All main sources still frozen.

## Three held-out seed main contrasts available (20:11 local)

63/106 configs complete,0failed,66,625responses generated. Main hidden/both/
budget results for seeds6100..6102 are all available (some auxiliaries maystill
be finishing); NOT finalfive-seed inference. Heldout both−hidden means:
ToolQ3 +0.2817, ToolQ25 +0.8920; signs vary acrossseeds. These heldout tool
means are mildlypositive, unlike the developmentmeans; do not force a
“negative/no average gain” narrative from earlier developmentorfirstseedresults.
NERQ3 +1.0589, NERQ25 +3.2458; versusbudget means +0.8899/+3.1930.
NERQ25 per-seed both−hidden=2.9460,3.7705,3.0208; versusbudget=2.9120,3.7540,2.9132.
Await five-seedfamily8CIs and conditionalbootstrap fromautomaticfinalanalysis.

V2README androotloraREADME now correctly state developmentfinished/heldoutactive;
replace withfinalstate aftercompletion. FEWSHOT_FROZEN's code_sha256 additionally
matched fewshot_diagnostic.py exactly; seal_results.py now explicitlychecks it
andrecords thathash inFINAL_AUDIT. No frozenfileschanged.

## Shared-GPU contention observed (20:17 local)

An unrelated other-user eight-GPU job started around20:09, adding~5.4GiB on
eachdevice and slowing inference. A separate unrelated VLLMengine runsGPU0.
Never stop/change those jobs. Our trainingcompleted18:55, contentprotocol
unchanged, allheldoutjobsstillhealthy. RESOURCE_CONTENTION.json records the
observation; rawseconds cannot support controlled throughputclaims.
Currentheldout66/106,0failed,68,716responses. Continue1..6asconfigured.

## Four held-out seeds complete (20:56 local)

All80fitted configurations for seeds6100..6103 have SUMMARY files, plusfourBase.
Current85/106complete,0failed,88,856responses. Lastseed6104andtwofewshotdiagnostics
remain; DO NOT finalizebeforethem. ForeignGPUcontentionstillpresent; continue
without touchingotherjobs. Code/data/scorer remainfrozen.
Four-seedNER means hidden/both/budget:
Q3 60.9407/62.3771/61.0219;Q25 59.1331/62.5754/59.2158.
AllfourNERseedboth−hiddenandboth−budgetdeltaspositiveinbothmodels, butfinal
five-seedfamily8CIsnotyetcomputed. Do not inferstatisticalconclusionsfromsignsalone.

Additional sourceinspectionconfirmed run.py calls model.train() at EVERYtraining
step, so seed6100's step16evaluation cannot leave subsequenttrainingindropouteval
mode. No fixneeded. seal_results.py now ALSO rehashes all14officialmodel files
at sealing (inadditionto pre-heldoutparameter audit and everyloadedtensorcheck).

## Final tail and report generator (21:47 local)

105/106heldout configs done,0failed,109,155/109,442responses. Onlyremainingjob
is diagnostic_cluener_qwen25_15b_base_two_shot_base, running since21:24.
No jobsqueued; finish it too. Main100fitted+4Base andQ3fewshot arecomplete.
Two owned supervisors remain6401(heldout),8634(analysiswaiting). Do NOT rerun
analysiswriters; automaticpipeline will audit109,442,analyze,plot,correctreports
andwriteANALYSIS_COMPLETEafterlastdiagnosticends.

Mainfive-seed provisionalrawsummarymeans (awaitindependentre-score):
ToolQ3 hidden56.309859,both57.098592,budget56.028169; both−hidden+.788732
CI8[-1.697741,3.275206],both−budget+1.070423CI8[-2.008049,4.148894].
ToolQ25 hidden61.408451,both62.985915,budget61.408451; both−hidden+1.577465
CI8[-1.248046,4.402975],both−budget+1.577465CI8[-1.917011,5.071941].
NERQ3 hidden60.816624,both62.516529,budget60.910690; both−hidden+1.699905
CI8[-.589167,3.988977],both−budget+1.605840CI8[-1.011685,4.223364].
NERQ25 hidden59.172615,both62.565222,budget59.240209; both−hidden+3.392608
CI8[2.258720,4.526495],both−budget+3.325014CI8[2.254037,4.395990].
AllfiveNERseeddeltaspositiveinbothmodels. OnlyQ25NER'stwofamily8intervals
excludezero. Allfourtask/modelmeanspositiveonheldout; donotsaytools“nogain”
whenitismeanpositivebutseedCIcrosseszero.
StandaloneheldoutBase/input/output:
ToolQ3 31.1268/39.9155/28.9577;ToolQ25 23.6620/42.8451/35.1268.
NERQ3 .5912/29.9685/18.6831;NERQ25 0/36.4431/31.3931.

NEW write_interpretation.py is prepared and syntaxchecked, NOT run. After
ANALYSIS_COMPLETE, first run describe_heldout_content.py, then
write_interpretation.py. It renders FINAL_INTERPRETATION_ZH.md from audited
analysis/auxfiles with maintables,CIs,standalone,fewshot,P/R,limits andlinks.
It assertsobservedmainCIpattern. Reviewgeneratedtextandheldoutplotvisually.
Then finishREADME/rootREADME/docsRESULTS_SO_FAR/paperevidencestatusupdates,
replacehandoffwithcompletedrecord,closebothfinishedsessions,seal_results.py
(without redirectinsidearchive),verify_seal.py read-only, thenfinalChineseanswer.
