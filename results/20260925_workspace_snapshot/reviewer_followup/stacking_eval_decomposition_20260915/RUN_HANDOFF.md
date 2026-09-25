# COMPLETE: evaluation decomposition

Completed 2026-09-15T14:20:17+08:00. All 24 checkpoint re-evaluations and 168 exploratory diagnostic contrasts complete. FINAL_AUDIT.json passed: 231768 public-reference old content-NLL/top1 replays, max NLL error0; 24672 internal-example replays maxerror3.0349e-6; FP64 CE check max8.3913e-7. No active processes or outstanding experiments. Scheduler99539 and all other tool sessions closed.

Main result: Q3 token-micro content CE improves all3seeds bothcontrols whereas macro reverses seed43. All4comparisons mean short-answer(1-4tokens) CE worsens, otherbins improve. Internal gains88-97% fromcontent, notEOS-only. Complete-reference-set meanprob gains both-none/budget Q3+.3910/+.3880pp, Q25+.6095/+.5869pp, all12paired signspositive; exploratory metric, no replacement of officialgen results. Q25 first-token predictions and later tokens can move oppositely.

Independent ANSWER_SET_CHECK.json passed direct FP64exp/sum/log aggregation and first/mean/min/max repeated-reference sensitivity. Duplicate identicalanswers DO have original-batch probability differences (max question setrange.082229, mean.0008133); initial strict equality checkfailed and preserved in protocol_history/duplicate_probability_equality_check. Even both-min minuscontrol-max observedduplicate bounds retainpositive mean gains for all12pairs. This is NOT a bound on allBF16 error or fullFP32 recomputation. Initial smoke topk-tie implementation failure also retained; full runs usedcorrectedargmax andlexicographictop5, oldevalunchanged.

FINAL_INTERPRETATION_ZH.md contains full nuanced conclusions and metrics. EVALUATION_V2.md documents updatedmetric/reportcontract. figures/evaluation_decomposition.png/.pdf reviewed afterlegendfix. OldA/B artifacts hashes recheckedunchanged. ARTIFACT_MANIFEST.json freezes finalresults,reports,figures,postprocessors,history files. Do not rerun analyze/plot/report overfrozenfiles; newwork goeselsewhere. Papertex notedited.

Historical running handoff follows:

---

# ACTIVE: evaluation decomposition, 2026-09-15

Latest user: 我们的效果评测是不是需要更新一下，你觉得呢，或者你可以更加深入的思考和验证一下. Authorized deeper verification; previous user asked foreground polling through all experiments. Continue until all 24 checkpoint evaluations, analysis/audit/plot/report complete, then final Chinese answer. No subagents, goals, new training, new benchmarks, or modifications of old frozen artifacts.

Cwd /commondocument/wz/cross_encoder_workspace/im_exp. Python /home/wz/anaconda3/envs/torch24/bin/python. Current directory D=lora/reviewer_followup/stacking_eval_decomposition_20260915. B=../stacking_cmrc_task_sft_20260915, A=../stacking_chinese_transfer_20260915. A+B all completed/audited in preceding turn and must remain unchanged.

Active main scheduler tool session 99539; stdout main_scheduler.log. All smoke/prepare sessions closed. Current ~14:03 +08:00: 8/24 done, 8 running seed43, 8 queued seed44. Qwen3 seed43 ~8200/9657 public refs; Qwen2.5 seed43 ~2700/9657. All internal-dev runs so far passed. Poll via Python D/poll.py every ~45s and give Chinese commentary <=60s. No formal failures. Expected finish around14:10.

D fixed main inputs manifest.json: shared.py/evaluate_parts.py/prepare.py/run.py/DESIGN.md/PREPARATION.json. DO NOT EDIT these now. PREPARATION verified full B final audit inputs/checkpoints/base weight hashes and A/B exported artifact manifests; both tokenizers internal1028 + public9657 template alignment all exact.

New evaluation loads all24 B checkpoints (2models x4arms x3seeds), internal_dev1028 with identical old full template/batch1, publicdev9657 refs with exact old prefix+answer/batches up to16/4096. Computes content CE and first/rest parts, EOS from last-content logits (no extra input token), top1/top5, gold logit margin, teacher_exact. Internal old NLL abs error<1e-4 allowance; public old NLL error<1e-8, tokens/top1 exact. FP64 reference<5e-6 first batch eachsplit. Per-token top1 must use argmax; top5 lexicographic descending logit ascending tokenID. First smoke mistakenly used topk[0] as top1; ties caused2oldtop1replay failures. Preserved all initial smoke in protocol_history/smoke_topk_tie, corrected before mainfreeze; all8 smoke then passed. Old upstream already argmax, unchanged. Failure is NEW diagnostic implementation, not old results.

prepare.py finished PREPARATION.json. CPU important finding: public CE weighting matters. both-none Qwen3 macro deltas seeds[-.002223809,+.000725047,-.000558089]; MICRO[-.001801282,-.000549662,-.000889879], all3 better. vsbudget MICRO[-.001676495,-.000265192,-.000578702] allbetter. Qwen2.5 macroboth-none mean+.001002337 (worse), MICRO mean-.000254478 (better but seeds mixed). Qwen2.5 vsbudget macro+.001265065, MICRO+.000011435 (near0/mixed). Deduprefs/firstref similar macro signs. Public3219q:2300unique1ref,869unique2,50unique3.

analyze.py implemented/runs --partial successfully, last RESULTS contains8 endpoints; no complete3seed contrasts yet. It reads only COMPLETE endpoints. Aggregates micro/macro content CE; total CE; additive first/rest/EOS contributions with same total-token denominator; first/rest/content top1,top5,margin; per-question dedup COMPLETE answer-set NLL=-logsumexp[-contentNLL-EOSNLL], probability and teacher-exact-any. Different complete strings+EOS disjoint; not all semantic aliases, not greedyaccuracy. All 21 diagnostics x2splits x2controls x2models produce168 nominal descriptive seedtCIs, NOT confirmatory metric selection. Public lengthbins1-4,5-8,9-16,17+ and micro-macro covariance identity. CE/generation win/loss concordance. No changes old EM/F1/AVG. New analysis/postprocessing not manifestfrozen until final.

First seed internal-decomposition (preliminary only): Q3 both-none totalCE-.001371879 = firstcontrib-.001169723 +rest-.000584733 +EOS+.000382577. Q25 both-none total-.001720921 =first-.000563544 +rest-.000999474 +EOS-.000157904. Thus NOT an EOS-only gain. First seed public: Q3 both-none micro-.001801282, total-.001215343, first-.000929461 rest-.000675965 EOS+.000390083, answerSetNLL-.010686916. Q25 both-none micro+.000351272, total+.000246275, first+.000949049 rest-.000635971 EOS-.000066803, answerSetNLL-.002406121. Need all3 seeds for final interpretation; don't extrapolate.

Remaining after all jobs complete:
1. Close scheduler99539 viawrite_stdin.
2. Run D/analyze.py (no --partial), wait; inspect full RESULTS.
3. Run D/final_audit.py ->final_audit.log (unrun). It checks all reusedinputs+artifacts again, checkpoints, raw IDs/counts/old replay, decomposition identities, recomputed summary. Writes FINAL_AUDIT.json with errors/counts. If bug fix only this unfrozen postprocessor, preserve failure note if appropriate; never frozen input edits.
4. Run D/plot.py (unrun) ->figures/evaluation_decomposition.png/pdf, view_image. Three panels: publicmicrovsmacro seed CIs; internal/public first/rest/EOS additive means. Fix readability if needed before finalartifactmanifest.
5. Write FINAL_INTERPRETATION_ZH.md from actual full results, with corrected nuanced conclusions. Already EVALUATION_V2.md created and readable: proposed implemented metric/report contract, formulas, limitations, sources. Need final report and possibly reportwriter script; not written yet.
6. Freeze final artifacts in ARTIFACT_MANIFEST.json inclpostprocessing scripts/results/figures/report/audits, record versions, update RUN_HANDOFF top COMPLETE. Verify links.
7. Final Chinese selfcontained explaining actual verified insights, what updates warranted, what earlier claim needs qualification, officialgenconclusionunchanged. Link finalreport and EVALUATION_V2. User asked deeper; provide enough concrete metrics, not just plan. No more training/benchmarks to fish positives.

Previous B main generation: both-none/budget Q3 +.467454/+ .190987pp, Q25 -.082601/+ .001886pp. All4 seed95CIs cross0; Q3both-none all3positive vsbudgetseed43negative. All24 cap0. Publicdev used previously; no freshblindtest. Original Chinese dialogue CE gains reproducible over6modelseeds bothcontrols (.0054/.0069) but different task/tokenweighting. Need clarify previous macro answerCE vs tokenweighted dialogue/internal CE was not directly comparable; oldnumberscorrect but inferred inconsistency partlyweighting.

Webprimary docs browsed: https://huggingface.co/docs/transformers/chat_templating (template matches modelcontroltokens), https://github.com/ymcui/cmrc2018 (spanextractive/publicdev vs hidden test), https://docs.pytorch.org/docs/2.14/generated/torch.topk.html (ties indices not stable). EVALUATION_V2 alreadycites inline. Other browse lm-eval taskguide notused. No need more sources unless new technicalrecommendation. If final mentions sourcedexternalfact cite normalMarkdown primaryURL. Localresults citeartifactlinks.
