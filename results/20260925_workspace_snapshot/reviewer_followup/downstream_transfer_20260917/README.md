# ANLI R1 / WikiSQL 下游任务扩展

本轮60次正式适配训练、4个Base对照、4次两步冒烟均已完成。两个模型、三个适配方案、五个配对种子。当前结论和限制以 [最终中文报告](FINAL_INTERPRETATION_ZH.md) 为准。

- [五种子测试表](TEST_RESULTS_ZH.md) / [完整统计](TEST_ANALYSIS.json)
- [开发集结果](DEV_RESULTS_ZH.md)
- [运行前方案](PROTOCOL.md) / [方法与限制](METHOD_AND_LIMITS_ZH.md)
- [内容与格式诊断](CONTENT_DIAGNOSTICS.json)
- [等参数组与普通组的事后核查](BUDGET_CONTROL_DIAGNOSTIC.json)
- [参数审计](PARAMETER_AUDIT.json) / [输出重评分](RESULT_AUDIT.json) / [训练编码和顺序审计](TRAINING_INPUT_AUDIT.json)
- [官方数据来源](DATA_SOURCES.json) / [复用文件校验](SOURCE_REUSE.json)

WikiSQL两个模型的叠加组均在五个种子中超过普通与严格等参数LoRA，本轮校正种子区间均高于零；等参比较的下界接近零，不能外推高度稳健。ANLI未见稳定叠加收益，完整保留。

`COMPLETION.json` 与 `FINAL_AUDIT.json` 表示完成，`ARTIFACT_MANIFEST.json` 记录封存文件哈希。封存后不要在本目录重跑 prepare/pipeline/analyze/write_interpretation/plot/seal 等写入脚本。只读核验使用：

```bash
/home/wz/anaconda3/envs/torch24/bin/python verify_seal.py
```

复现实验应复制到新的目录，保留固定源代码和协议，验证模型与数据哈希后再生成新的运行产物，不覆盖本轮记录。
