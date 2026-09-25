# 当前实验

2026-09-17：扩展到 ANLI R1 与 WikiSQL，两模型、三种适配方案、五种子，共60次正式训练及4个Base对照。另有4个两步冒烟。

完整设计见 PROTOCOL.md。基础权重复用但14个文件已重新哈希核验，旧实验封存清单和复用源码也已核验；不复用已拟合adapter。

数据冻结 SHA256：8d30a716cf0386f0353ab7a33e271bd2106d58136e756d567559b6cfe8ca0beb。
代码冻结 SHA256：d516b7c37738c9270fec4d46c76244ba01df65a23a26f53ced414ec269a15767。

运行：Python /home/wz/anaconda3/envs/torch24/bin/python；pipeline.py → scheduler.log。当前 exec session 26264。仅使用GPU1–6，0/7已有其他作业。调度器在失败时停止发新任务，等待已运行任务结束后报错；重启不覆盖不完整目录。

只读查看：运行 monitor.py，或查看 SMOKE_STATE.json / FORMAL_STATE.json；单个训练/评测进度在 checkpoints 和 evaluations。pipeline.py 最后自动执行参数审计、全部输出重评分及统计。看到 EXPERIMENT_COMPLETE.json 后仍需解释结果、更新文档并封存。

ANLI 官方数据跨split复用前提；训练前排除 dev/test 前提，原始抽样保存在 preflight_v1，修订记录 PREFLIGHT_REVISION.json。6255条候选再去重为6252条，选2048训练。dev/test各1000条。WikiSQL train2048/dev256/test1024，全部标准查询执行与官方核对通过。

旧目录 posttraining_tasks_grounded_20260916 不得写入或重新运行报告脚本。
