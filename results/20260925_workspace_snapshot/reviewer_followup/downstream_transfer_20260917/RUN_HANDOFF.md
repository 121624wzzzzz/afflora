# 完成状态

2026-09-17：60次正式适配训练、4个Base对照与4次冒烟均成功，五种子完整结果和解释见 FINAL_INTERPRETATION_ZH.md。调度器已退出，exec session 26264 已关闭，无本轮GPU任务待运行。

WikiSQL两个模型的四项主比较均值为正，五种子方向一致，本轮八项比较校正t区间高于零，但等参比较下界接近零。ANLI未见稳定叠加收益。

数据冻结 SHA256：8d30a716cf0386f0353ab7a33e271bd2106d58136e756d567559b6cfe8ca0beb。
代码冻结 SHA256：d516b7c37738c9270fec4d46c76244ba01df65a23a26f53ced414ec269a15767。

已复核104,960条输出、39,947个官方执行器预测比较、68份参数范围、14,752条token记录、8,192条训练编码和60份完整训练顺序。原始模型14文件重新哈希核对，无旧训练adapter复用。

运行历史、逐步损失、原始输出和初始化均保存在 scheduler.log、logs、checkpoints、evaluations。ANLI训练前排除与评测重复前提的修订记录在 PREFLIGHT_REVISION.json，原始抽样保存在 preflight_v1；修订发生在模型运行前。

用户关于等参数组小幅下降的核查见 BUDGET_CONTROL_DIAGNOSTIC.json：ANLI1.5B等参－普通平均−0.22pp，种子方向不一致，共有初始化/顺序/缩放检查通过。此为事后描述，不替代预定主比较。

最后新增的内容诊断、解释与作图脚本不改变运行前冻结的评分/训练/统计代码。封存后仅运行 verify_seal.py 作只读检查；请在新目录开展新实验。旧 posttraining_tasks_grounded_20260916 继续保持封存。
