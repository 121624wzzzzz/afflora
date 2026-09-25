# 项目盘安全去重

完成时间：2026-09-24T00:06:31.651738+08:00

- 120份重复初始checkpoint：释放6.97 GiB。
- 7个重复预训练权重分片：释放17.37 GiB，涉及Qwen2.5-0.5B/3B和Qwen3-1.7B/4B。
- 合计释放24.33 GiB；项目盘当前可用约43.54 GiB。

逐份SHA256校验后，将相同字节的不可变文件改为硬链接共享存储；替换后再次校验。所有原路径和内容保留，既有审计哈希不变。没有删除任何唯一模型、checkpoint、数据集、预测或报告。硬链接文件应继续视为冻结文件；如以后需要就地修改某份，应先复制为独立文件。

详见DEDUP_RESULT.json、MODEL_DEDUP_RESULT.json。新扩尺寸实验的checkpoint/预测另放/home/wz/experiment_artifacts/tied_model_expansion_20260923，项目内保留链接。
