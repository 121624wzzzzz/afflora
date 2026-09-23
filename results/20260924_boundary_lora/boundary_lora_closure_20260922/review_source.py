from common import *
import csv,io
old=Path(read(HERE/'SOURCE.json')['study']);r=read(old/'RESULTS.json');state=read(old/'STATE.json');groups=r['groups']
lines=['# 现有队列审阅与补充决策',f"记录时间：{now()}；源结果更新时间：{r['at']}。",f"源队列：{state['stage']}，{len(state['done'])}/684已完成，失败{len(state['failed'])}。",'', '## 调参后确认（只列完整的配对五种子）','|模型|任务|普通LoRA|aLoRA|差值|胜/平/负|','|---|---|---:|---:|---:|---|']
for a in groups:
 if a['stage']!='confirmation' or a['method']!='affine' or a['n']!=5:continue
 b=next((b for b in groups if b['stage']=='confirmation' and b['model']==a['model'] and b['task']==a['task'] and b['method']=='vocab' and b['n']==5),None)
 if not b:continue
 assert set(a['seeds'])==set(b['seeds']);ds=[a['seeds'][k]-b['seeds'][k] for k in a['seeds']]
 lines.append(f"|{a['model']}|{a['task']}|{b['mean']:.4f}|{a['mean']:.4f}|{a['mean']-b['mean']:+.4f}|{sum(d>0 for d in ds)}/{sum(d==0 for d in ds)}/{sum(d<0 for d in ds)}|")
lines += ['','## 审阅结论与范围','1. 现有确认是混合结果，不能写普遍性能优势；接近分数也不等于已通过非劣性检验。','2. 小尺寸tied的参数优势是精确计数事实；任务效果优势仍需同等验证集搜索。补Qwen3-0.6B和Qwen2.5-1.5B，共享r16 aLoRA对共享r1普通LoRA。','3. 单侧U仍缺搜索对照。补Qwen2.5-7B、Qwen3-8B、Llama3.1-8B两任务；保留Llama/WikiSQL这个近乎无收益条件，不能只选正条件。','4. 旧确认批次补40次同种子H-only，新确认批次每个模型任务补5个H-only。H固定2e-4，只比较边界增量；不声称H已经充分调参。','5. 新搜索每种方法4个边界LR倍率[1/16,1/4,1,4]×2个dev-only种子。前一轮多次最低候选胜出，统一加低候选；不是只为某方法调优。','6. 新确认采用远离历史编号的独立训练种子；测试集仍是既有测试集，不能称全新独立任务泛化。','7. 测实际训练显存/时间、checkpoint字节，并在同卡顺序微基准中比较H/共享aLoRA/共享普通LoRA；运行时共租争用会单独标注。合并正确性验证后再测推理。','8. 不扩模型族，不因测试正负追加seed；本轮不做rank/学习率全部交叉扫描。','', '## 留存限制','旧实验初始adapter张量快照当前缺失；初始化哈希与逐次审计保留。最终checkpoint、预测、训练与评分证据重新核验；不能称所有原始字节均已重新核验。详见SOURCE_AUDIT.json。']
(HERE.parent/'REVIEW_ZH.md').write_text('\n'.join(lines)+'\n')
fields=['stage','model','task','method','placement','rank','bias','n','mean','reused'];out=io.StringIO();w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
for g in groups:w.writerow({k:g[k] for k in fields})
(HERE.parent/'SOURCE_RESULTS.csv').write_text(out.getvalue())
write(HERE.parent/'SOURCE_PAIRED_CONTRASTS.json',read(old/'PAIRED_CONTRASTS.json'))
print('review and complete source tables written')
