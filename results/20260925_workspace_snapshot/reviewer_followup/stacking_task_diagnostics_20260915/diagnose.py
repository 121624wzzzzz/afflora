"""Post-hoc read-only diagnostics; preserve every completed experiment."""
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
CHAT=HERE.parent/'stacking_claim_checks_20260914/chat'
SOURCES={}
EXPECTED=json.loads((CHAT/'ANALYSIS_SOURCES.json').read_text())


def read(path,lines=False):
    data=path.read_bytes();h=hashlib.sha256(data).hexdigest()
    if str(path) in EXPECTED:assert h==EXPECTED[str(path)],path
    SOURCES[str(path)]=h
    return [json.loads(x) for x in data.decode().splitlines()] if lines else json.loads(data)


def cjk_fraction(text):
    letters=sum(c.isalpha() for c in text)
    return len(re.findall(r'[\u3400-\u9fff]',text))/max(letters,1)


def language_stats(texts):
    values=[cjk_fraction(x) for x in texts]
    return {'rows':len(values),'rows_any_cjk':sum(v>0 for v in values),
            'rows_cjk_fraction_gt_10pct':sum(v>.1 for v in values),
            'rows_cjk_fraction_gt_50pct':sum(v>.5 for v in values)}


def main():
    state=read(CHAT/'state.json');audit=read(CHAT/'FINAL_AUDIT.json')
    assert audit['status']=='passed' and audit['new_training_cells']==24
    data={s:read(CHAT/'data'/f'{s}.jsonl',True) for s in ['train','dev','test','ifeval']}
    result={'created_at':datetime.now().astimezone().isoformat(),
        'scope':'Post-hoc diagnostics. Character heuristics are not language identification. Cap partitions are descriptive partitions defined by both outputs, not causal or unbiased subgroup effects. Instruction categories are exploratory, not newly confirmed benchmarks.',
        'data_language':{},'endpoints':{},'contrasts':{}}
    for split,rows in data.items():
        if split=='ifeval':texts=[r['prompt'] for r in rows]
        else:texts=['\n'.join(c['content'] for c in r['conversations']) for r in rows]
        result['data_language'][split]=language_stats(texts)
        if split!='ifeval':
            result['data_language'][split]['assistant_targets']=language_stats([
                '\n'.join(c['content'] for c in r['conversations'] if c['role']=='assistant') for r in rows])
    obs={}
    for j in state['matrix']+state['references']:
        assert state['jobs'][j['name']]['status']=='complete'
        base=CHAT/'ifeval'/j['name']
        ce=read(CHAT/'reports'/f"{j['name']}.test.json")
        scores=read(base/'scores.json');responses=read(base/'responses.jsonl',True)
        assert scores['responses_sha256']==SOURCES[str(base/'responses.jsonl')]
        assert len(responses)==len(scores['per_example'])==len(data['ifeval'])==541
        assert [r['key'] for r in responses]==[r['key'] for r in data['ifeval']]
        assert [r['key'] for r in responses]==[r['key'] for r in scores['per_example']]
        assert [r['instruction_id_list'] for r in responses]==[r['instruction_id_list'] for r in data['ifeval']]
        assert len(ce['per_example'])==len(data['test'])==1000
        arm='reference' if j.get('reference') else j['placement']
        obs[j['model'],arm,j['seed']]={'ce':ce,'scores':scores,'responses':responses}
        non_language=[r['response'] for r in responses if not any(x.startswith('language:') for x in r['instruction_id_list'])]
        result['endpoints'][j['name']]={'model':j['model'],'arm':arm,'seed':j['seed'],
            'ce':ce['avg_ce'],'strict':scores['strict_prompt_accuracy'],
            'strict_instruction_accuracy':scores['strict_instruction_accuracy'],
            'cap_rate':scores['cap_hit_rate'],'mean_tokens':scores['mean_generated_tokens'],
            'response_language':language_stats([r['response'] for r in responses]),
            'response_language_without_explicit_language_instruction':language_stats(non_language)}
    for model in sorted({j['model'] for j in state['matrix']}):
        for treatment,control in [('both','none'),('both','hidden_budget'),('output','none')]:
            cells=[];families=defaultdict(list)
            for seed in [42,43,44]:
                a,b=obs[model,treatment,seed],obs[model,control,seed]
                paired=list(zip(a['scores']['per_example'],b['scores']['per_example'],a['responses'],b['responses']))
                assert all(x['key']==y['key']==rx['key']==ry['key'] for x,y,rx,ry in paired)
                assert all(len(rx['instruction_id_list'])==len(x['strict_instructions'])==len(y['strict_instructions']) for x,y,rx,ry in paired)
                wins=sum(x['strict'] and not y['strict'] for x,y,_,_ in paired)
                losses=sum(y['strict'] and not x['strict'] for x,y,_,_ in paired)
                partitions={}
                for capped in [False,True]:
                    pairs=[(x,y) for x,y,rx,ry in paired if (rx['hit_token_cap'] or ry['hit_token_cap'])==capped]
                    w=sum(x['strict'] and not y['strict'] for x,y in pairs)
                    l=sum(y['strict'] and not x['strict'] for x,y in pairs)
                    partitions['either_capped' if capped else 'neither_capped']={'n':len(pairs),'wins':w,'losses':l,'net_points_on_full_541':100*(w-l)/541}
                for x,y,rx,ry in paired:
                    for instruction,left,right in zip(rx['instruction_id_list'],x['strict_instructions'],y['strict_instructions']):
                        families[instruction.split(':')[0]].append(int(left)-int(right))
                ca,cb=a['ce']['per_example'],b['ce']['per_example']
                assert [(r['record_id'],r['token_count']) for r in ca]==[(r['record_id'],r['token_count']) for r in cb]
                ce_gain=b['ce']['avg_ce']-a['ce']['avg_ce']
                cells.append({'seed':seed,'wins':wins,'losses':losses,'net_prompts':wins-losses,
                    'score_discordant_prompts':wins+losses,'exact_response_changes':sum(x['response']!=y['response'] for x,y in zip(a['responses'],b['responses'])),
                    'cap_partition':partitions,'ce_reduction':ce_gain,
                    'gold_token_geometric_probability_relative_gain_pct':100*math.expm1(ce_gain),
                    'test_examples_lower_ce':sum(x['mean_ce']<y['mean_ce'] for x,y in zip(ca,cb)),
                    'test_examples_higher_ce':sum(x['mean_ce']>y['mean_ce'] for x,y in zip(ca,cb))})
            result['contrasts'][f'{model}: {treatment} - {control}']={'seeds':cells,
                'mean_ce_reduction':mean(c['ce_reduction'] for c in cells),
                'mean_gold_token_geometric_probability_relative_gain_pct':mean(c['gold_token_geometric_probability_relative_gain_pct'] for c in cells),
                'mean_net_prompts':mean(c['net_prompts'] for c in cells),
                'mean_score_discordant_prompts':mean(c['score_discordant_prompts'] for c in cells),
                'instruction_families':{k:{'instruction_seed_observations':len(v),'delta_accuracy_pp':100*mean(v)} for k,v in sorted(families.items())}}
    SOURCES[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['source_sha256']=SOURCES
    (HERE/'DIAGNOSTICS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['data_language','contrasts']},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
