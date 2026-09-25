"""Exploratory output-form diagnostics; do not replace frozen primary metrics."""
from statistics import mean
from common import HERE, now, read, rows, sha, write
from evaluate import official

def main():
    data=rows(HERE/'data/cmrc_eval.jsonl');manifest=read(HERE/'manifest.json')
    outputs={};stats=[];sources={str(HERE/'data/cmrc_eval.jsonl'):sha(HERE/'data/cmrc_eval.jsonl')}
    normalized=[{'answers':[official.remove_punctuation(a) for a in r['answers']],
        'context':official.remove_punctuation(r['context'])} for r in data]
    for j in manifest['endpoints']:
        folder=HERE/'outputs'/j['name'];path=folder/'cmrc_generation.jsonl';responses=rows(path)
        assert len(responses)==len(data) and [r['id'] for r in responses]==[r['id'] for r in data]
        sources[str(path)]=sha(path)
        stats_path=folder/'cmrc_scores.json';sources[str(stats_path)]=sha(stats_path)
        detail=read(stats_path)['per_example'];derived=[]
        for r,g,s in zip(responses,normalized,detail):
            text=official.remove_punctuation(r['response'])
            derived.append({**r,'contains_reference_string':any(a and a in text for a in g['answers']),
                'entire_response_is_context_substring':bool(text) and text in g['context'],
                'normalized_characters':len(text),'answer_ce':s['answer_ce']})
        arm='reference' if j.get('reference') else j['placement']
        outputs[j['model'],arm,j['seed']]=derived
        stats.append({'model':j['model'],'arm':arm,'seed':j['seed'],
            **{k:mean(r[k] for r in derived) for k in ['contains_reference_string','entire_response_is_context_substring','normalized_characters','generated_tokens','hit_token_cap','em','f1','answer_ce']}})
    comparisons=[]
    for model in sorted({j['model'] for j in manifest['endpoints']}):
        for control in ['none','hidden_budget']:
            for seed in [42,43,44]:
                left=outputs[model,'both',seed];right=outputs[model,control,seed]
                wins=[(a,b) for a,b in zip(left,right) if a['em'] and not b['em']]
                losses=[(a,b) for a,b in zip(left,right) if b['em'] and not a['em']]
                comparisons.append({'model':model,'control':control,'seed':seed,'em_wins':len(wins),'em_losses':len(losses),
                    'em_wins_where_control_already_contains_reference':sum(b['contains_reference_string'] for a,b in wins),
                    'em_losses_where_both_still_contains_reference':sum(a['contains_reference_string'] for a,b in losses),
                    'em_wins_where_control_lacks_reference_string':sum(not b['contains_reference_string'] for a,b in wins),
                    'em_losses_where_both_lacks_reference_string':sum(not a['contains_reference_string'] for a,b in losses),
                    'em_win_mean_token_length_change':mean(a['generated_tokens']-b['generated_tokens'] for a,b in wins) if wins else None,
                    'em_loss_mean_token_length_change':mean(a['generated_tokens']-b['generated_tokens'] for a,b in losses) if losses else None,
                    'questions_lower_answer_ce':sum(a['answer_ce']<b['answer_ce'] for a,b in zip(left,right)),
                    'questions_higher_answer_ce':sum(a['answer_ce']>b['answer_ce'] for a,b in zip(left,right))})
    write(HERE/'GENERATION_DIAGNOSTICS.json',{'created_at':now(),
        'scope':'Exploratory, designed after seeing seed 42. Reference substring presence does not establish semantic correctness (negations and irrelevant mentions can contain an answer). This is not a substitute metric, causal decomposition, or confirmed style-only explanation.',
        'endpoints':stats,'comparisons':comparisons,'source_sha256':sources})
    print('Exploratory generation diagnostics saved.')

if __name__=='__main__':main()
