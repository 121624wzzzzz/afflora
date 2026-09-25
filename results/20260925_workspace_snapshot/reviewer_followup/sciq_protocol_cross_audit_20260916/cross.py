import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'standalone_base_native_sciq_20260916'
CHAT = ROOT / 'standalone_single_boundary_sciq_20260915'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def task(row):
    return ('Answer the science question. Select the best option and respond with only '
            'its letter (A, B, C, or D).\n\nQuestion: ' + row['question'] + '\n\n'
            + '\n'.join(f'{c}. {v}' for c, v in zip('ABCD', row['choices'])))


def prompt(row, chat_tok, condition):
    content = task(row)
    if condition == 'plain_answer':
        return content + '\n\nAnswer:\n'
    if condition == 'plain_no_cue':
        return content + '\n\n'
    text = chat_tok.apply_chat_template([{'role': 'user', 'content': content}],
                                        tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
    if condition == 'chat_answer':
        text += 'Answer:\n'
    return text


def leading_letter(text):
    match = re.match(r'^\s*([ABCD])(?=$|[\s.,:;)\]])', text)
    return None if match is None else 'ABCD'.index(match.group(1))


def summarize(predictions):
    clean = [p for p in predictions if not p['ambiguous_gold']]
    assert len(clean) == 998
    return {'n': len(clean), 'candidate_accuracy': 100 * sum(p['correct'] for p in clean) / len(clean),
            'unrestricted_accuracy': 100 * sum(p['unrestricted_correct'] for p in clean) / len(clean),
            'valid_label_rate': 100 * sum(p['valid_label'] for p in clean) / len(clean),
            'prediction_counts': dict(Counter(p['prediction'] for p in clean))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    name = args.model
    family = name.rsplit('_', 1)[0]
    source = BASE if name.endswith('_base') else CHAT
    cfg = read(source / 'models.json')[name]
    chat_cfg = read(CHAT / 'models.json')[family + '_chat']
    for c in [cfg, chat_cfg]:
        for path, digest in c['files'].items():
            assert sha(path) == digest, path
    tok = AutoTokenizer.from_pretrained(cfg['path'], local_files_only=True)
    chat_tok = AutoTokenizer.from_pretrained(chat_cfg['path'], local_files_only=True)
    data = rows(BASE / 'data/test.jsonl')
    assert data == rows(CHAT / 'data/test.jsonl')
    saved_tokens = read(source / 'tokens' / f'{name}_test.json')
    original_condition = 'plain_answer' if source == BASE else 'chat'
    reference = next((source / 'checkpoints').glob(f'reference_{name}_*/test_predictions.jsonl'))
    old = [r for r in rows(reference) if r['rotation'] == 0]
    labels = [tok.encode(c, add_special_tokens=False) for c in 'ABCD']
    assert labels == [[32], [33], [34], [35]]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(cfg['path'], local_files_only=True,
                                                dtype=torch.bfloat16, attn_implementation='eager')
    model.requires_grad_(False)
    model = model.cuda().float().eval()
    assert not any(p.requires_grad for p in model.parameters())
    assert not any('lora_' in n or 'standalone_affine' in n for n, _ in model.named_parameters())
    conditions, examples = {}, {}
    with torch.inference_mode():
        for condition in ['plain_answer', 'plain_no_cue', 'chat', 'chat_answer']:
            inputs = []
            for i, row in enumerate(data):
                text = prompt(row, chat_tok, condition)
                ids = tok.encode(text, add_special_tokens=False)
                assert ids == chat_tok.encode(text, add_special_tokens=False)
                if condition == original_condition:
                    assert ids == saved_tokens[i]['input_ids']
                inputs.append(ids)
            examples[condition] = {'text': tok.decode(inputs[0]), 'ids': inputs[0]}
            pp = []
            start = time.monotonic()
            for begin in range(0, len(data), 16):
                sequences = inputs[begin:begin + 16]
                width = max(map(len, sequences))
                ids = torch.tensor([[tok.pad_token_id] * (width - len(s)) + s for s in sequences], device='cuda')
                mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences], device='cuda')
                pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                               use_cache=False, logits_to_keep=1).logits[:, -1].float()
                scores = logits[:, [32, 33, 34, 35]]
                for j, row in enumerate(data[begin:begin + 16]):
                    pred = int(scores[j].argmax())
                    token = int(logits[j].argmax())
                    pp.append({k: row[k] for k in ['id', 'gold', 'ambiguous_gold']} |
                              {'prediction': pred, 'correct': pred == row['gold'],
                               'label_logits': scores[j].cpu().tolist(), 'unrestricted_token': token,
                               'unrestricted_correct': token == 32 + row['gold'], 'valid_label': token in [32, 33, 34, 35]})
            path = HERE / f'{name}_{condition}.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in pp))
            result = summarize(pp) | {'seconds': time.monotonic() - start, 'sha256': sha(path)}
            if condition == original_condition:
                assert [r['id'] for r in pp] == [r['id'] for r in old]
                error = max(abs(a - b) for p, q in zip(pp, old) for a, b in zip(p['label_logits'], q['label_logits']))
                flips = sum(p['prediction'] != q['prediction'] for p, q in zip(pp, old))
                result.update(original_reference_max_logit_error=error, original_prediction_flips=flips)
                assert error < 5e-4 and flips == 0, (error, flips)
            conditions[condition] = result
            write(HERE / f'{name}_PROGRESS.json', conditions)
            print(name, condition, result, flush=True)
    generation = rows(reference.parent / 'test_generation.jsonl')
    clean = [r for r in generation if not r['ambiguous_gold']]
    gen_summary = {'n': len(clean),
                   'leading_letter_accuracy': 100 * sum(leading_letter(r['text']) == r['gold'] for r in clean) / len(clean),
                   'leading_letter_validity': 100 * sum(leading_letter(r['text']) is not None for r in clean) / len(clean),
                   'strict_accuracy': 100 * sum(r['strict_correct'] for r in clean) / len(clean),
                   'length_cap_rate': 100 * sum(r['hit_length_cap'] for r in clean) / len(clean),
                   'examples': [{k: r[k] for k in ['id', 'gold', 'text', 'strict_correct']} for r in clean[:8]]}
    write(HERE / f'{name}_PROMPTS.json', examples)
    write(HERE / f'{name}_RESULTS.json', {'model': name, 'conditions': conditions,
                                        'original_generation_reparsed': gen_summary,
                                        'model_files_verified': cfg['files'],
                                        'independent_full_forward': True, 'adapter_installed': False})


if __name__ == '__main__':
    main()
