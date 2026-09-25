import argparse
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cross import BASE, CHAT, HERE, leading_letter, read, rows, sha, task, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    assert args.shard in range(4)
    name = 'qwen3_06b_chat'
    cfg = read(CHAT / 'models.json')[name]
    for path, digest in cfg['files'].items():
        assert sha(path) == digest
    tok = AutoTokenizer.from_pretrained(cfg['path'], local_files_only=True)
    assert tok.encode('</think>', add_special_tokens=False) == [151668]
    data = rows(BASE / 'data/test.jsonl')[args.shard * 250:(args.shard + 1) * 250]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(20260916 + args.shard)
    model = AutoModelForCausalLM.from_pretrained(cfg['path'], local_files_only=True,
                                                dtype=torch.bfloat16, attn_implementation='eager')
    model.requires_grad_(False)
    model = model.cuda().float().eval()
    assert not any('lora_' in n or 'standalone_affine' in n for n, _ in model.named_parameters())
    stops = model.generation_config.eos_token_id
    stops = stops if isinstance(stops, list) else [stops]
    output = HERE / f'thinking_shard{args.shard}.jsonl'
    assert not output.exists(), output
    start = time.monotonic()
    with torch.inference_mode(), output.open('w') as out:
        for begin in range(0, len(data), 16):
            items = data[begin:begin + 16]
            sequences = [tok.apply_chat_template([{'role': 'user', 'content': task(r)}],
                                                 tokenize=True, return_dict=False,
                                                 add_generation_prompt=True, enable_thinking=True)
                         for r in items]
            width = max(map(len, sequences))
            ids = torch.tensor([[tok.pad_token_id] * (width - len(s)) + s for s in sequences], device='cuda')
            mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences], device='cuda')
            generated = model.generate(input_ids=ids, attention_mask=mask, do_sample=True,
                                       temperature=.6, top_p=.95, top_k=20, min_p=0.,
                                       repetition_penalty=1., max_new_tokens=2048,
                                       use_cache=True, pad_token_id=tok.pad_token_id,
                                       eos_token_id=stops)[:, width:].cpu().tolist()
            for item, seq in zip(items, generated):
                stop = next((i for i, token in enumerate(seq) if token in stops), None)
                if stop is not None:
                    seq = seq[:stop + 1]
                content = seq if stop is None else seq[:-1]
                closes = [i for i, token in enumerate(content) if token == 151668]
                final = tok.decode(content[closes[-1] + 1:], skip_special_tokens=False) if closes else ''
                pred = leading_letter(final) if closes else None
                record = {k: item[k] for k in ['id', 'gold', 'ambiguous_gold']} | {
                    'shard': args.shard, 'seed': 20260916 + args.shard,
                    'generated_ids': seq, 'text': tok.decode(content, skip_special_tokens=False),
                    'thinking_closed': bool(closes), 'final_text': final,
                    'prediction': pred, 'correct': pred == item['gold'],
                    'strict_correct': final.strip() == 'ABCD'[item['gold']] if closes else False,
                    'valid_answer': pred is not None, 'terminated': stop is not None,
                    'hit_length_cap': stop is None and len(seq) >= 2048,
                    'generated_tokens': len(seq)}
                out.write(__import__('json').dumps(record, ensure_ascii=False) + '\n')
            out.flush()
            progress = {'shard': args.shard, 'rows': min(begin + 16, len(data)), 'total': len(data),
                        'seconds': time.monotonic() - start}
            write(HERE / f'thinking_shard{args.shard}_PROGRESS.json', progress)
            print(progress, flush=True)
    write(HERE / f'thinking_shard{args.shard}_DONE.json', {'rows': len(data), 'sha256': sha(output),
                                                       'seconds': time.monotonic() - start,
                                                       'stop_ids': stops})


if __name__ == '__main__':
    main()
