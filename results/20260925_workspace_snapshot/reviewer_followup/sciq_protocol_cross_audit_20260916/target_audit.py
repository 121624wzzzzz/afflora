"""Compare the prior two-position SFT objective with HF's standard shifted loss."""
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cross import BASE, CHAT, HERE, read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    source = BASE if args.model.endswith('_base') else CHAT
    cfg = read(source / 'models.json')[args.model]
    meta = read(source / 'DATA_AUDIT.json')['tokenization'][args.model]
    items = read(source / 'tokens' / f'{args.model}_train.json')[:8]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(cfg['path'], dtype=torch.bfloat16,
                                                attn_implementation='eager', local_files_only=True)
    model.requires_grad_(False)
    model = model.cuda().float().eval()
    seqs = [r['input_ids'] + [meta['label_ids'][r['gold']], meta['eos_id']] for r in items]
    lengths = torch.tensor(list(map(len, seqs)), device='cuda')
    width = max(map(len, seqs))
    ids = torch.tensor([s + [meta['pad_id']] * (width - len(s)) for s in seqs], device='cuda')
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device='cuda')
    labels = torch.full_like(ids, -100)
    for i, s in enumerate(seqs):
        labels[i, len(s) - 2:len(s)] = ids[i, len(s) - 2:len(s)]
    assert int((labels != -100).sum()) == 16
    with torch.inference_mode():
        full = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
        # Full input includes both supervised tokens, so the predicting positions
        # are L-3 and L-2. The old prompt+answer input has length L-1 and selects
        # (L-1)-2 and (L-1)-1, exactly the same positions.
        pos = torch.stack([lengths - 3, lengths - 2], dim=1)
        selected = full.logits[torch.arange(len(items), device='cuda')[:, None], pos].float()
        gold = torch.tensor([[meta['label_ids'][r['gold']], meta['eos_id']] for r in items], device='cuda')
        manual = F.cross_entropy(selected.reshape(-1, selected.shape[-1]), gold.reshape(-1))
        error = float(abs(full.loss - manual))
        assert error < 1e-5, error
    report = {'model': args.model, 'examples': 8, 'supervised_tokens_per_example': 2,
              'targets': 'answer letter, native source-study end token',
              'native_hf_shifted_loss': float(full.loss), 'two_position_loss': float(manual),
              'abs_loss_error': error, 'standard_shift_check_passed': True,
              'eos_id': meta['eos_id']}
    write(HERE / f'{args.model}_TARGET_AUDIT.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
