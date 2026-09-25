"""Replay the first training batch without optimizer updates; post-hoc diagnosis."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'qwen35_transfer_20260919'
sys.path.insert(0, str(ROOT))
import torch
from modeling import adapter_state, batch, build, digest, frozen_digest, loss
from run import preflight


def read(path):
    return json.loads(path.read_text())


def main():
    assert not (ROOT / 'SEAL.json').exists()
    device = int(os.environ['CUDA_VISIBLE_DEVICES'])
    result = subprocess.run(['nvidia-smi', f'--id={device}', '--query-gpu=memory.free',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    assert int(result.stdout.strip()) >= 50000, 'Insufficient memory for isolated diagnostic'
    name = 'search_trec50_hidden_both_c5_s7600'
    checkpoint = ROOT / 'checkpoints' / name
    spec = read(checkpoint / 'spec.json')
    original = read(checkpoint / 'INITIALIZATION.json')
    model, audit = build(spec)
    data = read(ROOT / 'tokens/trec50_qwen35_4b_base_train.json')
    pf = preflight(model, spec, data)
    assert digest(adapter_state(model)) == original['initialization_sha256']
    before = frozen_digest(model)
    assert before == original['frozen_before']
    order = read(checkpoint / 'TRAIN_ORDER.json')['indices'][:32]
    denominator = sum(len(data[i]['target_ids']) for i in order)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    measurements = []
    for case in ['boundaries_repeat1', 'boundaries_repeat2', 'without_boundary_hooks']:
        if case == 'without_boundary_hooks':
            for handle in model._boundary_handles:
                handle.remove()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        model.train()
        model.zero_grad(set_to_none=True)
        pieces = []
        for start in range(0, 32, 2):
            ids, mask, labels = batch([data[i] for i in order[start:start+2]])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, ce, gold = loss(model, ids, mask, labels)
                value = ce.sum() / denominator
            assert torch.isfinite(value)
            value.backward()
            pieces.append(float(value.detach()))
        assert all(parameter.grad is None for name, parameter in model.named_parameters()
                   if not parameter.requires_grad)
        assert digest(adapter_state(model)) == original['initialization_sha256']
        measurement = {'case': case, 'first_batch_loss': sum(pieces), 'microbatch_contributions': pieces}
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)
    model.zero_grad(set_to_none=True)
    assert frozen_digest(model) == before
    report = {
        'at': datetime.now().astimezone().isoformat(), 'gpu': device,
        'status': 'diagnostic_completed', 'optimizer_updates': 0,
        'source_run': name, 'same_initial_parameters': True, 'frozen_weights_unchanged': True,
        'same_first_32_training_examples': True, 'preflight': pf,
        'recorded_source_first_loss': read(checkpoint / 'TRAINING.json')['history'][0]['loss'],
        'measurements': measurements,
        'scope': 'Post-hoc initial-forward/backward replay only; does not replace any fit or explain later accuracy causally.',
    }
    destination = ROOT / 'INITIAL_FORWARD_REPLAY.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(destination), 'status': report['status']}), flush=True)


if __name__ == '__main__':
    main()
