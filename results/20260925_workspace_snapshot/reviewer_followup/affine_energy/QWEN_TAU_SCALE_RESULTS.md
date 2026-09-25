# Qwen 3B/4B affine-energy tau sweep (input/shared/output)

Updated: 2026-07-13T21:43:41+08:00

All new rows use MetaMathQA-40K, one epoch, effective batch 16, hidden LoRA r4, A-LoRA r16 scale 1, seed 42, and lambda=100.

`shared` is the tied, mergeable parameterization: one affine map is shared by
the input embedding and transposed LM-head path, exactly matching a single
update merged into the tied vocabulary matrix. It is not the decoupled
input/output parameterization. `output` uses a multiplicative LM-head adapter
only and disables the softmax-invariant output bias; it is therefore also
mergeable into `lm_head.weight`. The reported rho is measured on the adapted
path (centered output rho for output-only).

## qwen25_3b

| topology | tau | measured rho | MATH | delta vs hidden | GSM8K | delta vs hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden r4 | - | - | 41.9820% | - | 78.0895% | - |
| input | 0.0500 | 0.04232661 | 42.0621% | +0.0801 pp | 78.0136% | -0.0758 pp |
| input | 0.0250 | 0.02322139 | 41.8619% | -0.1201 pp | 77.9378% | -0.1516 pp |
| input | 0.0125 | 0.01221454 | 41.7818% | -0.2002 pp | 79.1509% | +1.0614 pp |
| input | 0.00625 | 0.00613273 | 42.0020% | +0.0200 pp | 77.4071% | -0.6823 pp |
| shared | unconstrained | 0.05190600 | 42.1421% | +0.1602 pp | 77.7862% | -0.3033 pp |
| shared | 0.0500 | 0.04184839 | 42.0821% | +0.1001 pp | 78.0136% | -0.0758 pp |
| shared | 0.0250 | 0.02302829 | 42.2823% | +0.3003 pp | 78.1653% | +0.0758 pp |
| shared | 0.0125 | 0.01222545 | 42.3624% | +0.3804 pp | 78.6960% | +0.6065 pp |
| shared | 0.00625 | 0.00617909 | 42.0220% | +0.0400 pp | 78.0895% | +0.0000 pp |
| output | 0.0500 | 0.00882606 | 42.1021% | +0.1201 pp | 78.1653% | +0.0758 pp |
| output | 0.0250 | 0.00872387 | 41.9219% | -0.0601 pp | 78.7718% | +0.6823 pp |
| output | 0.0125 | 0.00873643 | 41.7818% | -0.2002 pp | 78.1653% | +0.0758 pp |
| output | 0.00625 | 0.00622224 | 42.4224% | +0.4404 pp | 78.6202% | +0.5307 pp |

## qwen3_4b

| topology | tau | measured rho | MATH | delta vs hidden | GSM8K | delta vs hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden r4 | - | - | 48.0280% | - | 84.3821% | - |
| input | 0.0500 | 0.04858436 | 47.9880% | -0.0400 pp | 84.5337% | +0.1516 pp |
| input | 0.0250 | 0.02488442 | 48.2282% | +0.2002 pp | 84.2305% | -0.1516 pp |
| input | 0.0125 | 0.01249769 | 48.3083% | +0.2803 pp | 84.4579% | +0.0758 pp |
| input | 0.00625 | 0.00624457 | 47.5275% | -0.5005 pp | 84.3063% | -0.0758 pp |
| shared | unconstrained | 0.10358613 | 47.9079% | -0.1201 pp | 84.3821% | +0.0000 pp |
| shared | 0.0500 | 0.04897605 | 48.3283% | +0.3003 pp | 84.3821% | +0.0000 pp |
| shared | 0.0250 | 0.02497865 | 48.1682% | +0.1401 pp | 84.4579% | +0.0758 pp |
| shared | 0.0125 | 0.01249816 | 47.6877% | -0.3403 pp | 84.4579% | +0.0758 pp |
| shared | 0.00625 | 0.00624892 | 47.9279% | -0.1001 pp | 84.0788% | -0.3033 pp |
| output | 0.0500 | 0.00814662 | 48.4885% | +0.4605 pp | 84.2305% | -0.1516 pp |
| output | 0.0250 | 0.00815237 | 48.1081% | +0.0801 pp | 83.9272% | -0.4549 pp |
| output | 0.0125 | 0.00816126 | 48.1281% | +0.1001 pp | 84.8370% | +0.4549 pp |
| output | 0.00625 | 0.00623145 | 48.3483% | +0.3203 pp | 84.4579% | +0.0758 pp |
