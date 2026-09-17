# Frozen ct_dma TrAISformer checkpoint

The accepted corrected-reference checkpoint is stored locally as
`traisformer.pt`. Model binaries are intentionally excluded from ordinary Git
because this state dictionary is 230,196,929 bytes.

- SHA-256: `227a7cf42d7ab587d914172c2bba259e2c246d634e8e4f7b179e925c34b2b0e2`
- Training run: `20260811T095837Z-c8ef666d-blurfix`
- Best validation epoch: 9 (zero-based)
- Best validation loss: `13.508481979370117`

The same state is retained at
`runs/20260811T095837Z-c8ef666d-blurfix/full_training/best_validation_state.pt`.
Both copies must have the SHA-256 above before evaluation. Store or distribute
the binary through an artifact store or Git LFS; do not silently replace it in
place.

The plain-PyTorch baseline in `models/baselines/traisformer.py` preserves all
147 state-dict names. Evaluate it through the shared command documented in the
root README; the old Lightning pipeline is not required.
