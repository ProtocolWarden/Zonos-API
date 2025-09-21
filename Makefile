.PHONY: docker-sanity

docker-sanity:
	docker run --rm -it zonos-tts-api:test python - <<'PY'
import importlib.util, torch
print('Torch:', torch.__version__, torch.version.cuda, 'CUDA avail:', torch.cuda.is_available())
for m in ('selective_scan_cuda','mamba_ssm','flash_attn','causal_conv1d','einops'):
    spec = importlib.util.find_spec(m)
    print(m, '->', getattr(spec, 'origin', None))
print("Sanity check OK")
PY
