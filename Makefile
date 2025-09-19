.PHONY: build-mamba-local
build-mamba-local:
	DOCKER_BUILDKIT=1 docker compose \
	  --progress=plain build \
	  --build-arg USE_MAMBA_PREBUILT=0 \
	  zonos-tts-api

.PHONY: build-mamba-prebuilt
build-mamba-prebuilt:
	# Provide a known-good sha256 when opting in.
	# Example SHA below is a placeholder; replace with the real one.
	DOCKER_BUILDKIT=1 docker compose \
	  --progress=plain build \
	  --build-arg USE_MAMBA_PREBUILT=1 \
	  --build-arg MAMBA_PREBUILT_SHA256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" \
	  zonos-tts-api
