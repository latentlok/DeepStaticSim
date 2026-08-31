# DeepStaticSim — repo-root shortcuts. Everything runs inside surrogate/'s uv
# environment (`uv run --no-sync` so a running server never races a lockfile
# change). surrogate/ has its own Makefile for training-side chores.
.PHONY: help app compare predict train test deploy

DL_DATA ?= /home/shared/resources/datasets/JEBsim/processed
CKPT    ?=

help:  ## list targets
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

app:  ## STL-upload web app on :8090 (APP_ARGS= for extras, e.g. --port 9000)
	cd surrogate && uv run --no-sync python ../app/server.py $(if $(CKPT),--ckpt $(CKPT)) $(APP_ARGS)

compare:  ## truth-vs-prediction viewer on :8081 (needs the DeepJEB store)
	cd surrogate && DL_DATA=$(DL_DATA) uv run --no-sync python utils/compare_server.py $(if $(CKPT),--ckpt $(CKPT))

predict:  ## one-shot: make predict STL=/path/to/part.stl [OUT=dir] [CKPT=...]
	@test -n "$(STL)" || { echo "usage: make predict STL=/path/to/part.stl [OUT=dir]"; exit 2; }
	cd surrogate && uv run --no-sync python ../app/runner.py $(abspath $(STL)) \
		--out-dir $(or $(OUT),jobs/$(basename $(notdir $(STL)))) $(if $(CKPT),--ckpt $(CKPT))

train:  ## train the surrogate (EXP=jeb_surface by default)
	cd surrogate && DL_DATA=$(DL_DATA) uv run python train.py experiment=$(or $(EXP),jeb_surface)

test:  ## full test suite (surrogate + app tests, CPU)
	cd surrogate && CUDA_VISIBLE_DEVICES="" uv run --no-sync pytest tests/ ../app/tests/ -q

deploy:  ## build + run the full app container on :8090 (ships deploy/data as /data)
	cd deploy && docker compose up -d --build && docker compose ps
