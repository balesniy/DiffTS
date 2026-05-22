.PHONY: build build-cuda128 shell shell-cuda128 test test-cuda128 build-legacy shell-legacy train

SHELL = /bin/sh

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

GPUS ?= 0
CONFIG ?= 'config_orchard.yaml'
CHECKPOINT ?= 'None'
WEIGHTS ?= 'None'
SERVICE ?= diffts
RUN_IN_CONTAINER = USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose run --rm -e PLS_CHECKPOINT=$(CHECKPOINT) -e PLS_CONFIG=$(CONFIG) -e CUDA_VISIBLE_DEVICES=$(GPUS) $(SERVICE)
FORMAT ?= 'lineset'
FILTERING ?= 'true'
PARAMS ?=

build: build-cuda128

build-cuda128:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose build diffts --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

shell-cuda128:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose run --rm -e CUDA_VISIBLE_DEVICES=$(GPUS) diffts bash

test-cuda128:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose run --rm -e CUDA_VISIBLE_DEVICES=$(GPUS) diffts python3 scripts/docker_cuda_self_check.py

build-legacy:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose --profile legacy build diffts-legacy --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

shell-legacy:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose --profile legacy run --rm -e CUDA_VISIBLE_DEVICES=$(GPUS) diffts-legacy bash

train:
	$(RUN_IN_CONTAINER) python3 train.py --config $(CONFIG) --checkpoint $(CHECKPOINT) $(PARAMS)

test:
	$(RUN_IN_CONTAINER) python3 train.py --test --weights ${WEIGHTS} --config $(CONFIG) $(PARAMS)

shell: shell-cuda128
