.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := src tests


# dev dependencies
install:
	uv venv ghpo_env --python 3.11 && . ghpo_env/bin/activate && uv pip install --upgrade pip
	uv pip install vllm==0.8.4
	uv pip install setuptools
	uv pip install flash-attn==2.7.4.post1 --no-build-isolation
	GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"

style:
	ruff format --line-length 119 --target-version py310 $(check_dirs) setup.py
	isort $(check_dirs) setup.py

quality:
	ruff check --line-length 119 --target-version py310 $(check_dirs) setup.py
	isort --check-only $(check_dirs) setup.py
	flake8 --max-line-length 119 $(check_dirs) setup.py

test:
	pytest -sv --ignore=tests/slow/ tests/

slow_test:
	pytest -sv -vv tests/slow/

# Evaluation

evaluate:
	$(eval PARALLEL_ARGS := $(if $(PARALLEL),$(shell \
		if [ "$(PARALLEL)" = "data" ]; then \
			echo "data_parallel_size=$(NUM_GPUS)"; \
		elif [ "$(PARALLEL)" = "tensor" ]; then \
			echo "tensor_parallel_size=$(NUM_GPUS)"; \
		fi \
	),))
	$(if $(filter tensor,$(PARALLEL)),export VLLM_WORKER_MULTIPROC_METHOD=spawn &&,) \
	MODEL_ARGS="pretrained=$(MODEL),dtype=bfloat16,$(PARALLEL_ARGS),max_model_length=32768,max_num_batched_tokens=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" && \
	if [ "$(TASK)" = "lcb" ]; then \
		lighteval vllm $$MODEL_ARGS "extended|lcb:codegeneration|0|0" \
			--use-chat-template \
			--output-dir data/evals/$(MODEL); \
	else \
		lighteval vllm $$MODEL_ARGS "lighteval|$(TASK)|0|0" \
			--use-chat-template \
			--output-dir data/evals/$(MODEL); \
	fi
