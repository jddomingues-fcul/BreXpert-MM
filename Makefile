DATASETS = advanced-mri-lesions breast-lesions-usg breast-micro-calc cbis-ddsm tompei-cmmd oasbud la-breast inbreast mama-mia embed bus-cot
PYTHON_INTERPRETER = python3

.PHONY: clean clean_logs smoke embed processed-datasets data2hf $(DATASETS)


#################################################################################
# SET HF CACHE																	#
#################################################################################
export HF_DATASETS_CACHE ?= /mnt/sam1/hf_datasets_cache
export HF_HUB_CACHE      ?= /mnt/sam1/hf_hub_cache


#################################################################################
# PROCESSED DATASETS                                                            #
#################################################################################
$(DATASETS):
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset $@

processed-datasets: $(DATASETS)
	@echo "All datasets processed"

#################################################################################
# SPLITS                                                                        #
#################################################################################
report_generation_dataset:
	$(PYTHON_INTERPRETER) -m scripts.split --processed_data_path ../data/processed --save_path ../data/report_generation_split --val_frac 0.05 --test_frac 0.05
	@echo "Completed"

# Train gets the augmentation (context withheld at random, shuffling within a
# stage, inapplicable-question negatives, some questions rendered as multiple
# choice). Val and test use --eval_preset: full context, fixed order, no
# negatives, so the evaluation sets are deterministic and every question is
# answerable from what the model was given.
#
# --max_turns 16 keeps the text budget per sample bounded, which matters when
# LoRA-tuning on a single 24 GB card: ~280 image tokens for a mammogram plus
# ~16 turns of dialogue stays near 1.2k tokens per sample.
VQA_TRAIN_FLAGS ?= --shuffle_context --shuffle_questions --context_dropout 0.2 --max_turns 16
VQA_EVAL_FLAGS ?= --eval_preset --max_turns 16

SPLIT_DIR ?= ../data/report_generation_split
VQA_DIR   ?= ../data/robust_vqa_split

create_vqa_datasets_robust:
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv $(SPLIT_DIR)/all-rg-train.csv --output_dir $(VQA_DIR) --split train $(VQA_TRAIN_FLAGS) --output_fn vqa_train.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv $(SPLIT_DIR)/all-rg-test.csv --output_dir $(VQA_DIR) --split test $(VQA_EVAL_FLAGS) --output_fn vqa_test.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv $(SPLIT_DIR)/all-rg-val.csv --output_dir $(VQA_DIR) --split val $(VQA_EVAL_FLAGS) --output_fn vqa_val.jsonl

# Split filenames use the short modality codes written by scripts/split.py
# (mg / us / mr), which is what these targets read.
create_vqa_datasets_robust_single_modalities:
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv $(SPLIT_DIR)/us-rg-train.csv --output_dir $(VQA_DIR) --split train $(VQA_TRAIN_FLAGS) --output_fn vqa_train_us.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv $(SPLIT_DIR)/us-rg-test.csv --output_dir $(VQA_DIR) --split test $(VQA_EVAL_FLAGS) --output_fn vqa_test_us.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv $(SPLIT_DIR)/us-rg-val.csv --output_dir $(VQA_DIR) --split val $(VQA_EVAL_FLAGS) --output_fn vqa_val_us.jsonl

	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv $(SPLIT_DIR)/mg-rg-train.csv --output_dir $(VQA_DIR) --split train $(VQA_TRAIN_FLAGS) --output_fn vqa_train_mg.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv $(SPLIT_DIR)/mg-rg-test.csv --output_dir $(VQA_DIR) --split test $(VQA_EVAL_FLAGS) --output_fn vqa_test_mg.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv $(SPLIT_DIR)/mg-rg-val.csv --output_dir $(VQA_DIR) --split val $(VQA_EVAL_FLAGS) --output_fn vqa_val_mg.jsonl

	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv $(SPLIT_DIR)/mr-rg-train.csv --output_dir $(VQA_DIR) --split train $(VQA_TRAIN_FLAGS) --output_fn vqa_train_mr.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv $(SPLIT_DIR)/mr-rg-test.csv --output_dir $(VQA_DIR) --split test $(VQA_EVAL_FLAGS) --output_fn vqa_test_mr.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv $(SPLIT_DIR)/mr-rg-val.csv --output_dir $(VQA_DIR) --split val $(VQA_EVAL_FLAGS) --output_fn vqa_val_mr.jsonl


#################################################################################
# DATA 2 HF HUB																	#
#################################################################################
data2hf:
	$(PYTHON_INTERPRETER) -m scripts.data2hf --split val
	$(PYTHON_INTERPRETER) -m scripts.data2hf --split test
	$(PYTHON_INTERPRETER) -m scripts.data2hf --split train

#################################################################################
# CLEANING																		#
#################################################################################
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "wandb" -exec rm -rf {} +
	find . -type d -name "artifacts" -exec rm -rf {} +
	find . -type d -name "lightning_logs" -exec rm -rf {} +
	find . -type f -name "*.index" -delete
	find . -type f -name "*.index.classes" -delete
	find . -type f -name "*.DS_Store" -delete
	find . -type d -name "extracted_features" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type d -name ".model_artefacts" -exec rm -rf {} +
	rm -rf plots/*
	rm nohup.out

clean_logs:
	rm -rf logs/*
