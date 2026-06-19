PYTHON_INTERPRETER = python3

#################################################################################
# PROCESSED DATASETS                                                            #
#################################################################################
advanced-mri-lesions:
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset advanced-mri-lesions

breast-lesion-usg:
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset breast-lesions-usg

breast-micro-calc:
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset breast-micro-calc

cbis-ddsm:
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset cbis-ddsm

tompei-cmmd:
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset tompei-cmmd

oasbud:
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset oasbud

bus-bra:
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset bus-bra

la-breast:
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset la-breast

inbreast:
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset inbreast

mama-mia:
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset mama-mia

rsna-bcd:
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset rsna-bcd

embed:
	$(PYTHON_INTERPRETER) -m scripts.preprocess --dataset embed

processed-datasets: advanced-mri-lesions breast-lesion-usg breast-micro-calc cbis-ddsm tompei-cmmd oasbud bus-bra la-breast inbreast mama-mia rsna-bcd embed
	@echo "All datasets processed"

#################################################################################
# SPLITS                                                                        #
#################################################################################
report_generation_dataset:
	$(PYTHON_INTERPRETER) -m scripts.split --processed_data_path ../data/processed --save_path ../data/report_generation_split

create_vqa_datasets_robust:
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv ../data/report_generation_split/all-rg-train.csv --output_dir ../data/robust_vqa_split --split train --shuffle_context --shuffle_questions --add_negative_examples_prob 0.5 --negative_examples_k 5 --output_fn vqa_train.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv ../data/report_generation_split/all-rg-test.csv --output_dir ../data/robust_vqa_split --split test --shuffle_context --shuffle_questions --add_negative_examples_prob 0.5 --negative_examples_k 5 --output_fn vqa_test.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv ../data/report_generation_split/all-rg-val.csv --output_dir ../data/robust_vqa_split --split val --shuffle_context --shuffle_questions --add_negative_examples_prob 0.5 --negative_examples_k 5 --output_fn vqa_val.jsonl

create_vqa_datasets_robust_single_modalities:
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv ../data/report_generation_split/us-rg-train.csv --output_dir ../data/robust_vqa_split --split train --shuffle_context --shuffle_questions --add_negative_examples_prob 0.5 --negative_examples_k 5 --output_fn vqa_train_us.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv ../data/report_generation_split/us-rg-test.csv --output_dir ../data/robust_vqa_split --split test --shuffle_context --shuffle_questions --add_negative_examples_prob 0.5 --negative_examples_k 5 --output_fn vqa_test_us.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv ../data/report_generation_split/us-rg-val.csv --output_dir ../data/robust_vqa_split --split val --shuffle_context --shuffle_questions --add_negative_examples_prob 0.5 --negative_examples_k 5 --output_fn vqa_val_us.jsonl

	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv ../data/report_generation_split/mg-rg-train.csv --output_dir ../data/robust_vqa_split --split train --shuffle_context --shuffle_questions --add_negative_examples_prob 0.5 --negative_examples_k 5 --output_fn vqa_train_mg.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv ../data/report_generation_split/mg-rg-test.csv --output_dir ../data/robust_vqa_split --split test --shuffle_context --shuffle_questions --add_negative_examples_prob 0.5 --negative_examples_k 5 --output_fn vqa_test_mg.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv ../data/report_generation_split/mg-rg-val.csv --output_dir ../data/robust_vqa_split --split val --shuffle_context --shuffle_questions --add_negative_examples_prob 0.5 --negative_examples_k 5 --output_fn vqa_val_mg.jsonl

	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv ../data/report_generation_split/mr-rg-train.csv --output_dir ../data/robust_vqa_split --split train --shuffle_context --shuffle_questions --add_negative_examples_prob 0.5 --negative_examples_k 5 --output_fn vqa_train_mr.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv ../data/report_generation_split/mr-rg-test.csv --output_dir ../data/robust_vqa_split --split test --shuffle_context --shuffle_questions --add_negative_examples_prob 0.5 --negative_examples_k 5 --output_fn vqa_test_mr.jsonl
	$(PYTHON_INTERPRETER) -m scripts.create_vqa --input_csv ../data/report_generation_split/mr-rg-val.csv --output_dir ../data/robust_vqa_split --split val --shuffle_context --shuffle_questions --add_negative_examples_prob 0.5 --negative_examples_k 5 --output_fn vqa_val_mr.jsonl

#################################################################################
# CLEANING
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
