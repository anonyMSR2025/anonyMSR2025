This is the anonymous code release for our MSR 2025 submission: **A Study on Using Explainable Retrieval for Assisting Vulnerability Patch Tracing**

### 1. How to reproduce

#### 0. Installing conda environment

Change the env path in environment.yml, then run the following:

```bash
conda env create -f environment.yml
conda activate msr2025
pip install numpy==1.23
python -m spacy download en_core_web_sm
```

#### 1. Data processing

1.a Split the diff code into segments of 64 tokens, and tokenize using codebert tokenizer
```bash
python3 Preprocess/process_data.py --data_dir ./Data/ --model_name microsoft/codebert-base --step 3
```
After this step, the tokenized data is saved under `./Data/Trace_microsoft-codebert-base/output_Dataset_train_2.json`

1.b Use TF-IDF to sort the segments for each commit:
```bash
python3 Preprocess/process_data.py --data_dir ./Data/ --model_name microsoft/codebert-base --step 4
```
After this step, the data with sorted list is saved under `./Data/Trace_microsoft-codebert-base/output_Dataset_test_ranked_attfirst.json`

#### 2. Training a multi-modal model for commit message and diff code

```bash
python train.py --device cuda --path best_model_json/trace_params.json --att_lambda 0.0 --batch_size 4 --epochs 4 --max_length 256 --codetopk 1 --model_name microsoft/codebert-base --attfirst --pooling_method cnn --is_augment both
```
After this step, the model is saved under the directory `Trace_microsoft-codebert-base_lam=0_maxlen=256_k=1_epoch=4_pooling=cnn_attfirst=True_augment=both`

#### 3. Using TfIdf-Highlight to explain the trained model

```bash
python3 explain_with_tfidf.py --device cuda --path best_model_json/trace_params.json --epochs 4 --max_length 256 --codetopk 1 --pooling_method cnn --attfirst --is_augment both --model_name microsoft/codebert-base --option tfidf --fold valid --highlight_mode codeonly --topk 10
```

#### 4. Using LIME to explain the trained model

```bash
python3 explain_with_lime.py --device cuda --path best_model_json/trace_params.json --epochs 4 --max_length 256 --codetopk 1 --pooling_method cnn --attfirst --is_augment both --model_name microsoft/codebert-base --fold valid --highlight_mode codeonly --topk 10 --num_samples 10
```

#### 5. Evaluating the faithfulness scores of LIME and TfIdf-Highlight

```bash
python3 compute_metrics.py --fold valid --attfirst --highlight_mode codeonly --max_length 256 --is_augment both --pooling_method cnn
```

#### 6. Manual labeling results

The manual labeling results and code for aggregating the results are in ./manual_labeling_results/
