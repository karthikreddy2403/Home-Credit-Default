# Home Credit Default Risk

An interactive LightGBM project built from Home Credit’s seven-table repayment benchmark. Select a fictional repayment history, enter loan amounts, and change or clear an external credit score to see the model’s positive-class score.

## Try the demo

Extract the ZIP and open **docs/index.html** in a browser. The fitted model, interface, and font are included. The page works offline and all input values stay on your device.

## Put it on GitHub

Create a repository and upload the extracted contents. Place **README.md**, **train.py**, and the **docs** folder at the top level of the repository. For a public interactive link, open Settings → Pages, choose **Deploy from a branch**, select **main** and **/docs**, and save. GitHub displays the published URL when the deployment completes. [GitHub Pages instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)


## Results for this implementation

| Measure | Value |
| --- | ---: |
| Labeled applications | 307,511 |
| Application and history features | 288 |
| Training applications | 246,008 |
| Held-out applications | 61,503 |
| Positive-label rate in the test set | 8.07% |
| Held-out ROC AUC | 0.7783 |

This repository contains a newly trained implementation. Karthik’s original portfolio project reported 0.73 AUC. The metrics above apply to the model distributed here and its own split.

## How the pipeline works

1. Use one row per current application. Join bureau monthly statuses to their loans by `SK_ID_BUREAU`, then summarize each source by `SK_ID_CURR` before joining.
2. Aggregate prior decisions, installment timing and payment ratios, POS loan delinquency, card utilization, and bureau history. Compute recent-window means and four ratios from application amounts. Applicant IDs, extra calendar dates in the mirror, and the target are excluded from the predictors.
3. Hold out a stratified 20% of applicants with seed 42. On a stratified subset of 60,000 training records, compare 15 versus 31 leaves with three-fold cross-validation. Fit the selected 15-leaf, 130-tree model on all training records. Use the test split once for the published ROC AUC.
4. Export the tree thresholds, missing-value directions, and leaf scores. The browser sums leaf values, applies the logistic function, and shows a benchmark score. This educational demo has no lending decision rule.

The interactive history options are group summaries from the training split. “Short” means up to four recorded installments, including applicants without installment rows. The two longer histories require more than four installments and late-payment shares of at most 10% or at least 35%, respectively. A feature stays missing when fewer than half of the group has an observed value. The score is the model’s response to a fictional combination, and its calibration has not been established.

## Reproduce training

The demo is ready to run as distributed. To refit it, use Python 3.12 and install the dependencies in a virtual environment.

```sh
python -m pip install -r requirements.txt
python download_data.py data
python train.py --data data --features-cache .cache/features.parquet
```

The downloader fetches approximately 1.1 GiB of pinned Parquet shards and verifies each SHA-256 checksum. As an alternative, put Kaggle’s original seven CSV files in `data/` and skip the download command. Use a new cache path when changing the input data. Training can require several gigabytes of RAM. The ZIP contains source and fitted model parameters. Raw applicant records are downloaded separately.

## Files

- **features.py** builds the applicant-level feature matrix.
- **train.py** fits LightGBM and writes **metrics.json**, **docs/model-data.js**, and the Python reference scores in **tests/**.
- **docs/predict.js** implements inference. **docs/main.js** connects it to the form.
- **tests/verify.cjs** compares JavaScript predictions with Python, including missing values. Run `node tests/verify.cjs` with Node.js.

[Home Credit competition and data dictionary](https://www.kaggle.com/c/home-credit-default-risk/data?select=application_train.csv) · [Pinned Parquet source](https://huggingface.co/datasets/Lixiaoxiofficial/home-credit-default-risk/tree/b79b5e7c5bdd2f7ff4a7d53ef4b44bcfdc88823e)
