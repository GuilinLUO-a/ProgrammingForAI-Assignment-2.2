## Module Changes & New Components

All components must now support **streaming settings**, with support for incremental `.partial_fit()` and per-chunk updates.

### `io.py`

- `load_csv` load all data at once

- `load_stream_csv(chunk_size=1000)` load data partly as the chunk_size

### `stream.py`
- Implements `StreamTrainer`, managing pipeline + logging metrics
- Supports `.fit_chunk(X, y)` fit X_chunk and y_chunk
- Support `.score_chunk(X, y)` 
  - predict X_chunk 
  - log metrics(cumulative accuracy,precision,recall,f1,auc(optional)) 
  - log memory footprint(training time(second), memory(MB))

### `tree.py`
- `DecisionTreeClassifier`
- `.partial_fit(X_chunk, y_chunk)` for online growth
- `.predict(X)` for predicting data
- Support config: `max_depth`, `min_samples_split`, `max_features` , `n_class`, `criterion(Entropy(default),Gini(optional))`, `window_size(optional)`

### `ensemble.py`
- `EnsembleClassifier`:（base_tree, n_estimators）
  - manages N decision trees 
  - base_tree (decision tree)
- `.partial_fit()`
  - fit the data with bootstrap sampling(bagging) 
- `.predict()` 
  - predict data by N decision trees
  - Output on majority vote

### `preprocessing.py`
- `Imputer` updates missing-value estimates on the fly
  - `.partial_fit(X)` or `.fit()` : maintain running mean, variance, mode (Using Welford)
  - `.transform(X)` fill in the empty data
  - support configuratioin `strategy`:`mean` or `mode`
- `OneHotEncoder` transform categories into onehot format
  - `.partial_fit(X)` or `.fit(X)` get categories
  - `.transform(X)` transform X into onhot format
- `StandardScaler` scales data using mean and variance(std)
  - `.partial_fit(X)` or `.fit(X)` maintain running mean, variance, mode (Using Welford)
  - `.transform(X)` scale X

### `stats.py`
- All statistical functions redesigned for streaming
- Implement chunk-based versions of:
  - Mean, variance, quantiles
  - Histograms 
- `.update_stats(X)` 
  - update overall mean, variance
  - update the quantiles and histograms of latest chunk 
- `.get_meanVar()` get the mean and variance of overall numeric data(Welford)
- `.get_quantiles()` get the quantiles of latest chunk(or window_sized quantiles)
- `.get_histograms()` get the histograms of latest chunk(or window_sized histograms)
- Support configurations  `window_size`


### `metrics.py`
- `StreamMetrics`:
- `.update(self, y_true_chunk, y_pred_chunk, y_score_chunk=None)`:
  - Update overall confusion matrix
  - Update AUC(optional)
- `.reset()` reset all metrcis
- `.result()`: get all metrics:
  - accuracy, precision, recall,f1( ,auc)
  - (rolling_accuracy, rolling_precision, rolling_recall, rolling_f1) when window_size is set
- All classification metrics:
  - `.accuracy()` get cumulative accuracy
  - `.precision()` get overall precision
  - `.recall()` get overall recall
  - `.f1()` get overall f1
  - `.auc()` get overall auc
  - `.rolling_auc()` get overall rolling_auc(window_sized accuracy)
  
### `pipeline.py`
- `LabelEncoder`:
  - `.partial_fit(y)` or `.fit(y)` update class_to_idx and idx_to_class for later transform
  - `.transform(y)` transform y label into index
  - `.inverse_transform(y)` map index back to label

- `Preprocessor`:
  - `.partial_fit(X)` or `.fit()` update for later transforming(map category data into onehot and scaler numeric data)
  - `.transform(X)` fill the empty values with mean or mode, then map category data into onehot and scaler numeric data
  - `reset` reset all maintained values

-  `Pipeline`
   - `.partial_fit(X, y)` and `.fit(X, y)` support for models and transformers
   - `.predict(X)` predict data
   - `score(X_pred, y_true)` calculate the accuracy
   - Support incremental transformation + prediction in chained pipeline
  ```python
  pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', RandomForestClassifier())
  ])
  pipe.partial_fit(X_chunk, y_chunk)
  ```
### `visualise.py`
- Provide reusable plotting functions using `matplotlib`
- Be usable across scripts, demos, or pipeline logs
- Required plots:
  - `plot_metric_over_time(metric_values, title, ylabel, save_path=None)`: Plot a metric (e.g., accuracy) across chunks
  - `compare_models(metric1, metric2, labels, save_path=None)`: Compare two models on streaming metrics
  - `plot_predictions_vs_ground_truth(y_true, y_pred, save_path=None)`: Visualise predictions vs. actuals on latest chunk
- Support options for saving to file or inline display

### Assignment structure

Folder PATH listing
Volume serial number is A0B0-3089
D:.
│   .coverage
│   .gitignore
│   README.md
│   
├───.vscode
│       settings.json
│       
├───benchmark
│       benchmark.ipynb
│       
├───data
│       adult.csv
│       
├───demo
│       stream_demo.ipynb
│       
├───numcompute_stream
│   │   ensemble.py
│   │   io.py
│   │   metrics.py
│   │   pipeline.py
│   │   preprocessing.py
│   │   stats.py
│   │   stream.py
│   │   tree.py
│   │   visualise.py
│   │   __init__.py
│   │   
│   └───__pycache__
│           ensemble.cpython-313.pyc
│           io.cpython-313.pyc
│           metrics.cpython-313.pyc
│           pipeline.cpython-313.pyc
│           preprocessing.cpython-313.pyc
│           stats.cpython-313.pyc
│           stream.cpython-313.pyc
│           tree.cpython-313.pyc
│           visualise.cpython-313.pyc
│           __init__.cpython-313.pyc
│           
└───tests
    │   test_ensemble.py
    │   test_io.py
    │   test_metrics.py
    │   test_pipeline.py
    │   test_preprocessing.py
    │   test_stats.py
    │   test_tree.py
    │   __init__.py
    │   
    └───__pycache__
            test_ensemble.cpython-313.pyc
            test_io.cpython-313.pyc
            test_metrics.cpython-313.pyc
            test_pipeline.cpython-313.pyc
            test_preprocessing.cpython-313.pyc
            test_stats.cpython-313.pyc
            test_tree.cpython-313.pyc
            __init__.cpython-313.pyc
            
