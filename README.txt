OFFLINE MACHINE LEARNING LAB PACK
=================================
All six CSVs are SYNTHETIC 500-row teaching datasets. They imitate the attributes/themes in your experiment sheets, but are NOT the original datasets.
No pip install or internet is required. Scripts use only Python standard-library modules. Python itself must already be installed.

Run one at a time from this folder:
python experiment1_linear_regression.py
python experiment2_logistic_regression.py
python experiment3_decision_tree.py
python experiment4_random_forest.py
python experiment5_boosting.py
python experiment6_hierarchical_clustering.py

Experiment mapping:
1. Boston Housing-like -> Linear Regression + MSE/RMSE/MAE
2. Titanic-like -> Logistic Regression + accuracy/confusion matrix/precision/recall/F1
3. Adult Census-like -> Decision Tree + classification metrics
4. Adult Census-like -> Random Forest + classification metrics
5. Adult Census-like -> AdaBoost-style boosting + classification metrics
6. Wholesale Customers-like -> Single Linkage hierarchical clustering

The algorithms are simplified hand-coded educational implementations to avoid scikit-learn, pandas, NumPy, Matplotlib, and internet. They print metrics but do not create a correlation heat map/plots. Experiment 6 clusters the first 30 rows (the CSV contains 500) to keep exact single-linkage computation quick in pure Python. Do not report generated metrics as results on the original datasets.
