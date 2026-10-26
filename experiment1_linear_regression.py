from common_utils import *
FILE="experiment1_boston_synthetic_500.csv"
rows=load_csv(FILE)
features=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
train,test=train_test_split(rows)
means={c:mean([safe_float(r[c]) for r in train]) for c in features}
scales={c:std([safe_float(r[c]) for r in train]) for c in features}
def vector(r): return [(safe_float(r[c])-means[c])/scales[c] for c in features]
X=[vector(r) for r in train]; y=[safe_float(r["MEDV"]) for r in train]
# Standardized features; batch gradient descent for linear regression
weights=[0.0]*len(features); bias=mean(y); lr=0.025
for epoch in range(1800):
    grad=[0.0]*len(weights); gb=0.0
    for xi,yi in zip(X,y):
        err=bias+sum(w*x for w,x in zip(weights,xi))-yi
        gb+=err
        for j,x in enumerate(xi): grad[j]+=err*x
    n=len(X); bias-=lr*gb/n
    for j in range(len(weights)): weights[j]-=lr*grad[j]/n
actual=[]; predicted=[]
for r in test:
    xi=vector(r); predicted.append(bias+sum(w*x for w,x in zip(weights,xi))); actual.append(safe_float(r["MEDV"]))
mse=mean([(a-p)**2 for a,p in zip(actual,predicted)])
mae=mean([abs(a-p) for a,p in zip(actual,predicted)])
print("EXPERIMENT 1: Linear Regression - Boston Housing-like data")
print("Rows:",len(rows)," Train:",len(train)," Test:",len(test))
print("Features:",", ".join(features))
print("Mean Squared Error (MSE): {:.4f}".format(mse))
print("Root Mean Squared Error (RMSE): {:.4f}".format(math.sqrt(mse)))
print("Mean Absolute Error (MAE): {:.4f}".format(mae))
print("First 10 predictions (actual -> predicted):")
for a,p in list(zip(actual,predicted))[:10]: print("{:.2f} -> {:.2f}".format(a,p))
