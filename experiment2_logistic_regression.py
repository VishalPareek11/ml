from common_utils import *
FILE="experiment2_titanic_synthetic_500.csv"
rows=load_csv(FILE)
numeric=["Pclass","Age","SibSp","Parch","Fare"]
categorical=["Sex","Embarked"]
train,test=train_test_split(rows)
# Impute missing numeric values with training means (works even if you add blanks)
num_means={c:mean([safe_float(r[c]) for r in train if r[c].strip()]) for c in numeric}
cats={c:sorted(set(r[c] for r in train)) for c in categorical}
def raw_vector(r):
    out=[]
    for c in numeric: out.append(safe_float(r[c],num_means[c]))
    for c in categorical:
        out.extend([1.0 if r[c]==v else 0.0 for v in cats[c][1:]]) # reference category dropped
    return out
Xraw=[raw_vector(r) for r in train]
mu=[mean([x[j] for x in Xraw]) for j in range(len(Xraw[0]))]
sd=[std([x[j] for x in Xraw]) for j in range(len(Xraw[0]))]
def vector(r): return [(v-m)/s for v,m,s in zip(raw_vector(r),mu,sd)]
X=[vector(r) for r in train]; y=[int(r["Survived"]) for r in train]
w=[0.0]*len(X[0]); b=0.0; lr=0.12
def sigmoid(z):
    z=max(-30,min(30,z)); return 1/(1+math.exp(-z))
for epoch in range(1800):
    gw=[0.0]*len(w); gb=0.0
    for xi,yi in zip(X,y):
        err=sigmoid(b+sum(a*x for a,x in zip(w,xi)))-yi
        gb+=err
        for j,x in enumerate(xi): gw[j]+=err*x
    n=len(X); b-=lr*gb/n
    for j in range(len(w)): w[j]-=lr*(gw[j]/n + 0.001*w[j])
pred=[]
for r in test: pred.append(int(sigmoid(b+sum(a*x for a,x in zip(w,vector(r))))>=0.5))
ytest=[int(r["Survived"]) for r in test]
print("EXPERIMENT 2: Logistic Regression - Titanic-like data")
print("Rows:",len(rows)," Train:",len(train)," Test:",len(test))
print("Features used:",", ".join(numeric+categorical))
print_classification(ytest,pred,1)
print("Note: synthetic teaching data, not the original Titanic passenger records.")
