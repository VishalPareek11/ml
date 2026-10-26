from common_utils import *
FILE="experiment5_adult_synthetic_500.csv"
rows=load_csv(FILE); target="income"
numeric=["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
categorical=["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
for r in rows:
    for c in numeric: r[c]=safe_float(r[c])
train,test=train_test_split(rows)
def stump_candidates(data):
    out=[]
    for c in numeric:
        vals=sorted(set(r[c] for r in data))
        if len(vals)>10: vals=[vals[int(i*(len(vals)-1)/9)] for i in range(10)]
        for a,b in zip(vals,vals[1:]): out.append(("num",c,(a+b)/2))
    for c in categorical:
        for v in set(r[c] for r in data): out.append(("cat",c,v))
    return out
def pred_stump(st,r):
    typ,c,v=st
    return int(r[c]<=v) if typ=="num" else int(r[c]==v)
ytrain=[int(r[target]==">50K") for r in train]
weights=[1/len(train)]*len(train); models=[]; candidates=stump_candidates(train)
for round_no in range(12):
    best=None; best_err=1e9; best_polarity=1
    for st in candidates:
        raw=[pred_stump(st,r) for r in train]
        for polarity in (1,-1):
            pred=[v if polarity==1 else 1-v for v in raw]
            err=sum(w for w,a,b in zip(weights,ytrain,pred) if a!=b)
            if err<best_err: best_err=err; best=st; best_polarity=polarity
    if best is None or best_err>=0.5-1e-9: break
    err=max(1e-9,min(1-1e-9,best_err)); alpha=0.5*math.log((1-err)/err)
    models.append((best,best_polarity,alpha))
    raw=[pred_stump(best,r) for r in train]
    for i,(a,v) in enumerate(zip(ytrain,raw)):
        pr=v if best_polarity==1 else 1-v
        weights[i]*=math.exp(-alpha*(1 if a==pr else -1))
    total=sum(weights); weights=[w/total for w in weights]
def predict(row):
    score=0
    for st,pol,alpha in models:
        v=pred_stump(st,row); v=v if pol==1 else 1-v
        score+=alpha*(1 if v==1 else -1)
    return ">50K" if score>=0 else "<=50K"
y=[r[target] for r in test]; pred=[predict(r) for r in test]
print("EXPERIMENT 5: AdaBoost-style boosting - Adult Census Income-like data")
print("Rows:",len(rows)," Train:",len(train)," Test:",len(test))
print("Boosting rounds trained:",len(models))
print("Accuracy: {:.4f}".format(accuracy(y,pred)))
all_labels=["<=50K",">50K"]
print("Confusion matrix (actual rows, predicted columns; labels={}):".format(all_labels))
for a in all_labels: print(a, [sum(1 for yt,yp in zip(y,pred) if yt==a and yp==b) for b in all_labels])
positive=">50K"
tp=sum(a==positive and b==positive for a,b in zip(y,pred)); fp=sum(a!=positive and b==positive for a,b in zip(y,pred)); fn=sum(a==positive and b!=positive for a,b in zip(y,pred))
p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f=2*p*r/(p+r) if p+r else 0
print("Positive class (>50K) Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(p,r,f))
print("Synthetic dataset; metrics are for demonstration, not real Adult Census findings.")
