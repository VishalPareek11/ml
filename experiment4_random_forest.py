from common_utils import *
FILE="experiment4_adult_synthetic_500.csv"
rows=load_csv(FILE)
target="income"
numeric=["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
categorical=["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
# Encode categories as deterministic integer IDs. Tree uses category equality splits for categorical fields.
for r in rows:
    for c in numeric: r[c]=safe_float(r[c])
train,test=train_test_split(rows)
def gini(labels):
    n=len(labels)
    if not n:return 0
    counts=Counter(labels)
    return 1-sum((v/n)**2 for v in counts.values())
def majority(rows): return Counter(r[target] for r in rows).most_common(1)[0][0]
def best_split(data):
    parent=gini([r[target] for r in data]); best=None; best_gain=0
    n=len(data)
    for c in numeric:
        vals=sorted(set(r[c] for r in data))
        if len(vals)<2: continue
        # cap candidate thresholds to keep this offline implementation fast
        if len(vals)>12:
            vals=[vals[int(i*(len(vals)-1)/11)] for i in range(12)]
        thresholds=[(vals[i]+vals[i+1])/2 for i in range(len(vals)-1)]
        for t in thresholds:
            left=[r for r in data if r[c]<=t]; right=[r for r in data if r[c]>t]
            if not left or not right: continue
            gain=parent-len(left)/n*gini([r[target] for r in left])-len(right)/n*gini([r[target] for r in right])
            if gain>best_gain: best_gain=gain; best=("num",c,t,left,right)
    for c in categorical:
        for v in set(r[c] for r in data):
            left=[r for r in data if r[c]==v]; right=[r for r in data if r[c]!=v]
            if not left or not right: continue
            gain=parent-len(left)/n*gini([r[target] for r in left])-len(right)/n*gini([r[target] for r in right])
            if gain>best_gain: best_gain=gain; best=("cat",c,v,left,right)
    return best
def build(data,depth=0,max_depth=4,min_leaf=8):
    labels=[r[target] for r in data]
    if len(set(labels))==1 or depth>=max_depth or len(data)<2*min_leaf:
        return ("leaf",majority(data))
    split=best_split(data)
    if not split: return ("leaf",majority(data))
    typ,c,val,left,right=split
    if len(left)<min_leaf or len(right)<min_leaf: return ("leaf",majority(data))
    return ("node",typ,c,val,build(left,depth+1,max_depth,min_leaf),build(right,depth+1,max_depth,min_leaf))
def predict(node,r):
    if node[0]=="leaf": return node[1]
    _,typ,c,val,left,right=node
    go_left=(r[c]<=val) if typ=="num" else (r[c]==val)
    return predict(left if go_left else right,r)
forest=[]
for seed in range(9):
    rr=random.Random(seed+100)
    boot=[train[rr.randrange(len(train))] for _ in range(len(train))]
    forest.append(build(boot,max_depth=4,min_leaf=8))
y=[r[target] for r in test]
pred=[]
for row in test:
    votes=Counter(predict(tree,row) for tree in forest)
    pred.append(votes.most_common(1)[0][0])
print("EXPERIMENT 3: Random Forest (9 trees) - Adult Census Income-like data")
print("Rows:",len(rows)," Train:",len(train)," Test:",len(test))
print("Hyperparameters: 9 trees, max_depth=4, min_samples_leaf=8; bootstrap sampling, majority vote")
print("Accuracy: {:.4f}".format(accuracy(y,pred)))
labels=sorted(set(y+pred)); print("Confusion matrix (actual rows, predicted columns; labels={}):".format(labels))
for a in labels: print(a, [sum(1 for yt,yp in zip(y,pred) if yt==a and yp==b) for b in labels])
positive=">50K"
tp=sum(a==positive and b==positive for a,b in zip(y,pred)); fp=sum(a!=positive and b==positive for a,b in zip(y,pred)); fn=sum(a==positive and b!=positive for a,b in zip(y,pred))
p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f=2*p*r/(p+r) if p+r else 0
print("Positive class (>50K) Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(p,r,f))
print("Synthetic dataset; metrics are for demonstration, not real Adult Census findings.")
