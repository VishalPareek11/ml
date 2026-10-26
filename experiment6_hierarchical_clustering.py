from common_utils import *
FILE="experiment6_wholesale_synthetic_500.csv"
rows=load_csv(FILE)
features=["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]
sample=rows[:30]
cols={c:[safe_float(r[c]) for r in sample] for c in features}
mu={c:mean(cols[c]) for c in features}; sdv={c:std(cols[c]) for c in features}
X=[[(safe_float(r[c])-mu[c])/sdv[c] for c in features] for r in sample]
def dist(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
clusters=[{i} for i in range(len(X))]
def single_link(A,B): return min(dist(X[i],X[j]) for i in A for j in B)
print("EXPERIMENT 6: Hierarchical Clustering (Single Linkage) - Wholesale-like data")
print("Dataset rows:",len(rows),"; clustering sample rows:",len(sample))
print("Features:",", ".join(features))
step=0
while len(clusters)>3:
    best=None; best_d=float("inf")
    for i in range(len(clusters)):
        for j in range(i+1,len(clusters)):
            d=single_link(clusters[i],clusters[j])
            if d<best_d: best_d=d; best=(i,j)
    i,j=best; A=clusters[i]; B=clusters[j]; merged=A|B; step+=1
    if step<=12 or len(clusters)<=5:
        print("Merge {:02d}: {} + {} at distance {:.3f}".format(step, sorted(A), sorted(B), best_d))
    clusters=[c for k,c in enumerate(clusters) if k not in (i,j)]+[merged]
print("\\nFinal 3 clusters (row indices are one-based):")
for i,c in enumerate(sorted(clusters,key=lambda c:min(c))):
    print("Cluster",i+1,":",sorted(j+1 for j in c))
print("Note: the handout describes Single Linkage on Wholesale Customers; synthetic data is used here.")
