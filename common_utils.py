#!/usr/bin/env python3
"""
Offline ML Lab experiments.
Only Python standard library modules are used; no pip installation or internet required.
Run from this folder with: python <script_name.py>
"""
import csv, math, random, statistics, os
from collections import Counter, defaultdict

def load_csv(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def train_test_split(rows, test_ratio=0.2, seed=42):
    data = rows[:]
    random.Random(seed).shuffle(data)
    cut = int(len(data)*(1-test_ratio))
    return data[:cut], data[cut:]

def accuracy(y, pred):
    return sum(a==b for a,b in zip(y,pred))/len(y) if y else 0.0

def confusion_binary(y, pred, positive=1):
    tp=sum(a==positive and b==positive for a,b in zip(y,pred))
    tn=sum(a!=positive and b!=positive for a,b in zip(y,pred))
    fp=sum(a!=positive and b==positive for a,b in zip(y,pred))
    fn=sum(a==positive and b!=positive for a,b in zip(y,pred))
    return [[tn,fp],[fn,tp]]

def classification_report(y, pred, positive):
    cm=confusion_binary(y,pred,positive)
    tn,fp=cm[0]; fn,tp=cm[1]
    p=tp/(tp+fp) if tp+fp else 0
    r=tp/(tp+fn) if tp+fn else 0
    f=2*p*r/(p+r) if p+r else 0
    return cm, p, r, f

def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

def std(xs):
    if len(xs)<2: return 1.0
    m=mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/len(xs)) or 1.0

def safe_float(v, default=0.0):
    try: return float(v)
    except (ValueError, TypeError): return default

def print_classification(y, pred, positive):
    cm,p,r,f=classification_report(y,pred,positive)
    print("Accuracy : {:.4f}".format(accuracy(y,pred)))
    print("Confusion matrix [[TN, FP], [FN, TP]]:", cm)
    print("Precision: {:.4f}".format(p))
    print("Recall   : {:.4f}".format(r))
    print("F1 score : {:.4f}".format(f))
