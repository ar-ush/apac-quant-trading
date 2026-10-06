from trackC_lib import *
import time
t0=time.time()
eq,_ = T.run_v1(); add("v1",eq)
r = lambda n: C/C.shift(n)-1
r168,r336,r672 = r(168),r(336),r(672)
POOLm = L.POOL
def run(label, score, k=3, hyst=2, thr=0.0, **kw):
    eq,_ = T.run_v1(k=k,hyst=hyst,score=score,thr=thr,**kw); add(label,eq)
# 1 ensemble rank
def pr(x): return x.where(POOLm).rank(axis=1,pct=True)
ens3 = (pr(r168)+pr(r336)+pr(r672))/3
run("ens-rank 168/336/672 (+336>0)", ens3.where(r336>0), thr=-1)
ens2 = (pr(r168)+pr(r336))/2
run("ens-rank 168/336 (+336>0)", ens2.where(r336>0), thr=-1)
ens2b = (pr(r336)+pr(r504:=r(504)))/2
run("ens-rank 336/504 (+336>0)", ens2b.where(r336>0), thr=-1)
ensr = (r168/168+r336/336+r672/672)/3
run("ens-avg-daily-ret 168/336/672 (+336>0)", ensr.where(r336>0), thr=-1)
# 3 skip recent
for s in (24,48):
    sc = C.shift(s)/C.shift(336+s)-1
    run(f"skip-last-{s}h 336 (+336>0)", sc.where(r336>0), thr=-1)
# 4 thr
for th in (0.05,0.10,0.20):
    run(f"abs-thr {th}", r336, thr=th)
# 5 up-day fraction
d = C.pct_change(24)  # daily changes sampled hourly, use daily closes grid via 24h returns each hour; fraction of the 14 non-overlapping? use rolling mean of (24h ret>0) over 336h
upf = (d>0).astype(float).where(d.notna()).rolling(336,min_periods=200).mean()
for q in (0.5,0.55):
    run(f"upfrac336>={q}", r336.where(upf>=q), thr=0.0)
# 6 own trend
for sp in (168,336):
    ema = C.ewm(span=sp,adjust=False).mean()
    run(f"coin>EMA{sp}", r336.where(C>ema), thr=0.0)
ema_f=C.ewm(span=168,adjust=False).mean(); ema_s=C.ewm(span=672,adjust=False).mean()
run("coin EMA168>EMA672", r336.where(ema_f>ema_s), thr=0.0)
# 7 breakout / nearness to 336h high
hi = C.rolling(336,min_periods=300).max()
near = C/hi
for q in (0.90,0.95):
    run(f"near336hi>={q}", r336.where(near>=q), thr=0.0)
run("score=near336hi (+336>0)", near.where(r336>0), thr=-1)
run("score=ret336*near", (r336*near).where(r336>0), thr=-1)
# 8 acceleration
run("req ret168>0", r336.where(r168>0), thr=0.0)
run("req ret72>0", r336.where(r(72)>0), thr=0.0)
run("req ret24>-3%", r336.where(r(24)>-0.03), thr=0.0)
run("req ret168>ret336/2", r336.where(r168>r336/2), thr=0.0)
# 15 hyst
for h in (0,1,3,4):
    run(f"hyst{h}", r336, hyst=h)
print(table()); print(time.time()-t0)
pd.DataFrame(rows).to_csv("out_trackC_1.csv",index=False)
