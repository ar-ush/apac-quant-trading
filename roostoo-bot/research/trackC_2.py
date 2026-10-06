from trackC_lib import *
import numba, time
eq,_ = T.run_v1(); add("v1",eq)
r = lambda n: C/C.shift(n)-1
r336 = r(336)
BTC, ETH = C["BTC"], C["ETH"]
def run(label, **kw):
    eq,_ = T.run_v1(**kw); add(label,eq)
# buffered gate
spread = (BTC.ewm(span=168,adjust=False).mean()/BTC.ewm(span=672,adjust=False).mean()-1).values
def schmitt(on_th, off_th):
    g = np.zeros(len(spread),bool); s=False
    for i,x in enumerate(spread):
        if not s and x>on_th: s=True
        elif s and x<off_th: s=False
        g[i]=s
    return g
for on_th,off_th in [(0.01,0.0),(0.02,0.0),(0.0,-0.01),(0.0,-0.02),(0.01,-0.01)]:
    run(f"schmitt on>{on_th} off<{off_th}", gate=schmitt(on_th,off_th))
# ETH confirm
ge = (ETH.ewm(span=168,adjust=False).mean()>ETH.ewm(span=672,adjust=False).mean()).values
run("gate BTC AND ETH", gate=GATE & ge)
run("gate BTC OR ETH", gate=GATE | ge)
# equal-weight pool index gate
rr = C.pct_change().where(L.POOL).mean(axis=1).fillna(0)
idx = (1+rr).cumprod()
gi = (idx.ewm(span=168,adjust=False).mean()>idx.ewm(span=672,adjust=False).mean()).values
run("gate pool-index EMA168/672", gate=gi)
run("gate BTC AND pool-index", gate=GATE&gi)
# BTC realised-vol veto
rv = BTC.pct_change().rolling(720).std().values*np.sqrt(24*365)
for q in (0.8,1.0):
    run(f"gate AND BTC rv30d<{q}", gate=GATE&(rv<q))
# rebal freq
for freq in (6,12,48):
    for off in (0,):
        run(f"rebal every {freq}h", rebal=T.rebal_mask(off,freq))
for off in (3,6,9,12,18):
    run(f"rebal 24h phase {off}", rebal=T.rebal_mask(off,24))
# dynamic k
ML = {k: L.membership(r336,k,2,0.0,+1,pool=L.POOL,reset=~GATE,rebal=L.REBAL) for k in (2,3,5)}
top1 = r336.where(L.POOL).max(axis=1).values
posn = (r336.where(L.POOL)>0).sum(axis=1).values
disp = r336.where(L.POOL).std(axis=1).values
def dyn(label, kfun):
    M = np.zeros_like(ML[3]); W = np.zeros(M.shape)
    for t in np.where(L.REBAL)[0]:
        k = kfun(t); M[t]=ML[k][t]; W[t]=L.eqw(ML[k][t:t+1],k)[0]
    eq,_ = T.run_v1(ML=M,W=W,k=3); add(label,eq)
for th in (0.3,0.5):
    dyn(f"dyn k: top1>{th}->2 else 3", lambda t,th=th: 2 if top1[t]>th else 3)
    dyn(f"dyn k: top1>{th}->3 else 5", lambda t,th=th: 3 if top1[t]>th else 5)
dyn("dyn k: >=12 pos coins->3 else 2", lambda t: 3 if posn[t]>=12 else 2)
dyn("dyn k: >=12 pos coins->5 else 3", lambda t: 5 if posn[t]>=12 else 3)
md = np.nanmedian(disp[~np.isnan(disp)])
dyn("dyn k: disp>med->2 else 4", lambda t: 2 if disp[t]>md else 3)
# weighting: inverse vol
vol = C.pct_change().rolling(168).std()
ML3 = ML[3]
for pw,lab in ((1.0,"invvol"),(0.5,"sqrt-invvol")):
    W = np.zeros(ML3.shape)
    iv = (1/vol).values**pw
    for t in np.where(L.REBAL)[0]:
        m = ML3[t]
        if m.any():
            w = np.where(m, iv[t], 0.0); w = w/np.nansum(w)*min(0.98, m.sum()/3*0.98)
            W[t]=np.nan_to_num(w)
    run(f"weights {lab}", ML=ML3, W=W)
print(table()); pd.DataFrame(rows).to_csv("out_trackC_2.csv",index=False)
