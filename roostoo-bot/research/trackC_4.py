from trackC_lib import *
BTC=C["BTC"]
spread = (BTC.ewm(span=168,adjust=False).mean()/BTC.ewm(span=672,adjust=False).mean()-1).values
def schmitt(on_th, off_th):
    g = np.zeros(len(spread),bool); s=False
    for i,x in enumerate(spread):
        if not s and x>on_th: s=True
        elif s and x<off_th: s=False
        g[i]=s
    return g
def debounce_off(n):  # stays on until n consecutive off bars
    raw=GATE; g=np.zeros(len(raw),bool); s=False; c=0
    for i,x in enumerate(raw):
        if s:
            c = c+1 if not x else 0
            if c>=n: s=False; c=0
        else:
            if x: s=True; c=0
        g[i]=s
    return g
eq,_=T.run_v1(); add("v1",eq)
for n in (6,12,24,48,72):
    eq,_=T.run_v1(gate=debounce_off(n)); add(f"debounce-off {n}h",eq)
# daily-only gate check: gate sampled at 00:00 and held
gd = pd.Series(GATE,index=C.index).where(C.index.hour==0).ffill().fillna(False).values.astype(bool)
eq,_=T.run_v1(gate=gd); add("gate checked daily 00:00",eq)
for off in (-0.005,-0.01,-0.015,-0.02,-0.03,-0.05):
    eq,_=T.run_v1(gate=schmitt(0,off)); add(f"schmitt off<{off}",eq)
print(table()); pd.DataFrame(rows).to_csv("out_trackC_4.csv",index=False)
# toggle diagnostics
for nm,g in (("v1",GATE),("schmitt-1%",schmitt(0,-0.01))):
    s=pd.Series(g,index=C.index); ch=s.astype(int).diff().abs()
    print(nm,"toggles/yr",ch.groupby(C.index.year).sum().to_dict(),"on-frac by yr",s.groupby(C.index.year).mean().round(3).to_dict())
# walk-forward on schmitt choice: eval by year for off=-1%,-2% vs v1 using only windows within years
