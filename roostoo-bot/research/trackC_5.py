from trackC_lib import *
r336 = C/C.shift(336)-1
eq,_=T.run_v1(); add("v1 k3",eq)
for k in (1,2,4):
    eq,_=T.run_v1(k=k); add(f"v1 k{k}",eq)
ML3 = L.membership(r336,3,2,0.0,+1,pool=L.POOL,reset=~GATE,rebal=L.REBAL)
cols=list(C.columns)
for fill in ("BTC","ETH"):
    j=cols.index(fill); M=ML3.copy(); W=np.zeros(M.shape)
    for t in np.where(L.REBAL)[0]:
        if not GATE[t]: continue
        m=M[t].copy(); n=m.sum()
        w=np.where(m,1/3,0.0)
        if n<3 and not m[j]:
            m[j]=True; w[j]=(3-n)/3
        M[t]=m; W[t]=w*0.98
    eq,_=T.run_v1(ML=M,W=W); add(f"fill idle slots with {fill}",eq)
# fill idle only if k<=1
print(table()); pd.DataFrame(rows).to_csv("out_trackC_5.csv",index=False)
# fraction of gate-on rebalances with idle slots
n=ML3[L.REBAL & GATE].sum(axis=1); print("gate-on rebalances idle-slot dist:",pd.Series(n).value_counts(normalize=True).sort_index().round(3).to_dict())
