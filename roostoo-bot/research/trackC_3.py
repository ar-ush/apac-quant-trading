from trackC_lib import *
BTC, ETH = C["BTC"], C["ETH"]
spread = (BTC.ewm(span=168,adjust=False).mean()/BTC.ewm(span=672,adjust=False).mean()-1).values
def schmitt(on_th, off_th):
    g = np.zeros(len(spread),bool); s=False
    for i,x in enumerate(spread):
        if not s and x>on_th: s=True
        elif s and x<off_th: s=False
        g[i]=s
    return g
ge = (ETH.ewm(span=168,adjust=False).mean()>ETH.ewm(span=672,adjust=False).mean()).values
ges = ETH.ewm(span=168,adjust=False).mean().values/ETH.ewm(span=672,adjust=False).mean().values-1
cands = {"v1":GATE, "schmitt off<-1%":schmitt(0,-0.01), "BTC OR ETH":GATE|ge, "(BTC schmitt-1%) OR ETH":schmitt(0,-0.01)|ge}
print({k:round(v.mean(),3) for k,v in cands.items()})
out=[]
for name,g in cands.items():
    for off in range(0,24,2):
        eq,_ = T.run_v1(gate=g, rebal=T.rebal_mask(off,24)); r=T.evaluate(eq,name); r["off"]=off; out.append(r)
df=pd.DataFrame(out)
agg=df.groupby("label")[["mean","P20","P30","comp","tot","mdd","p10","p90","mean2023","mean2024","mean2025","mean2026"]].agg(["mean","min"]).round(3)
pd.set_option("display.width",300); pd.set_option("display.max_columns",60)
print(agg[[("mean","mean"),("mean","min"),("P20","mean"),("comp","mean"),("tot","mean"),("tot","min"),("mdd","mean"),("p10","mean")]])
print(df.groupby("label")[["mean2023","mean2024","mean2025","mean2026"]].mean().round(3))
# paired: fraction of phases where cand mean > v1 mean same phase
v=df[df.label=="v1"].set_index("off")
for name in list(cands)[1:]:
    c=df[df.label==name].set_index("off")
    print(name,"wins mean:",(c["mean"]>v["mean"]).sum(),"/12  wins tot:",(c["tot"]>v["tot"]).sum(),"/12  wins comp:",(c["comp"]>v["comp"]).sum(),"/12")
df.to_csv("out_trackC_3.csv",index=False)
