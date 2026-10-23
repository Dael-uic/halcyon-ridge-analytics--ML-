"""Generates the synthetic datasets used by the Halcyon Ridge site (all data is fictional)."""
import random, csv, json, math
random.seed(26)
ch = lambda o, w: random.choices(o, w)[0]
T = ["Urban","Mountain","Jungle","Desert","Plains"]
TP = {"Urban":1.5,"Mountain":1.2,"Jungle":1.0,"Desert":-.5,"Plains":-1}
IW = {"Urban":[3,4,1,2],"Mountain":[4,1,3,2],"Jungle":[5,2,1,2],"Desert":[1,4,3,2],"Plains":[1,1,3,5]}
INC = ["Ambush","IED","Indirect Fire","Direct Assault"]
ops = []
for i in range(1000):
    t = random.choice(T); w = ch(["Clear","Rain","Fog","Storm"],[5,2,2,1]); d = ch(["Day","Dusk","Night"],[5,2,3])
    dur = random.randint(1,30); force = random.randint(40,600); er = round(random.uniform(.3,3),2)
    air = int(random.random()<.5); intel = random.randint(1,10); armor = random.randint(0,60); med = random.randint(10,180)
    cr = 3+3.2*er-.5*intel-2.2*air+.045*med+.08*dur-.03*armor+TP[t]+{"Clear":0,"Rain":.6,"Fog":.8,"Storm":1.5}[w]+random.gauss(0,1.2)
    cr = round(max(.2,min(40,cr)),1)
    sc = cr+(5-intel)*.4+random.gauss(0,2)
    out = "Success" if sc<7 else "Partial" if sc<11 else "Failure"
    iw = IW[t][:]
    if d=="Night": iw[0]*=2
    if w in("Fog","Storm"): iw[2]*=.5
    if d=="Day" and w=="Clear": iw[3]*=1.5
    ops.append(dict(op_id=f"OP-{1001+i}",terrain=t,weather=w,time=d,duration=dur,force=force,enemy_ratio=er,air=air,
        intel=intel,armor=armor,medevac=med,casualty=cr,outcome=out,incident=ch(INC,iw),mce="Yes" if cr>=10 else "No"))
tri = []
for i in range(1200):
    r = dict(case_id=f"CS-{5001+i}",age=random.randint(18,50),hr=random.randint(55,160),sbp=random.randint(55,135),
        rr=random.randint(8,38),gcs=min(15,max(3,int(random.gauss(11,3.5)))),bleed=ch([0,1,2,3],[3,3,2,1]),evac=random.randint(10,180))
    r["s"]=(15-r["gcs"])*.6+r["bleed"]*1.2+(100-r["sbp"])/15+(r["hr"]-90)/20+abs(r["rr"]-16)/6+random.gauss(0,.5)
    tri.append(r)
ss = sorted(r["s"] for r in tri); n=len(ss); m=sum(ss)/n; sd=(sum((x-m)**2 for x in ss)/n)**.5
c1,c2,c3 = ss[int(n*.25)],ss[int(n*.55)],ss[int(n*.85)]
for r in tri:
    s=r.pop("s"); r["triage"]="Minor" if s<c1 else "Delayed" if s<c2 else "Immediate" if s<c3 else "Expectant"
    r["survived"]=int(random.random()<1/(1+math.exp(-(1.6-1.4*(s-m)/sd-.012*(r["evac"]-95)))))
for name,rows in (("operations",ops),("casualty_triage",tri)):
    with open(f"data/{name}.csv","w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
with open("data/dataset.js","w") as f:
    f.write("window.OPS="+json.dumps(ops)+";\nwindow.TRI="+json.dumps(tri)+";\n")
print(len(ops),len(tri))
