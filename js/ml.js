/* Halcyon Ridge ML toolkit: six algorithms written from scratch (no libraries). */
const ML={};
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const sd=a=>{const m=mean(a);return Math.sqrt(mean(a.map(v=>(v-m)**2)))||1};
const dot=(w,x)=>x.reduce((s,v,j)=>s+v*w[j],0);
ML.split=(rows,f=.8)=>{const n=Math.floor(rows.length*f);return[rows.slice(0,n),rows.slice(n)]};
ML.scaler=(rows,cols)=>{const m=cols.map(c=>mean(rows.map(r=>r[c]))),s=cols.map(c=>sd(rows.map(r=>r[c])));return{m,s,tf:r=>cols.map((c,i)=>(r[c]-m[i])/s[i])}};

// 1. Linear regression (gradient descent on standardized features)
ML.linreg=(tr,te,cols,y)=>{const sc=ML.scaler(tr,cols),X=tr.map(sc.tf),Y=tr.map(r=>r[y]);let w=cols.map(()=>0),b=mean(Y);
 for(let e=0;e<1500;e++){const gw=w.map(()=>0);let gb=0;X.forEach((x,i)=>{const er=b+dot(w,x)-Y[i];gb+=er;x.forEach((v,j)=>gw[j]+=er*v)});w=w.map((v,j)=>v-.05*gw[j]/X.length);b-=.05*gb/X.length}
 const pred=r=>b+dot(w,sc.tf(r)),P=te.map(pred),T=te.map(r=>r[y]),mt=mean(T);
 return{pred,w,b,sc,test:{P,Y:T,rmse:Math.sqrt(mean(P.map((p,i)=>(p-T[i])**2))),r2:1-P.reduce((s,p,i)=>s+(p-T[i])**2,0)/T.reduce((s,v)=>s+(v-mt)**2,0)}}};

// 2. Naive Bayes (categorical, Laplace smoothing)
ML.nb=(tr,te,cols,y)=>{const cl=[...new Set(tr.map(r=>r[y]))],pri={},lik={};
 cl.forEach(c=>{const s=tr.filter(r=>r[y]==c);pri[c]=s.length/tr.length;lik[c]=cols.map(f=>{const vs=[...new Set(tr.map(r=>r[f]))],o={};vs.forEach(v=>o[v]=(s.filter(r=>r[f]==v).length+1)/(s.length+vs.length));return o})});
 const prob=r=>{const l=cl.map(c=>Math.log(pri[c])+cols.reduce((s,f,i)=>s+Math.log(lik[c][i][r[f]]||1e-6),0)),m=Math.max(...l),e=l.map(v=>Math.exp(v-m)),t=e.reduce((a,b)=>a+b);return cl.map((c,i)=>({c,p:e[i]/t})).sort((a,b)=>b.p-a.p)};
 return{cl,prob,acc:mean(te.map(r=>+(prob(r)[0].c==r[y]))),top2:mean(te.map(r=>+prob(r).slice(0,2).some(x=>x.c==r[y])))}};

// 3. K-nearest neighbours (Euclidean distance on standardized features)
ML.knn=(tr,te,cols,y)=>{const sc=ML.scaler(tr,cols),T=tr.map(r=>({r,x:sc.tf(r)}));
 const vote=(r,k)=>{const x=sc.tf(r),n=T.map(t=>({r:t.r,d:Math.hypot(...x.map((v,i)=>v-t.x[i]))})).sort((a,b)=>a.d-b.d).slice(0,k),c={};n.forEach(t=>c[t.r[y]]=(c[t.r[y]]||0)+1);return{n,c,p:Object.entries(c).sort((a,b)=>b[1]-a[1])[0][0]}};
 return{vote,acc:k=>mean(te.map(r=>+(vote(r,k).p==r[y])))}};

// 4. Support vector machine (linear, Pegasos sub-gradient training)
ML.svm=(tr,te,cols,y,pos)=>{const sc=ML.scaler(tr,cols),X=tr.map(sc.tf),Y=tr.map(r=>r[y]==pos?1:-1),lam=.01;let w=cols.map(()=>0),b=0,t=0;
 for(let e=0;e<60;e++)for(let i=0;i<X.length;i++){t++;const eta=1/(lam*(t+50)),m=Y[i]*(b+dot(w,X[i]));w=w.map(v=>v*(1-eta*lam));if(m<1){w=w.map((v,j)=>v+eta*Y[i]*X[i][j]);b+=eta*Y[i]*.1}}
 const f=r=>b+dot(w,sc.tf(r)),tp=te.filter(r=>r[y]==pos&&f(r)>0).length,fn=te.filter(r=>r[y]==pos&&f(r)<=0).length,fp=te.filter(r=>r[y]!=pos&&f(r)>0).length;
 return{f,w,acc:mean(te.map(r=>+((f(r)>0)==(r[y]==pos)))),recall:tp/(tp+fn||1),precision:tp/(tp+fp||1)}};

// 5. Decision tree (CART, Gini impurity)
ML.tree=(rows,cols,y,depth)=>{const cnt=s=>{const c={};s.forEach(r=>c[r[y]]=(c[r[y]]||0)+1);return c};
 const gini=s=>1-Object.values(cnt(s)).reduce((a,v)=>a+(v/s.length)**2,0),maj=s=>Object.entries(cnt(s)).sort((a,b)=>b[1]-a[1])[0][0];
 const build=(s,d)=>{const n={n:s.length,cls:maj(s),g:gini(s)};if(!d||n.g<.02||s.length<12)return n;let best=null;
  cols.forEach(f=>{const v=[...new Set(s.map(r=>r[f]))].sort((a,b)=>a-b);for(let i=0;i<v.length-1;i++){const t=(v[i]+v[i+1])/2,L=s.filter(r=>r[f]<=t),R=s.filter(r=>r[f]>t);if(L.length<6||R.length<6)continue;const g=(L.length*gini(L)+R.length*gini(R))/s.length;if(!best||g<best.g)best={g,f,t,L,R}}});
  if(!best||best.g>=n.g)return n;n.f=best.f;n.t=best.t;n.l=build(best.L,d-1);n.r=build(best.R,d-1);return n};
 return build(rows,depth)};
ML.treePredict=(n,r)=>n.f?ML.treePredict(r[n.f]<=n.t?n.l:n.r,r):n.cls;

// 6. Artificial neural network (1 hidden tanh layer, sigmoid output, SGD)
ML.ann=(tr,te,cols,y,H=8,epochs=150)=>{const sc=ML.scaler(tr,cols),X=tr.map(sc.tf),Y=tr.map(r=>+r[y]);let s=7;const rnd=()=>{s=(s*16807)%2147483647;return s/2147483647-.5};
 const W1=Array.from({length:H},()=>cols.map(rnd)),B1=Array(H).fill(0),W2=Array.from({length:H},rnd);let B2=0;const sig=z=>1/(1+Math.exp(-z));
 const fwd=x=>{const h=W1.map((w,j)=>Math.tanh(B1[j]+dot(w,x)));return{h,o:sig(B2+dot(W2,h))}},loss=[],lr=.05;
 for(let e=0;e<epochs;e++){let L=0;X.forEach((x,i)=>{const{h,o}=fwd(x),d=o-Y[i];L-=Y[i]*Math.log(o+1e-9)+(1-Y[i])*Math.log(1-o+1e-9);
  h.forEach((hv,j)=>{const dh=d*W2[j]*(1-hv*hv);W2[j]-=lr*d*hv;B1[j]-=lr*dh;W1[j].forEach((_,k)=>W1[j][k]-=lr*dh*x[k])});B2-=lr*d});loss.push(L/X.length)}
 const pred=r=>fwd(sc.tf(r)).o;return{pred,loss,H,acc:mean(te.map(r=>+((pred(r)>.5)==!!r[y])))}};
