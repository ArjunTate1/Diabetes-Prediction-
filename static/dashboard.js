let SD={};

const MODELS=[
  {key:'lr',  diagId:'diag-lr',  cardId:'pcard-lr',  barY:'blry',  barN:'blrn',  pY:'plry',  pN:'plrn'},
  {key:'svm', diagId:'diag-svm', cardId:'pcard-svm', barY:'bsvmy', barN:'bsvmn', pY:'psvmy', pN:'psvmn'},
  {key:'rf',  diagId:'diag-rf',  cardId:'pcard-rf',  barY:'brfy',  barN:'brfn',  pY:'prfy',  pN:'prfn'},
  {key:'gb',  diagId:'diag-gb',  cardId:'pcard-gb',  barY:'bgby',  barN:'bgbn',  pY:'pgby',  pN:'pgbn'},
];

const NAMES={lr:'Logistic Regression',svm:'SVM',rf:'Random Forest',gb:'Gradient Boosting'};

async function loadStats(){
  try{
    const r=await fetch('/stats');
    SD=await r.json();
    fillStats();
    fillCM();
    fillFI();
  }catch(e){console.warn('Stats load failed',e);}
}

function p2(v){return(v*100).toFixed(2)+'%';}
function f3(v){return parseFloat(v).toFixed(3);}
function set(id,val){const el=document.getElementById(id);if(el)el.textContent=val;}

function fillStats(){
  for(const k of ['lr','svm','rf','gb']){
    const s=SD[k];if(!s)continue;
    set(`${k}-acc`, p2(s.accuracy));
    set(`${k}-prec`,f3(s.precision));
    set(`${k}-rec`, f3(s.recall));
    set(`${k}-f1`,  f3(s.f1_score));
    set(`${k}-auc`, f3(s.roc_auc));
  }
}

function fillCM(){
  for(const k of ['lr','svm','rf','gb']){
    const s=SD[k];if(!s||!s.confusion_matrix)continue;
    const c=s.confusion_matrix;
    set(`${k}-tn`,c[0][0]);set(`${k}-fp`,c[0][1]);
    set(`${k}-fn`,c[1][0]);set(`${k}-tp`,c[1][1]);
  }
}

function fillFI(){
  const rf=SD.rf;if(!rf||!rf.feature_importance)return;
  const fi=Object.entries(rf.feature_importance).sort((a,b)=>b[1]-a[1]);
  const max=fi[0][1];
  document.getElementById('fi-bars').innerHTML=fi.map(([k,v])=>`
    <div class="fir">
      <div class="fin" title="${k}">${k}</div>
      <div class="fit"><div class="fif" style="width:${(v/max*100).toFixed(1)}%"></div></div>
      <div class="fip">${(v*100).toFixed(1)}%</div>
    </div>`).join('');
}

function bestOf(vals){
  // vals = array of {key, val}
  const max=Math.max(...vals.map(v=>v.val));
  const winners=vals.filter(v=>Math.abs(v.val-max)<0.0001);
  return winners.map(w=>NAMES[w.key]).join(' / ');
}

async function runPredict(){
  const btn=document.getElementById('runBtn');
  const fields=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'];
  const payload={};
  for(const f of fields){
    const v=parseFloat(document.getElementById(f).value);
    if(isNaN(v)){alert('Please enter valid value for '+f);return;}
    payload[f]=v;
  }
  btn.disabled=true;
  btn.innerHTML='<span class="sp"></span> Running all 4 models...';
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d=await r.json();
    if(!d.success)throw new Error(d.error);
    showResults(d.results);
  }catch(e){alert('Error: '+e.message);}
  finally{btn.disabled=false;btn.innerHTML='&#9889; Run Prediction on All 4 Models';}
}

function showResults(res){
  document.getElementById('res').style.display='block';
  document.getElementById('res').scrollIntoView({behavior:'smooth',block:'start'});

  // Render each prediction card
  for(const m of MODELS){
    const data=res[m.key];if(!data||data.error)continue;
    renderCard(m,data);
  }

  // Insight banner
  const preds=MODELS.map(m=>res[m.key]?.prediction);
  const allAgree=preds.every(p=>p===preds[0]);
  const diabeticCount=preds.filter(p=>p===1).length;
  const diag=preds[0]===1?'Diabetic':'Non-Diabetic';
  let insText='<strong>Clinical Interpretation:</strong> ';
  if(allAgree){
    insText+=`All 4 models unanimously predict <strong>${diag}</strong>. `;
  }else{
    insText+=`Models show mixed predictions — ${diabeticCount}/4 models predict <strong>Diabetic</strong>. `;
  }
  insText+=`<strong>Random Forest</strong> and <strong>Gradient Boosting</strong> (ensemble methods) generally outperform classical models on this dataset. `+
           `<strong>Gradient Boosting</strong> achieves the highest Recall, minimizing missed diabetic cases. `;
  document.getElementById('ins-banner').innerHTML=insText;

  // Fill comparison table
  const keys=['lr','svm','rf','gb'];
  const tidMap={lr:'lr',svm:'svm',rf:'rf',gb:'gb'};
  for(const k of keys){
    const d=res[k];if(!d||d.error)continue;
    set(`t${k.replace('svm','svm')}p`,d.diagnosis);
    set(`t${k}b`,d.probability.toFixed(2)+'%');
  }
  // Fix table id prefix
  set('tlrp',res.lr?.diagnosis||'--');
  set('tsvmp',res.svm?.diagnosis||'--');
  set('trfp',res.rf?.diagnosis||'--');
  set('tgbp',res.gb?.diagnosis||'--');
  set('tlrb',(res.lr?.probability||0).toFixed(2)+'%');
  set('tsvmb',(res.svm?.probability||0).toFixed(2)+'%');
  set('trfb',(res.rf?.probability||0).toFixed(2)+'%');
  set('tgbb',(res.gb?.probability||0).toFixed(2)+'%');

  // Agreement
  const agreeEl=document.getElementById('tagr');
  if(agreeEl)agreeEl.innerHTML=allAgree?`<span class="wn">All Agree</span>`:`${diabeticCount}/4 Diabetic`;

  // Metrics from stats
  const metrics=[
    {stat:'accuracy',  tids:['tlra','tsvma','trfa','tgba'],  wid:'twa'},
    {stat:'precision', tids:['tlrpc','tsvmpc','trfpc','tgbpc'],wid:'twp'},
    {stat:'recall',    tids:['tlrr','tsvmr','trfr','tgbr'],  wid:'twr'},
    {stat:'f1_score',  tids:['tlrf','tsvmf','trff','tgbf'],  wid:'twf'},
    {stat:'roc_auc',   tids:['tlrau','tsvmau','trfau','tgbau'],wid:'twau'},
  ];
  const mkeys=['lr','svm','rf','gb'];
  for(const m of metrics){
    const vals=mkeys.map((k,i)=>({key:k,val:SD[k]?SD[k][m.stat]:0}));
    for(let i=0;i<mkeys.length;i++){
      const s=SD[mkeys[i]];
      if(s) set(m.tids[i], m.stat==='accuracy'?p2(s[m.stat]):f3(s[m.stat]));
    }
    const winner=bestOf(vals);
    const el=document.getElementById(m.wid);
    if(el)el.innerHTML=`<span class="wn">${winner}</span>`;
  }
}

function renderCard(m,data){
  const isDi=data.prediction===1;
  const card=document.getElementById(m.cardId);
  card.className='pcard '+(isDi?'di':'he');
  const diag=document.getElementById(m.diagId);
  diag.className='pd '+(isDi?'di':'he');
  diag.textContent=data.diagnosis;
  const pY=parseFloat(data.probability.toFixed(2));
  const pN=parseFloat((100-pY).toFixed(2));
  set(m.pY,pY+'%');
  set(m.pN,pN+'%');
  setTimeout(()=>{
    const by=document.getElementById(m.barY);
    const bn=document.getElementById(m.barN);
    if(by)by.style.width=Math.min(pY,100)+'%';
    if(bn)bn.style.width=Math.min(pN,100)+'%';
  },60);
}

loadStats();
