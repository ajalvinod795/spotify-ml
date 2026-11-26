const $=id=>document.getElementById(id);
const palette=['#1ed760','#ff9f43','#61b8ff','#b69cff','#f16f89'];
async function api(path,opts){const r=await fetch(path,opts);if(!r.ok)throw new Error(await r.text());return r.json()}
function num(n){return Number(n).toLocaleString('en-US')}
async function init(){
 try{
 const [o,g]=await Promise.all([api('/api/overview'),api('/api/genres')]);
 $('stats').innerHTML=[
  [num(o.tracks),'Tracks in dataset'],[o.genres,'Distinct music genres'],[o.mean_popularity,'Mean popularity / 100'],[o.hit_rate+'%','Popularity ≥ 60']
 ].map((x,i)=>`<div class="stat"><small>${['TRACKS ANALYZED','UNIQUE GENRES','AVERAGE POPULARITY','HIT RATE'][i]}</small><strong>${x[0]}</strong><span>${x[1]}</span></div>`).join('');
 new Chart($('popChart'),{type:'bar',data:{labels:o.popularity_bins.map(x=>x.range),datasets:[{data:o.popularity_bins.map(x=>x.tracks),backgroundColor:'#1ed760aa',borderRadius:5}]},options:chartOpts()});
 new Chart($('genreChart'),{type:'bar',data:{labels:o.genres_top.map(x=>x.genre),datasets:[{data:o.genres_top.map(x=>x.tracks),backgroundColor:'#61b8ffaa',borderRadius:4}]},options:{...chartOpts(),indexAxis:'y'}});
 g.genres.forEach(v=>{ $('genre').add(new Option(v,v));$('filterGenre').add(new Option(v,v));});
 await Promise.all([loadMoods(),loadTracks(),loadModels()]);
 }catch(e){console.error(e);$('stats').innerHTML='<div class="stat">API unavailable. Start the backend with the README instructions.</div>'}
}
function chartOpts(){return {responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{color:'#25302d'},ticks:{color:'#87948e',font:{size:10}}},y:{grid:{color:'#25302d'},ticks:{color:'#87948e',font:{size:10}}}}}}
async function loadMoods(){
 const d=await api('/api/moods'), names=Object.values(d.mood_names), groups=[...new Set(d.points.map(p=>p.mood))];
 const traces=groups.map(m=>{const a=d.points.filter(p=>p.mood===m);return {x:a.map(p=>p.x),y:a.map(p=>p.y),text:a.map(p=>`${p.track} — ${p.artist}<br>${p.genre} · popularity ${p.popularity}`),name:m,mode:'markers',type:'scattergl',marker:{size:5,opacity:.62,color:palette[groups.indexOf(m)]}}});
 Plotly.newPlot('moodPlot',traces,{paper_bgcolor:'#13191c',plot_bgcolor:'#13191c',font:{color:'#aab6b0',size:10},margin:{l:45,r:15,t:10,b:45},xaxis:{title:'Principal component 1',gridcolor:'#29332f',zerolinecolor:'#3b4942'},yaxis:{title:'Principal component 2',gridcolor:'#29332f',zerolinecolor:'#3b4942'},legend:{orientation:'h',y:-.2}},{responsive:true,displaylogo:false});
 $('moodInfo').textContent=`${d.points.length.toLocaleString()} sampled tracks · PCA explains ${((d.explained_variance[0]+d.explained_variance[1])*100).toFixed(1)}% of standardized audio-feature variance across the two axes.`;
 $('moodCards').innerHTML=groups.map((m,i)=>`<div class="mood-card"><b style="color:${palette[i]}">${m}</b><span>${d.points.filter(p=>p.mood===m).length.toLocaleString()} projected tracks</span></div>`).join('');
}
async function loadTracks(){
 const params=new URLSearchParams({q:$('search').value,genre:$('filterGenre').value,sort:$('sort').value,limit:50});
 const d=await api('/api/tracks?'+params);
 $('trackRows').innerHTML=d.tracks.map(t=>`<tr><td>${esc(t.track_name)}<span class="track-sub">${esc(t.album_name)}</span></td><td>${esc(t.artists)}</td><td>${esc(t.track_genre)}</td><td class="pop">${t.popularity}</td><td>${Number(t.energy).toFixed(2)}</td><td>${Number(t.danceability).toFixed(2)}</td></tr>`).join('')||'<tr><td colspan="6">No tracks found.</td></tr>';
}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function loadModels(){
 const d=await api('/api/models');
 $('modelCards').innerHTML=Object.entries(d.metrics).map(([name,m])=>`<article class="panel model-card"><h3>${name}</h3>${Object.entries(m).map(([k,v])=>`<div class="metric"><span>${k.replace('_',' ').toUpperCase()}</span><b>${(v*100).toFixed(1)}%</b></div>`).join('')}</article>`).join('');
 $('params').innerHTML='<b>Best tuned boosting parameters</b><br>'+Object.entries(d.best_boosting_params).map(([k,v])=>`${esc(k)}: <b>${esc(v)}</b>`).join(' &nbsp; · &nbsp; ');
}
async function runPrediction(){
 const fields=['danceability','energy','valence','acousticness','speechiness','instrumentalness','tempo','loudness'];
 const payload={track_genre:$('genre').value,key:5,mode:1,time_signature:4,duration_ms:210000};
 fields.forEach(k=>payload[k]=Number($(k).value));
 $('predictBtn').disabled=true;$('predictBtn').textContent='Analyzing audio DNA…';
 try{const d=await api('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
 const p=d.consensus_probability;
 $('prediction').innerHTML=`<div class="score">${(p*100).toFixed(1)}%</div><div class="score-label">ENSEMBLE HIT PROBABILITY</div><h3 style="color:${p>=.5?'#1ed760':'#ff9f43'};margin:10px">${p>=.5?'Promising hit potential':'An under-the-radar sound'}</h3>${Object.entries(d.models).map(([name,v])=>`<div class="model-result"><div><span>${name}</span><b>${(v.hit_probability*100).toFixed(1)}% · ${v.prediction}</b></div><div class="bar"><i style="width:${v.hit_probability*100}%"></i></div></div>`).join('')}<p class="muted footnote">Model estimates are learned from historical dataset patterns, not a guarantee of future popularity.</p>`;
 }catch(e){$('prediction').textContent='Prediction failed: '+e.message}finally{$('predictBtn').disabled=false;$('predictBtn').innerHTML='✦ &nbsp; Predict hit potential'}
}
['danceability','energy','valence','acousticness','speechiness','instrumentalness','tempo','loudness'].forEach(k=>$(k).addEventListener('input',()=>{$(k+'-v').textContent=Number($(k).value).toFixed(k==='tempo'?0:k==='loudness'?1:2)+(k==='tempo'?' BPM':k==='loudness'?' dB':'')}));
$('predictBtn').addEventListener('click',runPrediction);$('searchBtn').addEventListener('click',loadTracks);$('search').addEventListener('keydown',e=>{if(e.key==='Enter')loadTracks()});$('filterGenre').addEventListener('change',loadTracks);$('sort').addEventListener('change',loadTracks);
init();