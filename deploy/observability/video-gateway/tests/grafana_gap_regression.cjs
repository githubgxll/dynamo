// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Runs saved Grafana 12.2.1 transformations and its actual uPlot line builder.
const fs = require('node:fs');
const vm = require('node:vm');
const path = process.env.GRAFANA_BUNDLE_DIR;
if (!path) throw Error('set GRAFANA_BUNDLE_DIR to saved Grafana 12.2.1 assets');
const ctx = { self: {}, console, setTimeout, clearTimeout, URL, URLSearchParams, TextEncoder, TextDecoder, performance, navigator: {userAgent:'node'}, location:{href:'http://localhost/'} };
ctx.window=ctx;ctx.globalThis=ctx;ctx.self=ctx;
vm.createContext(ctx);
for (const name of fs.readdirSync(path)) {
  if (!name.endsWith('.js.下载') || name.startsWith('runtime.')) continue;
  vm.runInContext(fs.readFileSync(path+'/'+name,'utf8'), ctx, {filename:name});
}
const chunks = Object.entries(ctx).filter(([key,v])=>key.startsWith('webpackChunk') && Array.isArray(v));
const modules={};for (const [_,items] of chunks) for (const chunk of items) Object.assign(modules,chunk[1]);
const cache={};
function req(id){ if(cache[id])return cache[id].exports;if(!modules[id])throw Error('missing module '+id); const mod={exports:{}};cache[id]=mod;modules[id](mod,mod.exports,req);return mod.exports; }
req.d=(obj,defs)=>{for(const key in defs)if(!Object.hasOwn(obj,key))Object.defineProperty(obj,key,{enumerable:true,get:defs[key]});};
req.r=obj=>{Object.defineProperty(obj,Symbol.toStringTag,{value:'Module'});Object.defineProperty(obj,'__esModule',{value:true});};
req.n=mod=>{const getter=mod&&mod.__esModule?()=>mod.default:()=>mod;req.d(getter,{a:getter});return getter;};
req.o=(obj,k)=>Object.hasOwn(obj,k);req.g=ctx;req.hmd=mod=>mod;req.nmd=mod=>mod;req.p='';
const assert = require('node:assert/strict');
const transformers=req(52645).c;
const of=req(62467).of;
function transform(config,frames){
  const transformer=Object.values(transformers).find(t=>t.id===config.id);
  assert.ok(transformer,`unknown transformer ${config.id}`);
  let output;
  of(frames).pipe(transformer.operator(structuredClone(config.options),{})).subscribe({
    next:value=>output=value,error:error=>{throw error;},
  });
  return output;
}

// uPlot can load without a browser. Record its canvas paths, using its unchanged
// bundled line/gap algorithm; no DOM, remote requests or synthetic gap logic.
const savedWindow=ctx.window;
ctx.window=undefined;
ctx.Path2D=class {
  constructor(other){this.operations=other?.operations?.slice()||[];}
  moveTo(...args){this.operations.push(['moveTo',...args]);}
  lineTo(...args){this.operations.push(['lineTo',...args]);}
  rect(...args){this.operations.push(['rect',...args]);}
  closePath(){this.operations.push(['closePath']);}
};
const uplot=req(36613).A;
ctx.window=savedWindow;
function linePaths(times,values,width=1000){
  const u={
    mode:1,
    series:[{scale:'x'},{scale:'y',pxRound:v=>v,spanGaps:false,
      gaps:(u,s,a,b,gaps)=>gaps,fill:null}],
    scales:{x:{key:'x',ori:0,dir:1,min:times[0],max:times.at(-1)},
      y:{key:'y',ori:1,dir:1,min:0,max:120}},
    bbox:{left:0,top:0,width,height:200},_data:[times,values],bands:[],
    valToPosH:(v,s,d,o)=>o+(v-s.min)/(s.max-s.min)*d,
    valToPosV:(v,s,d,o)=>o+d-(v-s.min)/(s.max-s.min)*d,
    posToVal:value=>times[0]+value/width*(times.at(-1)-times[0]),
  };
  return uplot.paths.linear()(u,1,0,times.length-1);
}
function frame(name,refId,times,values,includeName=true){
  return {...(includeName?{name:'raw '+name}:{}),length:times.length,refId,fields:[
    {name:'Time',type:'time',values:times,config:{}},
    {name:'Value',type:'number',values,labels:{pool:'fl2va'},
      config:{displayNameFromDS:name}},
  ]};
}

// Counterexample: the old test's `v == null` accepted undefined. The actual
// renderer connects all 38 undefined cells with no clipping at all.
const shortTimes=Array.from({length:42},(_,i)=>i*15000);
const sparse=frame('fl2va / mean','A',[0,15000,600000,615000],[108,108,9.9,9.9]);
const axis=frame('__sample_timeline__','Z',shortTimes,shortTimes.map(()=>0));
const old=transform({id:'joinByField',options:{byField:'Time',mode:'outer'}},[sparse,axis]);
const oldValues=old[0].fields[1].values;
assert.equal(Array.from(oldValues).filter(v=>v===undefined).length,38);
assert.equal(Array.from(oldValues).filter(v=>v===null).length,0);
assert.equal(linePaths(shortTimes,oldValues).gaps.length,0);
assert.equal(linePaths(shortTimes,oldValues).clip,null);

function* panels(items){for(const p of items){yield p;yield* panels(p.panels||[]);}}
const testedPanels=[];
let cases=0;
for(const filename of ['video-gateway.json','video-worker-omni.json']){
  const dashboard=JSON.parse(fs.readFileSync(require('node:path').join(__dirname,'../grafana',filename),'utf8'));
  const snapshot=dashboard.templating.list.find(v=>v.name==='snapshot_interval');
  assert.ok(snapshot?.auto && snapshot.hide===2 && snapshot.skipUrlSync);
  for(const [hours,expected] of [[1,'15s'],[6,'15s'],[720,'30s']]){
    const range={from:new Date(0),to:new Date(hours*3600000)};
    assert.equal(req(17548).calculateInterval(range,snapshot.auto_count,snapshot.auto_min).interval,expected);
  }
  for(const panel of panels(dashboard.panels)){
    const axisTarget=panel.targets?.find(t=>t.refId==='Z');
    if(!axisTarget)continue;
    testedPanels.push({filename,id:panel.id,title:panel.title});
    assert.equal(panel.fieldConfig.defaults.custom.spanNulls,false);
    const hidden=panel.fieldConfig.overrides.find(o=>o.matcher.options===axisTarget.legendFormat);
    assert.deepEqual(hidden.properties[0].value,{tooltip:true,viz:true,legend:true});
    const matcher=req(36638).sJ.get(req(57866).Ct.byName).get(hidden.matcher.options);
    const ranges=[...[1,6].flatMap(hours=>[15000,30000,60000].map(step=>[hours,step])),[720,3600000]];
    for(const [hours,step] of ranges)for(const includeName of [true,false]){
      const count=hours*3600000/step+1;
      const times=Array.from({length:count},(_,i)=>1792112400000+i*step);
      const left=Math.floor(count/3),right=Math.floor(count*5/6);
      const sparseIndices=times.map((_,i)=>i).filter(i=>i<=left||i>=right);
      const metricTimes=sparseIndices.map(i=>times[i]);
      const source=[
        frame('fl2va / falling','A',metricTimes,sparseIndices.map(i=>i<=left?108:9.9),includeName),
        frame('fl2va / flat','B',metricTimes,metricTimes.map(()=>.24),includeName),
        frame('fl2va / zero','A',metricTimes,metricTimes.map(()=>0),includeName),
        frame('background / continuous','A',times,times.map(()=>.001),includeName),
        frame('explicit / null','A',times,times.map((_,i)=>i>left&&i<right?null:.06),includeName),
        frame(axisTarget.legendFormat,'Z',times,times.map(()=>0),includeName),
      ];
      let result=structuredClone(source);
      for(const config of panel.transformations)result=transform(config,result);
      assert.equal(result.length,1);
      const output=result[0],timeField=output.fields.find(f=>f.type==='time');
      assert.deepEqual(Array.from(timeField.values),times);
      assert.equal(output.fields.length,7);
      const axisField=output.fields.find(f=>f.config.displayNameFromDS===axisTarget.legendFormat);
      assert.equal(matcher(axisField,output,result),true);
      for(const metric of source.slice(0,-1)){
        const field=output.fields.find(f=>f.config.displayNameFromDS===metric.fields[1].config.displayNameFromDS);
        assert.ok(field,'legend and series must be preserved');
        assert.equal(field.labels.pool,metric.fields[1].labels.pool);
        assert.equal(matcher(field,output,result),false);
        const expected=new Map(metric.fields[0].values.map((t,i)=>[t,metric.fields[1].values[i]]));
        for(let i=0;i<times.length;i++)assert.equal(field.values[i],expected.get(times[i])??null);
        assert.equal(Array.from(field.values).some(v=>v===undefined),false);
        for(const width of [320,1000]){
          const paths=linePaths(times,field.values,width);
          if(field.config.displayNameFromDS==='background / continuous'){
            assert.equal(paths.gaps.length,0);
          }else{
            assert.ok(paths.gaps.some(g=>g[1]-g[0]>width*.4),'idle half-range must be clipped');
            assert.ok(paths.clip,'renderer must clip the long connecting line');
          }
        }
      }
      cases++;
    }
    // Same displayed legend in two namespaces must remain two distinct series.
    const duplicates=[
      frame('same legend','A',[0,15000,600000,615000],[108,108,9.9,9.9]),
      frame('same legend','A',[0,15000,600000,615000],[100,100,12,12]),
      frame(axisTarget.legendFormat,'Z',shortTimes,shortTimes.map(()=>0)),
    ];
    duplicates[0].fields[1].labels.namespace='one';
    duplicates[1].fields[1].labels.namespace='two';
    let distinct=structuredClone(duplicates);
    for(const config of panel.transformations)distinct=transform(config,distinct);
    assert.equal(distinct[0].fields.length,4);
    for(const [namespace,first,last] of [['one',108,9.9],['two',100,12]]){
      const field=distinct[0].fields.find(f=>f.labels?.namespace===namespace);
      assert.ok(field,'same-legend series must preserve label identity');
      assert.equal(field.values[0],first);
      assert.equal(field.values.at(-1),last);
      assert.equal(Array.from(field.values).filter(v=>v===null).length,38);
    }
    let empty=[frame(axisTarget.legendFormat,'Z',shortTimes,shortTimes.map(()=>0))];
    for(const config of panel.transformations)empty=transform(config,empty);
    assert.equal(empty[0].fields.length,2,'no metric result may create fake zero-valued series');
    assert.equal(matcher(empty[0].fields[1],empty[0],empty),true);
  }
}
assert.equal(testedPanels.length,12);
console.log(JSON.stringify({status:'PASS',grafana_version:'12.2.1',
  loaded_modules:Object.keys(modules).length,old_bug:{undefined_cells:38,null_cells:0,rendered_gaps:0},
  cases,panel_count:testedPanels.length,testedPanels,
  checks:['strict null fill','actual uPlot gap clipping','1h, 6h and 30d ranges',
    'actual Grafana auto interval: 15s for 1h/6h, 30s for 30d',
    '15s, 30s, 60s query steps','flat and sloping sparse series',
    'continuous sibling series','source nulls and real zero values',
    'legend and label preservation','same-legend series in different namespaces',
    'hidden timeline','empty result'],
},null,2));
