const {BalloonSim}=require('../src/sim.js');
const n=+process.argv[2]||3200;
const s=new BalloonSim({speed:360,caliber:9,radius:0.08,height:1.0,offset:0,gravity:9.81,snap:30,count:n,energy:1,mist:1,cavity:0.5,seed:7});
const a=Date.now(); s.advanceTo(s.tImpact+0.8,1e9); const ms=Date.now()-a; console.log(n,'steps',s.steps,'ms/step',(ms/s.steps).toFixed(2),'total',ms);
