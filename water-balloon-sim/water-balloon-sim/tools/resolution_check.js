const {BalloonSim}=require('../src/sim.js');
const n=+process.argv[2];
const s=new BalloonSim({speed:360,caliber:9,radius:0.08,height:1.0,offset:0,gravity:9.81,snap:30,count:n,energy:1,mist:1,cavity:0.5,seed:7});
for(const t of [1,5,20,50,150]){s.advanceTo(s.tImpact+t*1e-3,1e9);
 let N=0,c=[0,0,0];for(let i=0;i<s.N;i++)if(s.alive[i]&&s.nb[i]>=6){N++;for(let a=0;a<3;a++)c[a]+=s.pos[i*3+a];}c=c.map(v=>v/N);
 let sd=0,ke=0;for(let i=0;i<s.N;i++)if(s.alive[i]&&s.nb[i]>=6){for(let a=0;a<3;a++)sd+=(s.pos[i*3+a]-c[a])**2;}
 let M=0;for(let i=0;i<s.N;i++)if(s.alive[i]){const v2=s.vel[i*3]**2+s.vel[i*3+1]**2+s.vel[i*3+2]**2;ke+=v2;M++;}
 console.log(n,'t',t,'bulk',N,'/',s.N,'rms',Math.sqrt(sd/N).toFixed(3),'KE/mass(J/kg)',(0.5*ke/s.N).toFixed(2),'fx',s.fx.n);}
