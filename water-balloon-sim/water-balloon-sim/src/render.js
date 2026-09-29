// ---------------------------------------------------------------------------
// WebGL2 renderer.  Pass 1 ray-traces the scene (water iso-surface from a 3D
// density field with refraction/TIR, latex membrane, bullet, studio set) into an
// HDR target with depth.  Pass 2 draws spray drops + mist as depth-tested
// impostors.  Pass 3 tone-maps.
// ---------------------------------------------------------------------------
const TEX_MAX = 144;

const GLSL_COMMON = `#version 300 es
precision highp float;
precision highp sampler3D;
uniform vec3 uCamPos, uCamRight, uCamUp, uCamFwd;
uniform float uTanHalf, uAspect;
uniform mat4 uViewProj;
uniform vec3 uL; uniform float uLInt;        // key light direction (towards light), intensity
uniform vec3 uFillDir;
uniform vec3 uKeyC, uKeyU, uKeyV; uniform vec2 uKeyHalf;
uniform vec3 uFillC, uFillU, uFillV;
uniform vec3 uFocus;                          // balloon centre (set dressing is placed around it)
uniform sampler2D uWet; uniform float uWetExt;

float hash13(vec3 p){ p = fract(p*0.1031); p += dot(p, p.zyx+31.32); return fract((p.x+p.y)*p.z); }
float vnoise(vec3 p){
  vec3 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
  float n000=hash13(i), n100=hash13(i+vec3(1,0,0)), n010=hash13(i+vec3(0,1,0)), n110=hash13(i+vec3(1,1,0));
  float n001=hash13(i+vec3(0,0,1)), n101=hash13(i+vec3(1,0,1)), n011=hash13(i+vec3(0,1,1)), n111=hash13(i+vec3(1,1,1));
  return mix(mix(mix(n000,n100,f.x),mix(n010,n110,f.x),f.y), mix(mix(n001,n101,f.x),mix(n011,n111,f.x),f.y), f.z);
}
float fbm(vec3 p){ float a=0.5, s=0.0; for(int i=0;i<5;i++){ s+=a*vnoise(p); p*=2.03; a*=0.5; } return s; }
float schlick(float c, float f0){ return f0 + (1.0-f0)*pow(clamp(1.0-c,0.0,1.0),5.0); }

// soft-edged rectangular area light, as seen along a ray
float rectLight(vec3 ro, vec3 rd, vec3 c, vec3 u, vec3 v, vec2 hs, float rough){
  vec3 n = normalize(cross(u, v));
  float dn = dot(rd, n);
  if (abs(dn) < 1e-4) return 0.0;
  float t = dot(c - ro, n) / dn;
  if (t <= 0.0) return 0.0;
  vec3 p = ro + rd*t - c;
  vec2 q = vec2(dot(p,u), dot(p,v)) / hs;
  float e = 0.03 + rough*1.6;
  float m = (1.0 - smoothstep(1.0-e, 1.0+e, abs(q.x))) * (1.0 - smoothstep(1.0-e, 1.0+e, abs(q.y)));
  // diffuser fabric: slightly brighter centre
  return m * (0.8 + 0.2*(1.0 - dot(q,q)*0.5));
}

// ---- static set: grid board, concrete floor, dark stage ------------------------
const float BOARD_Z = -1.35;
float gPixA = 0.001, gPath = 0.0;   // ray-cone spread (rad) and path length so far: texture filtering
float gridLine(float d, float hw, float fw){
  // filtered thin line: fades to its mean coverage when the footprint is wider than the line
  float w = max(fw, hw);
  return (1.0 - smoothstep(hw*0.5, hw*0.5 + w, d)) * clamp(hw / w, 0.0, 1.0);
}
vec3 boardAlbedo(vec2 p, float fw){
  float a = 0.40;
  vec2 g = abs(fract(p/0.05 + 0.5) - 0.5)*0.05;
  float minor = gridLine(min(g.x, g.y), 0.0016, fw);
  minor = mix(minor, 0.064, smoothstep(0.004, 0.02, fw));
  vec2 G = abs(fract(p/0.25 + 0.5) - 0.5)*0.25;
  float major = gridLine(min(G.x, G.y), 0.0032, fw);
  major = mix(major, 0.026, smoothstep(0.01, 0.06, fw));
  a *= 1.0 - 0.30*minor; a *= 1.0 - 0.5*major;
  a *= 0.92 + 0.08*vnoise(vec3(p*9.0, 1.0));
  return vec3(a*0.97, a*0.98, a);
}
vec3 floorAlbedo(vec2 p){
  float n = fbm(vec3(p*3.1, 0.0)), n2 = vnoise(vec3(p*40.0, 2.0));
  float c = 0.13 + 0.06*n + 0.015*n2;
  vec2 s = abs(fract(p/1.2) - 0.5);
  c *= 1.0 - 0.35*(1.0 - smoothstep(0.0, 0.004, 0.5 - max(s.x, s.y)));
  return vec3(c*1.02, c, c*0.96);
}
float wetAt(vec2 xz){
  vec2 uv = xz / (2.0*uWetExt) + 0.5;
  if (any(lessThan(uv, vec2(0))) || any(greaterThan(uv, vec2(1)))) return 0.0;
  return texture(uWet, uv).r;
}
float stageFalloff(vec3 p){ vec3 d = p - vec3(uFocus.x, 0.0, uFocus.z); return exp(-dot(d.xz,d.xz)*0.09); }
vec3 roomColor(vec3 rd){
  return mix(vec3(0.035,0.036,0.040), vec3(0.16,0.165,0.175), smoothstep(-0.4, 0.9, rd.y));
}
vec3 ambientAt(vec3 n){ return mix(vec3(0.030,0.030,0.028), vec3(0.075,0.080,0.090), n.y*0.5+0.5); }

// Direct light on a diffuse surface (no occlusion)
vec3 lightDiffuse(vec3 n){
  float k = max(dot(n, uL), 0.0) * uLInt * 2.6;
  float f = max(dot(n, uFillDir), 0.0) * 0.35;
  return vec3(1.0,0.97,0.93)*k + vec3(0.85,0.9,1.0)*f;
}

// radiance of the environment (everything except the dynamic objects) along a ray
vec3 envRadiance(vec3 ro, vec3 rd, float rough){
  float best = 1e9; vec3 col = roomColor(rd);
  float kl = rectLight(ro, rd, uKeyC, uKeyU, uKeyV, uKeyHalf, rough);
  float fl = rectLight(ro, rd, uFillC, uFillU, uFillV, vec2(0.45,0.45), rough);
  // floor
  if (rd.y < -1e-4){
    float t = -ro.y / rd.y;
    if (t > 0.0 && t < best){
      best = t; vec3 p = ro + rd*t;
      vec3 a = floorAlbedo(p.xz);
      float w = clamp(wetAt(p.xz), 0.0, 1.0);
      a *= mix(1.0, 0.5, w);
      col = a * (lightDiffuse(vec3(0,1,0)) + ambientAt(vec3(0,1,0))) * stageFalloff(p);
    }
  }
  // grid board behind the balloon
  if (rd.z < -1e-4){
    float t = (BOARD_Z - ro.z) / rd.z;
    if (t > 0.0 && t < best){
      vec3 p = ro + rd*t;
      if (abs(p.x - uFocus.x) < 1.7 && p.y > 0.0 && p.y < 2.3){
        best = t;
        vec3 n = vec3(0,0,1);
        col = boardAlbedo(p.xy, gPixA*(gPath + t)) * (lightDiffuse(n)*0.8 + ambientAt(n)) * (0.55 + 0.45*stageFalloff(p));
      }
    }
  }
  // area lights are in front of everything they overlap
  col += vec3(1.0,0.97,0.92) * kl * uLInt * 16.0 + vec3(0.8,0.88,1.0) * fl * 2.2;
  return col;
}
`;

const TRACE_FS = GLSL_COMMON + `
in vec2 vUv;
out vec4 outColor;
uniform sampler3D uDens;
uniform vec3 uGridMin, uTexWorld, uBoxMin, uBoxMax;
uniform float uIso, uVoxel, uKR, uHasWater, uAerate, uPixA; uniform int uDbg;
uniform vec3 uBulletTip; uniform float uBulletR, uBulletLen, uBulletOn;
uniform vec3 uMemC; uniform float uMemR, uMemRy, uMemOn;
uniform vec3 uE1, uE2; uniform float uTh1, uTh2, uAlpha;
uniform vec3 uLatex;
uniform float uRemnant;
uniform float uStringTop;

// ---------------- water ----------------
float dens(vec3 p){ return texture(uDens, (p - uGridMin) / uTexWorld).r; }
uniform highp sampler3D uFoam; uniform float uFoamOn; uniform vec3 uBlobC; uniform float uRip;
float foamAt(vec3 p){ return uFoamOn > 0.5 ? texture(uFoam, (p - uGridMin) / (uTexWorld)).r : 0.0; }
vec3 waterNs(vec3 p, float e){
  vec3 g = vec3(dens(p+vec3(e,0,0)) - dens(p-vec3(e,0,0)), dens(p+vec3(0,e,0)) - dens(p-vec3(0,e,0)), dens(p+vec3(0,0,e)) - dens(p-vec3(0,0,e)));
  return -normalize(g + vec3(1e-7));
}
vec3 waterN(vec3 p){
  float e = uVoxel * 1.4;
  vec3 g = vec3(dens(p+vec3(e,0,0)) - dens(p-vec3(e,0,0)), dens(p+vec3(0,e,0)) - dens(p-vec3(0,e,0)), dens(p+vec3(0,0,e)) - dens(p-vec3(0,0,e)));
  return -normalize(g + vec3(1e-7));
}
bool boxHit(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax, out float t0, out float t1){
  vec3 inv = 1.0 / rd;
  vec3 a = (bmin - ro) * inv, b = (bmax - ro) * inv;
  vec3 lo = min(a,b), hi = max(a,b);
  t0 = max(max(lo.x, lo.y), lo.z); t1 = min(min(hi.x, hi.y), hi.z);
  return t1 > max(t0, 0.0);
}
float waterEnter(vec3 ro, vec3 rd, float tmax){
  if (uHasWater < 0.5) return -1.0;
  float t0, t1;
  if (!boxHit(ro, rd, uBoxMin, uBoxMax, t0, t1)) return -1.0;
  t0 = max(t0, 0.0); t1 = min(t1, tmax);
  float t = t0, pt = t0;
  for (int i = 0; i < 220; i++){
    if (t > t1) return -1.0;
    float d = dens(ro + rd*t);
    if (d > uIso){
      float a = pt, b = t;
      for (int k = 0; k < 6; k++){ float m = 0.5*(a+b); if (dens(ro+rd*m) > uIso) b = m; else a = m; }
      return b;
    }
    pt = t;
    t += d < 0.004 ? 0.27*uKR : 0.5*uVoxel;
  }
  return -1.0;
}
float waterExit(vec3 ro, vec3 rd){
  float t0, t1;
  boxHit(ro, rd, uBoxMin, uBoxMax, t0, t1);
  float t = 0.4*uVoxel, pt = 0.0;
  for (int i = 0; i < 200; i++){
    if (t > t1) return t1;
    float d = dens(ro + rd*t);
    if (d < uIso){
      float a = pt, b = t;
      for (int k = 0; k < 6; k++){ float m = 0.5*(a+b); if (dens(ro+rd*m) < uIso) b = m; else a = m; }
      return b;
    }
    pt = t;
    t += 0.55*uVoxel;
  }
  return t;
}

// ---------------- latex membrane (ellipsoid with propagating tears) ----------------
float tearNoise(vec3 d){ return (vnoise(d*7.0) - 0.5)*0.22 + (vnoise(d*23.0) - 0.5)*0.09; }
// returns >0 if intact; value = angular distance to the nearest tear edge
float memIntact(vec3 d){
  if (uMemOn < 0.5) return -1.0;
  float n = tearNoise(d);
  float m = 10.0;
  if (uTh1 > 0.0) m = min(m, acos(clamp(dot(d, uE1), -1.0, 1.0)) - (uTh1 + n));
  if (uTh2 > 0.0) m = min(m, acos(clamp(dot(d, uE2), -1.0, 1.0)) - (uTh2 + n*1.2));
  m = min(m, (uAlpha + n*0.8) - acos(clamp(d.y, -1.0, 1.0)));
  return m;
}
vec2 ellipsoidHits(vec3 ro, vec3 rd){
  vec3 sc = vec3(uMemR, uMemR*uMemRy, uMemR);
  vec3 o = (ro - uMemC) / sc, d = rd / sc;
  float a = dot(d,d), b = dot(o,d), c = dot(o,o) - 1.0;
  float h = b*b - a*c;
  if (h < 0.0) return vec2(-1.0);
  h = sqrt(h);
  return vec2((-b - h)/a, (-b + h)/a);
}
vec3 memDirAt(vec3 p){ vec3 sc = vec3(uMemR, uMemR*uMemRy, uMemR); return normalize((p - uMemC)/sc); }
float memEl(vec3 p){ vec3 sc = vec3(uMemR, uMemR*uMemRy, uMemR); return length((p - uMemC)/sc); }
vec3 memNormalAt(vec3 p){ vec3 sc = vec3(uMemR, uMemR*uMemRy, uMemR); return normalize((p - uMemC)/(sc*sc)); }

// neck + knot + (after the pop) the crumpled rubber that stays on the knot
vec3 knotBase(){ return uMemC + vec3(0.0, uMemR*uMemRy*0.985, 0.0); }
float sdCappedCone(vec3 p, vec3 a, vec3 b, float ra, float rb){
  float rba = rb-ra; float baba = dot(b-a,b-a); float papa = dot(p-a,p-a); float paba = dot(p-a,b-a)/baba;
  float x = sqrt(max(papa - paba*paba*baba, 0.0));
  float cax = max(0.0, x - ((paba<0.5)?ra:rb));
  float cay = abs(paba-0.5)-0.5;
  float k = rba*rba + baba;
  float f = clamp((rba*(x-ra)+paba*baba)/k, 0.0, 1.0);
  float cbx = x-ra - f*rba; float cby = paba - f;
  float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
  return s*sqrt(min(cax*cax + cay*cay*baba, cbx*cbx + cby*cby*baba));
}
float sdKnot(vec3 p){
  vec3 kb = knotBase();
  float s = uMemR / 0.08;
  // neck: a trumpet that flares smoothly into the skin (the balloon's teardrop top)
  float yy = (p.y - kb.y) / s;
  float rr = 0.0036 + 0.024*exp(-(yy + 0.006)/0.0045);
  float d = max(length(p.xz - kb.xz)/s - rr, max(-0.004 - yy, yy - 0.017)) * s * 0.55;
  vec3 q = p - (kb + vec3(0.0, 0.022, 0.0)*s);
  d = min(d, length(q*vec3(1.0,1.25,1.0)) - 0.0065*s + 0.0008*s*vnoise(q*500.0/s));
  if (uRemnant > 0.0){
    // torn rubber skirt left on the neck: a thin flared shell with a ragged lower edge
    float L = 0.024*s*uRemnant;
    vec3 top = kb + vec3(0.0, 0.002*s, 0.0), bot = kb - vec3(0.0, L, 0.0);
    vec3 q2 = p - kb;
    float ang = atan(q2.z, q2.x);
    float wob = (vnoise(vec3(cos(ang)*3.0, sin(ang)*3.0, 1.7)) - 0.5);
    float sh = abs(sdCappedCone(p + vec3(wob*0.004*s, 0.0, 0.0), bot, top, 0.012*s*(0.8 + 0.5*wob), 0.0055*s)) - 0.0007*s;
    float edge = (kb.y - L*(0.55 + 0.9*vnoise(vec3(cos(ang)*5.0, sin(ang)*5.0, 3.1)))) - p.y;
    d = min(d, max(sh, edge) * 0.8);
  }
  return d;
}
float knotHit(vec3 ro, vec3 rd, float tmax){
  vec3 c = knotBase() + vec3(0.0, 0.01, 0.0)*(uMemR/0.08);
  float br = 0.045*(uMemR/0.08);
  vec3 oc = ro - c; float b = dot(oc, rd); float h = b*b - dot(oc,oc) + br*br;
  if (h < 0.0) return -1.0;
  h = sqrt(h);
  float t = max(-b - h, 0.0), te = min(-b + h, tmax);
  for (int i = 0; i < 64; i++){
    if (t > te) return -1.0;
    float d = sdKnot(ro + rd*t);
    if (d < 0.00012) return t;
    t += d;
  }
  return -1.0;
}
vec3 knotN(vec3 p){
  vec2 e = vec2(0.0002, 0.0);
  return normalize(vec3(sdKnot(p+e.xyy)-sdKnot(p-e.xyy), sdKnot(p+e.yxy)-sdKnot(p-e.yxy), sdKnot(p+e.yyx)-sdKnot(p-e.yyx)));
}

// ---------------- string ----------------
float stringHit(vec3 ro, vec3 rd, float tmax, float r){
  vec3 kb = knotBase() + vec3(0.0, 0.026*(uMemR/0.08), 0.0);
  vec2 o = ro.xz - kb.xz, d = rd.xz;
  float a = dot(d,d), b = dot(o,d), c = dot(o,o) - r*r;
  float h = b*b - a*c;
  if (h < 0.0 || a < 1e-8) return -1.0;
  float t = (-b - sqrt(h))/a;
  if (t < 0.0 || t > tmax) return -1.0;
  float y = ro.y + rd.y*t;
  if (y < kb.y || y > uStringTop) return -1.0;
  return t;
}

// ---------------- bullet: jacketed round-nose / spitzer profile ----------------
float sdBullet(vec3 p){
  vec3 q = p - uBulletTip;           // x axis = flight direction, tip at origin
  float u = -q.x;                    // distance back from the tip
  float r = length(q.yz);
  float L = uBulletLen, R = uBulletR, nose = L*0.42;
  float prof;
  if (u < nose){ float t = 1.0 - u/nose; prof = R*sqrt(max(0.0, 1.0 - t*t*t)); }
  else prof = R * (1.0 - 0.08*smoothstep(L*0.85, L, u));
  float d = (r - prof) * 0.7;
  d = max(d, -u);
  d = max(d, u - L);
  return d;
}
float bulletHit(vec3 ro, vec3 rd, float tmax){
  if (uBulletOn < 0.5) return -1.0;
  vec3 c = uBulletTip - vec3(uBulletLen*0.5, 0.0, 0.0);
  float br = uBulletLen*0.6;
  vec3 oc = ro - c; float b = dot(oc, rd); float h = b*b - dot(oc,oc) + br*br;
  if (h < 0.0) return -1.0;
  h = sqrt(h);
  float t = max(-b - h, 0.0), te = min(-b + h, tmax);
  for (int i = 0; i < 80; i++){
    if (t > te) return -1.0;
    float d = sdBullet(ro + rd*t);
    if (d < 0.00004) return t;
    t += max(d, 0.00003);
  }
  return -1.0;
}
vec3 bulletN(vec3 p){
  vec2 e = vec2(0.00006, 0.0);
  return normalize(vec3(sdBullet(p+e.xyy)-sdBullet(p-e.xyy), sdBullet(p+e.yxy)-sdBullet(p-e.yxy), sdBullet(p+e.yyx)-sdBullet(p-e.yyx)));
}

// ---------------- shadows onto the floor / board ----------------
vec3 shadowAt(vec3 p){
  vec3 s = vec3(1.0);
  // membrane
  if (uMemOn > 0.5){
    vec2 h = ellipsoidHits(p, uL);
    for (int k = 0; k < 2; k++){
      float t = k == 0 ? h.x : h.y;
      if (t > 0.0){ vec3 d = memDirAt(p + uL*t); if (memIntact(d) > 0.0) s *= mix(vec3(1.0), uLatex*0.9 + 0.05, 0.6); }
    }
  }
  // water: dark rim, focused caustic core
  if (uHasWater > 0.5){
    float t = waterEnter(p, uL, 10.0);
    if (t > 0.0){
      vec3 q = p + uL*t;
      vec3 n = waterNs(q, uVoxel*2.5);
      float c = max(dot(n, uL), 0.0);
      s *= 0.38 + 1.1*pow(c, 4.0);
    }
  }
  // knot / rubber
  if (knotHit(p, uL, 10.0) > 0.0) s *= 0.3;
  return s;
}

// shade the static set with shadows (primary rays only)
vec3 shadeFloor(vec3 p, vec3 rd){
  vec3 a = floorAlbedo(p.xz);
  float w = wetAt(p.xz);
  // sharp-edged stain and standing film (a thin film has a crisp contact line, not a soft cloud)
  float nz = vnoise(vec3(p.xz*22.0, 3.0));
  float wet = smoothstep(0.10, 0.17, w + (nz - 0.5)*0.10);
  float film = smoothstep(0.50, 0.62, w + (nz - 0.5)*0.35);
  a *= mix(1.0, 0.5, wet);
  vec3 n = vec3(0,1,0);
  if (film > 0.0 || wet > 0.0){
    // meniscus at the contact line: tilt the normal along the film-thickness gradient
    float e = 2.0*uWetExt/256.0;
    float gx = wetAt(p.xz + vec2(e,0)) - wetAt(p.xz - vec2(e,0));
    float gz = wetAt(p.xz + vec2(0,e)) - wetAt(p.xz - vec2(0,e));
    float edge = film*(1.0 - film)*4.0;
    vec2 g = vec2(gx, gz) * (0.35 + 2.5*edge);
    // faint capillary ripples on the film
    float h0 = vnoise(vec3(p.xz*60.0, 0.0));
    vec2 rip = vec2(vnoise(vec3((p.xz+vec2(0.002,0))*60.0,0.0)) - h0, vnoise(vec3((p.xz+vec2(0,0.002))*60.0,0.0)) - h0) * 0.18 * film;
    n = normalize(vec3(-g.x - rip.x, 1.0, -g.y - rip.y));
  }
  vec3 sh = shadowAt(p);
  float k = max(dot(n, uL), 0.0) * uLInt * 2.6;
  vec3 col = a * (vec3(1.0,0.97,0.93)*k*sh + vec3(0.85,0.9,1.0)*max(dot(n,uFillDir),0.0)*0.35 + ambientAt(n));
  // wet concrete is glossy; the standing film is a clean mirror
  float F = schlick(max(dot(-rd, n), 0.0), 0.02);
  float refl = mix(0.35*wet, 1.0, film);
  vec3 r = reflect(rd, n);
  col += refl * F * envRadiance(p + n*0.001, r, mix(0.3, 0.0, film)) * (0.5 + 0.5*sh);
  // sharp key-light glint on the film
  col += film * pow(max(dot(r, uL), 0.0), 900.0) * uLInt * 40.0 * sh;
  // dry concrete sheen
  col += (1.0-wet) * 0.012 * pow(max(dot(r, uL), 0.0), 8.0) * uLInt;
  return col * stageFalloff(p);
}
vec3 shadeBoard(vec3 p, float fw){
  vec3 n = vec3(0,0,1);
  vec3 sh = shadowAt(p);
  float k = max(dot(n, uL), 0.0) * uLInt * 2.6;
  return boardAlbedo(p.xy, fw) * ((vec3(1.0,0.97,0.93)*k*sh + vec3(0.85,0.9,1.0)*max(dot(n,uFillDir),0.0)*0.35)*0.8 + ambientAt(n)) * (0.55 + 0.45*stageFalloff(p));
}

void main(){
  vec2 ndc = vUv*2.0 - 1.0;
  vec3 ro = uCamPos;
  vec3 rd = normalize(uCamFwd + ndc.x*uTanHalf*uAspect*uCamRight + ndc.y*uTanHalf*uCamUp);
  vec3 col = vec3(0.0), thr = vec3(1.0);
  float depth = 1.0; bool first = true;
  bool done = false; bool skipW = false; bool fromSkin = false;
  gPixA = uPixA; gPath = 0.0;
  for (int bounce = 0; bounce < 8 && !done; bounce++){
    float tMax = 1e6;
    // candidate surfaces
    float tFloor = rd.y < -1e-5 ? -ro.y/rd.y : -1.0;
    float tBoard = -1.0;
    if (rd.z < -1e-5){ float t = (BOARD_Z - ro.z)/rd.z; vec3 p = ro + rd*t; if (t > 0.0 && abs(p.x - uFocus.x) < 1.7 && p.y > 0.0 && p.y < 2.3) tBoard = t; }
    float tEnv = 1e6;
    if (tFloor > 0.0) tEnv = min(tEnv, tFloor);
    if (tBoard > 0.0) tEnv = min(tEnv, tBoard);

    int kind = 0; float tHit = tEnv;
    if (tFloor > 0.0 && tFloor <= tEnv) kind = 1;
    if (tBoard > 0.0 && tBoard <= tEnv) kind = 2;
    float tb = bulletHit(ro, rd, tHit); if (tb > 0.0 && tb < tHit){ tHit = tb; kind = 3; }
    float tk = knotHit(ro, rd, tHit); if (tk > 0.0 && tk < tHit){ tHit = tk; kind = 4; }
    // string: analytic pixel coverage of a thin vertical cylinder (no aliasing)
    {
      vec3 kb = knotBase() + vec3(0.0, 0.026*(uMemR/0.08), 0.0);
      vec2 o2 = ro.xz - kb.xz, d2 = rd.xz;
      float dd = dot(d2, d2);
      if (dd > 1e-8){
        float ts = -dot(o2, d2) / dd;
        float ys = ro.y + rd.y*ts;
        if (ts > 0.0 && ts < tHit && ys > kb.y && ys < uStringTop){
          float dist = length(o2 + d2*ts);
          float fw = gPixA*(gPath + ts);
          float rS = 0.0007;
          float cov = clamp((rS + 0.5*fw - dist) / fw, 0.0, 1.0) * min(1.0, 2.0*rS/fw);
          if (cov > 0.0){
            vec3 ns = normalize(vec3(o2.x + d2.x*ts, 0.0, o2.y + d2.y*ts) - vec3(rd.x, 0.0, rd.z)*0.5*rS);
            vec3 cs = vec3(0.72,0.70,0.66) * (lightDiffuse(ns)*0.8 + ambientAt(ns) + 0.25);
            col += thr * cs * cov; thr *= 1.0 - cov;
          }
        }
      }
    }
    // membrane: nearest intact shell crossing
    float tm = -1.0; float edge = 0.0;
    if (uMemOn > 0.5 && uDbg != 1 && uDbg != 2){
      vec2 h = ellipsoidHits(ro, rd);
      for (int k = 0; k < 2; k++){
        float t = k == 0 ? h.x : h.y;
        if (t > 1e-5 && t < tHit){ float m = memIntact(memDirAt(ro + rd*t)); if (m > 0.0){ tm = t; edge = m; break; } }
      }
      if (tm > 0.0){ tHit = tm; kind = 5; }
    }
    float tw = skipW ? -1.0 : waterEnter(ro, rd, tHit);
    // just passed inward through the rubber into water that fills it: the water surface is the skin
    if (fromSkin && !skipW && uHasWater > 0.5 && (dens(ro) > uIso*0.8 || dens(ro + rd*0.012) > uIso)) tw = 1e-5;
    fromSkin = false;
    skipW = false;
    if (tw > 0.0 && tw < tHit){
      // water bulging through an intact rubber skin is not possible: the skin is the boundary
      vec3 pw = ro + rd*tw;
      bool clipped = uMemOn > 0.5 && memEl(pw) > 1.001 && memIntact(memDirAt(pw)) > 0.0;
      if (!clipped){ tHit = tw; kind = 7; }
    }

    vec3 p = ro + rd*tHit;
    gPath += tHit;
    if (first && kind != 0){ vec4 cl = uViewProj*vec4(p,1.0); depth = clamp(cl.z/cl.w*0.5+0.5, 0.0, 1.0); first = false; }

    if (kind == 0){ col += thr * envRadiance(ro, rd, 0.0); done = true; }
    else if (kind == 1){
      // area-light reflections are visible when looking at the floor through the scene
      col += thr * shadeFloor(p, rd); done = true;
    }
    else if (kind == 2){ col += thr * shadeBoard(p, gPixA*gPath); done = true; }
    else if (kind == 3){
      vec3 n = bulletN(p);
      vec3 F0 = vec3(0.96, 0.64, 0.50);
      float c = max(dot(-rd, n), 0.0);
      vec3 F = F0 + (1.0-F0)*pow(1.0-c, 5.0);
      vec3 r = reflect(rd, n);
      vec3 e = envRadiance(p + n*1e-4, r, 0.22);
      col += thr * (F * e + F0 * 0.04 * lightDiffuse(n));
      done = true;
    }
    else if (kind == 4 || kind == 6){
      vec3 n = kind == 4 ? knotN(p) : normalize(vec3(p.x - knotBase().x, 0.0, p.z - knotBase().z));
      vec3 alb = kind == 4 ? uLatex*0.75 : vec3(0.72,0.70,0.66);
      float c = max(dot(-rd, n), 0.0);
      float F = schlick(c, 0.045);
      vec3 sh = vec3(1.0);
      vec3 cc = alb * (lightDiffuse(n)*sh + ambientAt(n)) * (1.0-F) + F * (kind == 6 ? 0.0 : 1.0) * envRadiance(p + n*1e-4, reflect(rd, n), 0.25);
      col += thr * cc; done = true;
    }
    else if (kind == 5){
      vec3 n = memNormalAt(p);
      vec3 d = memDirAt(p);
      if (dot(n, rd) > 0.0) n = -n;
      float rim = 1.0 - smoothstep(0.0, 0.07, edge);          // rolled-up rubber at the tear edge
      float thick = mix(0.6, 1.0, smoothstep(0.55, 1.0, d.y)); // thicker towards the neck
      float a = clamp(0.42*thick + 0.5*rim, 0.0, 0.96);
      float c = max(dot(-rd, n), 0.0);
      float F = schlick(c, 0.045);
      vec3 spec = F * envRadiance(p + n*1e-4, reflect(rd, n), 0.12);
      vec3 diff = uLatex * (lightDiffuse(n) + ambientAt(n)) * a;
      vec3 back = uLatex * max(dot(rd, uL), 0.0) * uLInt * 0.9 * (1.0 - a*0.5);   // light through the rubber
      col += thr * (spec + diff*0.9 + back*0.35);
      thr *= (1.0 - F) * mix(vec3(1.0), uLatex*1.05, clamp(a*1.6, 0.0, 1.0)) * (1.0 - a*0.55);
      fromSkin = dot(rd, memNormalAt(p)) < 0.0;
      ro = p + rd*2e-4;
    }
    else if (kind == 7 && uDbg == 2){ col = waterN(p)*0.5+0.5; col*=col; done = true; }
    else if (kind == 7){
      // water surface: reflect (single bounce to environment) + refract through the volume
      vec3 n = waterN(p);
      if (uMemOn > 0.5){
        float el = memEl(p);
        if (el > 0.9 && memIntact(memDirAt(p)) > 0.0) n = normalize(mix(n, memNormalAt(p), smoothstep(0.9, 0.975, el)));
      }
      // capillary ripples riding on the free surface (left by the retracting rubber and the shot),
      // in a frame that moves with the water so they do not swim over the surface
      if (uRip > 0.0){
        vec3 q4 = (p - uBlobC)*190.0;
        float h1 = vnoise(q4);
        vec3 g1 = vec3(vnoise(q4 + vec3(0.3,0,0)) - h1, vnoise(q4 + vec3(0,0.3,0)) - h1, vnoise(q4 + vec3(0,0,0.3)) - h1);
        n = normalize(n + g1 * uRip);
      }
      // splashing water on the floor is covered in capillary ripples and impact roughness
      float fl = 1.0 - smoothstep(0.012, 0.05, p.y);
      if (fl > 0.0){
        vec3 q3 = p*140.0;
        float h0 = vnoise(q3);
        vec3 gr = vec3(vnoise(q3 + vec3(0.35,0,0)) - h0, vnoise(q3 + vec3(0,0.35,0)) - h0, vnoise(q3 + vec3(0,0,0.35)) - h0);
        n = normalize(n + gr * 1.6 * fl);
      }
      if (dot(n, rd) > 0.0) n = -n;
      float c = max(dot(-rd, n), 0.0);
      float F = schlick(c, 0.02);
      col += thr * F * envRadiance(p + n*2e-4, reflect(rd, n), 0.0);
      thr *= (1.0 - F);
      vec3 dIn = refract(rd, n, 1.0/1.333);
      vec3 q = p - n*uVoxel*0.35;
      vec3 dcur = dIn;
      float travelled = 0.0, foamPath = 0.0;
      bool escaped = false;
      // bubbly water: a whitish, scattering layer at the surface
      float fS = foamAt(p);
      if (fS > 0.02){
        vec3 lit = lightDiffuse(n)*0.6 + ambientAt(n) + vec3(0.12);
        float cover = clamp(fS*0.35, 0.0, 0.4);
        col += thr * cover * lit * vec3(0.92, 0.95, 0.97);
        thr *= 1.0 - cover;
      }
      for (int k = 0; k < 3; k++){
        // the bullet can be seen through the water
        float te = waterExit(q, dcur);
        foamPath += (foamAt(q + dcur*te*0.25) + foamAt(q + dcur*te*0.75)) * 0.5 * te;
        float tb2 = bulletHit(q, dcur, te);
        if (tb2 > 0.0){
          vec3 bp = q + dcur*tb2; vec3 bn = bulletN(bp);
          vec3 F0 = vec3(0.96,0.64,0.50);
          col += thr * F0 * envRadiance(bp + bn*1e-4, reflect(dcur, bn), 0.25) * exp(-travelled*vec3(0.4,0.07,0.03));
          done = true; escaped = true; break;
        }
        bool onSkin = false;
        if (uMemOn > 0.5){
          vec2 hm = ellipsoidHits(q, dcur);
          float tx = hm.y;
          if (tx > 0.0){
            vec3 pe = q + dcur*tx;
            if ((tx < te || memEl(q + dcur*te) > 0.93) && memIntact(memDirAt(pe)) > 0.0){ te = tx; onSkin = true; }
          }
        }
        if (dcur.y < -1e-4){
          float tf = -q.y / dcur.y;
          if (tf > 0.0 && tf < te){
            travelled += tf;
            vec3 fp = q + dcur*tf;
            // wet concrete under the water, lit through the water surface: light focused by the
            // curved surface above forms caustics (brighter where the surface converges the light)
            vec3 a = floorAlbedo(fp.xz) * 0.6;
            vec3 tr0 = exp(-travelled * vec3(0.45, 0.07, 0.03));
            vec3 sn = waterNs(fp + vec3(0.0, 0.5*uVoxel, 0.0), uVoxel*1.5);
            float caus = 0.75 + 1.1*pow(clamp(dot(sn, uL), 0.0, 1.0), 6.0);
            col += thr * tr0 * a * (lightDiffuse(vec3(0,1,0))*caus + ambientAt(vec3(0,1,0))) * stageFalloff(fp);
            done = true; escaped = true; break;
          }
        }
        travelled += te;
        vec3 e = q + dcur*te;
        vec3 no = onSkin ? memNormalAt(e) : waterN(e);
        if (dot(no, dcur) < 0.0) no = -no;
        vec3 dOut = refract(dcur, -no, 1.333);
        if (dot(dOut, dOut) < 1e-6){ dcur = reflect(dcur, -no); q = e - no*uVoxel*0.35; continue; }
        float Fo = schlick(max(dot(dcur, no), 0.0), 0.02);
        rd = dOut;
        if (onSkin){ ro = e - no*1e-4; skipW = true; } else ro = e + no*uVoxel*0.6;
        thr *= (1.0 - Fo);
        escaped = true; break;
      }
      gPixA *= 4.0; gPath += travelled;
      // Beer-Lambert absorption + aeration (micro-bubbles whiten the water after the hit)
      float optical = travelled*uAerate*10.0 + foamPath*22.0;       // bubble scattering
      vec3 tr = exp(-travelled * vec3(0.45, 0.07, 0.03)) * exp(-optical);
      vec3 inscat = (1.0 - exp(-optical)) * (lightDiffuse(vec3(0,1,0))*0.3 + ambientAt(vec3(0,1,0)) + uLInt*0.2);
      col += thr * inscat * 0.65;
      thr *= tr;
      if (!escaped){ col += thr * envRadiance(ro, rd, 0.3) * 0.5; done = true; }
      if (uDbg == 7 && escaped && !done){ col = rd*0.5+0.5; done = true; }
      if (uDbg == 6){ col = escaped ? vec3(0.0, travelled*5.0, 0.0) : vec3(1.0,0.0,0.0); done = true; }
    }
    if (max(thr.r, max(thr.g, thr.b)) < 0.01) done = true;
  }
  outColor = vec4(col, 1.0);
  gl_FragDepth = depth;
}
`;

const FULL_VS = `#version 300 es
in vec2 aPos; out vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const SPRITE_VS = `#version 300 es
precision highp float;
in vec2 aCorner;
in vec4 aPS;      // xyz, radius
in vec4 aInfo;    // type, alpha, age, spare
uniform mat4 uViewProj; uniform vec3 uCamRight, uCamUp, uCamPos;
uniform float uPixelWorld; // world size of one pixel at distance 1
out vec2 vC; out vec4 vInfo; out vec3 vWorld; out float vCover;
void main(){
  float dist = length(aPS.xyz - uCamPos);
  float minR = uPixelWorld * dist * 0.9;
  float r = max(aPS.w, minR);
  vCover = clamp((aPS.w*aPS.w)/(r*r), 0.0, 1.0);
  if (aInfo.x > 0.5) { r = aPS.w; vCover = 1.0; }
  vec3 w = aPS.xyz + (aCorner.x*uCamRight + aCorner.y*uCamUp) * r;
  vC = aCorner; vInfo = aInfo; vWorld = w;
  gl_Position = uViewProj * vec4(w, 1.0);
}`;

const SPRITE_FS = GLSL_COMMON + `
in vec2 vC; in vec4 vInfo; in vec3 vWorld; in float vCover;
out vec4 outColor;
void main(){
  float r2 = dot(vC, vC);
  if (vInfo.x < 0.5){
    // water droplet: tiny refracting sphere
    if (r2 > 1.0) discard;
    vec3 back = -uCamFwd;
    gPixA = 0.02;
    vec3 n = normalize(vC.x*uCamRight + vC.y*uCamUp + sqrt(1.0 - r2)*back);
    vec3 rd = normalize(vWorld - uCamPos);
    float c = max(dot(-rd, n), 0.0);
    float F = schlick(c, 0.02);
    vec3 refl = envRadiance(vWorld, reflect(rd, n), 0.0);
    vec3 t1 = refract(rd, n, 0.75);
    vec3 t2 = normalize(t1 + (t1 - rd)*1.6);   // exit through the far side of the drop: image flips
    vec3 refr = mix(envRadiance(vWorld, t2, 0.05), envRadiance(vWorld, rd, 0.05), 0.35 + 0.4*(1.0 - vCover));
    vec3 col = F*refl + (1.0-F)*refr*0.95;
    float lobe = mix(60.0, 600.0, vCover);
    col += pow(max(dot(reflect(rd, n), uL), 0.0), lobe) * uLInt * 30.0 * (lobe/600.0 + 0.1);
    float a = vCover * vInfo.y;
    outColor = vec4(col*a, a);
  } else {
    // mist: forward-scattering soft puff
    float fall = exp(-r2*3.2);
    if (fall < 0.01) discard;
    vec3 rd = normalize(vWorld - uCamPos);
    float ph = 0.35 + 1.4*pow(max(dot(rd, uL), 0.0), 6.0) + 0.3*max(dot(-rd, uL),0.0);
    vec3 lit = vec3(0.9,0.92,0.95) * (uLInt*0.9*ph + 0.18);
    float a = fall * vInfo.y;
    outColor = vec4(lit*a, a);
  }
}`;

const COMP_FS = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 outColor;
uniform sampler2D uHdr; uniform float uExposure; uniform vec2 uRes; uniform float uSeed;
vec3 aces(vec3 x){ return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0); }
float h(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233)) + uSeed)*43758.5453); }
void main(){
  vec3 c = textureLod(uHdr, vUv, 0.0).rgb;
  // soft bloom from the brightest glints: smooth mip chain instead of a sparse tap lattice
  vec3 b = max(textureLod(uHdr, vUv, 2.0).rgb - 1.6, 0.0) * 0.05
         + max(textureLod(uHdr, vUv, 3.0).rgb - 1.2, 0.0) * 0.05
         + max(textureLod(uHdr, vUv, 4.5).rgb - 0.9, 0.0) * 0.05;
  c += b;
  c *= uExposure;
  c = aces(c);
  vec2 q = vUv - 0.5;
  c *= 1.0 - 0.28*dot(q,q)*1.6;
  c = pow(c, vec3(1.0/2.2));
  c += (h(gl_FragCoord.xy) - 0.5) * 0.012;
  outColor = vec4(c, 1.0);
}`;

class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = (this.gl = canvas.getContext('webgl2', { antialias: false, alpha: false, premultipliedAlpha: false, preserveDrawingBuffer: true }));
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('EXT_color_buffer_half_float');
    this.progTrace = this.program(FULL_VS, TRACE_FS);
    this.progSprite = this.program(SPRITE_VS, SPRITE_FS);
    this.progComp = this.program(FULL_VS, COMP_FS);
    // full-screen triangle pair
    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    this.vaoFull = gl.createVertexArray();
    gl.bindVertexArray(this.vaoFull);
    for (const p of [this.progTrace, this.progComp]) { const l = gl.getAttribLocation(p, 'aPos'); if (l >= 0) { gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 2, gl.FLOAT, false, 0, 0); } }
    // sprites
    this.spriteCap = 40000;
    this.spriteData = new Float32Array(this.spriteCap * 8);
    this.vaoSprite = gl.createVertexArray();
    gl.bindVertexArray(this.vaoSprite);
    const corner = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, corner);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    let l = gl.getAttribLocation(this.progSprite, 'aCorner');
    gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 2, gl.FLOAT, false, 0, 0);
    this.instBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.spriteData.byteLength, gl.DYNAMIC_DRAW);
    l = gl.getAttribLocation(this.progSprite, 'aPS');
    gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 4, gl.FLOAT, false, 32, 0); gl.vertexAttribDivisor(l, 1);
    l = gl.getAttribLocation(this.progSprite, 'aInfo');
    gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 4, gl.FLOAT, false, 32, 16); gl.vertexAttribDivisor(l, 1);
    gl.bindVertexArray(null);

    // density volume
    this.dens = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, this.dens);
    gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R16F, TEX_MAX, TEX_MAX, TEX_MAX);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
    this.densData = new Float32Array(TEX_MAX * TEX_MAX * TEX_MAX);
    // entrained-air (foam) field on a half-resolution grid aligned with the density grid
    this.foamTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, this.foamTex);
    gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R16F, TEX_MAX / 2, TEX_MAX / 2, TEX_MAX / 2);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
    this.foamData = new Float32Array((TEX_MAX / 2) ** 3);
    this.grid = { has: false };
    // wetness
    this.wetTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.wetTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R16F, WET_RES, WET_RES);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.fbo = null; this.fw = 0; this.fh = 0;
    this.uniCache = new Map();
    this.frame = 0;
  }

  program(vs, fs) {
    const gl = this.gl;
    const mk = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) + '\n' + src.split('\n').map((l, i) => i + 1 + ': ' + l).join('\n').slice(0, 200));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  u(p, name) {
    let m = this.uniCache.get(p); if (!m) { m = new Map(); this.uniCache.set(p, m); }
    if (!m.has(name)) m.set(name, this.gl.getUniformLocation(p, name));
    return m.get(name);
  }

  ensureTargets(w, h) {
    const gl = this.gl;
    if (this.fbo && w === this.fw && h === this.fh) return;
    if (this.fbo) { gl.deleteFramebuffer(this.fbo); gl.deleteTexture(this.hdr); gl.deleteRenderbuffer(this.depthRb); }
    this.fw = w; this.fh = h;
    this.hdr = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.hdr);
    this.hdrLevels = Math.min(6, Math.floor(Math.log2(Math.max(w, h))) + 1);
    gl.texStorage2D(gl.TEXTURE_2D, this.hdrLevels, gl.RGBA16F, w, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.depthRb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.depthRb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.hdr, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.depthRb);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ---- splat the fluid particles into the density volume ------------------------------
  buildVolume(sim) {
    const { pos, alive, nbSmooth, N } = sim;
    const s = sim.s, r = 1.75 * s, r2 = r * r;
    // Laplacian-smoothed particle centres (Yu & Turk 2013): removes the "bag of marbles" surface
    const SP = this.smoothPos && this.smoothPos.length === N * 3 ? this.smoothPos : (this.smoothPos = new Float32Array(N * 3));
    {
      const acc = this.smAcc && this.smAcc.length === N * 4 ? this.smAcc : (this.smAcc = new Float32Array(N * 4));
      acc.fill(0);
      for (let i = 0; i < N; i++) { acc[i * 4] = pos[i * 3]; acc[i * 4 + 1] = pos[i * 3 + 1]; acc[i * 4 + 2] = pos[i * 3 + 2]; acc[i * 4 + 3] = 1; }
      const { pa, pb, pq, npairs } = sim;
      for (let k = 0; k < npairs; k++) {
        const w0 = 1 - pq[k]; if (w0 <= 0) continue; const w = w0 * w0 * w0;
        const a = pa[k], b = pb[k];
        acc[a * 4] += w * pos[b * 3]; acc[a * 4 + 1] += w * pos[b * 3 + 1]; acc[a * 4 + 2] += w * pos[b * 3 + 2]; acc[a * 4 + 3] += w;
        acc[b * 4] += w * pos[a * 3]; acc[b * 4 + 1] += w * pos[a * 3 + 1]; acc[b * 4 + 2] += w * pos[a * 3 + 2]; acc[b * 4 + 3] += w;
      }
      const lam = 0.85;
      for (let i = 0; i < N; i++) {
        const iw = 1 / acc[i * 4 + 3];
        for (let c = 0; c < 3; c++) SP[i * 3 + c] = (1 - lam) * pos[i * 3 + c] + lam * acc[i * 4 + c] * iw;
      }
    }
    // anisotropic kernels (Yu & Turk 2013): stretch each particle's kernel along the local sheet,
    // squash it along the surface normal -> flat, glassy surfaces and thin sheets instead of blobs
    const AN = this.aniso && this.aniso.length === N * 9 ? this.aniso : (this.aniso = new Float32Array(N * 9));
    const AF = this.anisoF && this.anisoF.length === N ? this.anisoF : (this.anisoF = new Float32Array(N));
    {
      const cov = this.covAcc && this.covAcc.length === N * 10 ? this.covAcc : (this.covAcc = new Float32Array(N * 10));
      cov.fill(0);
      const { pa, pb, pq, npairs } = sim;
      const cnt = this.covCnt && this.covCnt.length === N ? this.covCnt : (this.covCnt = new Uint16Array(N));
      cnt.fill(0);
      for (let k = 0; k < npairs; k++) {
        const q = pq[k]; if (q >= 1) continue;
        const w = 1 - q * q * q;
        const a = pa[k], b = pb[k];
        const dx = pos[b * 3] - pos[a * 3], dy = pos[b * 3 + 1] - pos[a * 3 + 1], dz = pos[b * 3 + 2] - pos[a * 3 + 2];
        // covariance about the particle itself (weighted second moment); symmetric for both ends
        for (let e = 0; e < 2; e++) {
          const i = e ? b : a, sg = e ? -1 : 1, o = i * 10;
          cov[o] += w * dx * dx; cov[o + 1] += w * dx * dy; cov[o + 2] += w * dx * dz;
          cov[o + 3] += w * dy * dy; cov[o + 4] += w * dy * dz; cov[o + 5] += w * dz * dz; cov[o + 6] += w;
          cov[o + 7] += sg * w * dx; cov[o + 8] += sg * w * dy; cov[o + 9] += sg * w * dz;
          cnt[i]++;
        }
      }
      const A = new Float64Array(9), V = new Float64Array(9);
      for (let i = 0; i < N; i++) {
        const o = i * 10, m = i * 9;
        const c = cnt[i];
        // blend toward isotropic for sparse neighbourhoods (drops, thin ligaments)
        const tAn = this.noAniso ? 0 : Math.max(0, Math.min(1, (c - 6) / 10));
        if (tAn <= 0 || cov[o + 6] <= 0) { AN.fill(0, m, m + 9); AN[m] = AN[m + 4] = AN[m + 8] = 1; AF[i] = 1; continue; }
        const iw = 1 / cov[o + 6];
        const mx = cov[o + 7] * iw, my = cov[o + 8] * iw, mz = cov[o + 9] * iw;   // central covariance
        A[0] = cov[o] * iw - mx * mx; A[1] = cov[o + 1] * iw - mx * my; A[2] = cov[o + 2] * iw - mx * mz;
        A[3] = A[1]; A[4] = cov[o + 3] * iw - my * my; A[5] = cov[o + 4] * iw - my * mz;
        A[6] = A[2]; A[7] = A[5]; A[8] = cov[o + 5] * iw - mz * mz;
        V.fill(0); V[0] = V[4] = V[8] = 1;
        // cyclic Jacobi eigen-decomposition of the 3x3 symmetric matrix
        for (let sweep = 0; sweep < 6; sweep++) {
          for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
            const apq = A[p * 3 + q];
            if (Math.abs(apq) < 1e-14) continue;
            const app = A[p * 3 + p], aqq = A[q * 3 + q];
            const th = (aqq - app) / (2 * apq);
            const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
            const cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
            for (let k = 0; k < 3; k++) {
              const akp = A[k * 3 + p], akq = A[k * 3 + q];
              A[k * 3 + p] = cs * akp - sn * akq; A[k * 3 + q] = sn * akp + cs * akq;
            }
            for (let k = 0; k < 3; k++) {
              const apk = A[p * 3 + k], aqk = A[q * 3 + k];
              A[p * 3 + k] = cs * apk - sn * aqk; A[q * 3 + k] = sn * apk + cs * aqk;
            }
            for (let k = 0; k < 3; k++) {
              const vkp = V[k * 3 + p], vkq = V[k * 3 + q];
              V[k * 3 + p] = cs * vkp - sn * vkq; V[k * 3 + q] = sn * vkp + cs * vkq;
            }
          }
        }
        let l0 = Math.max(A[0], 1e-12), l1 = Math.max(A[4], 1e-12), l2 = Math.max(A[8], 1e-12);
        const lmax = Math.max(l0, l1, l2);
        // clamp anisotropy (Yu-Turk k_r = 4 on sigma)
        const lmin = lmax / 16;
        l0 = Math.max(l0, lmin); l1 = Math.max(l1, lmin); l2 = Math.max(l2, lmin);
        let f0 = Math.sqrt(l0), f1 = Math.sqrt(l1), f2 = Math.sqrt(l2);
        const g = Math.cbrt(f0 * f1 * f2);
        f0 /= g; f1 /= g; f2 /= g;                     // volume-preserving stretch factors
        f0 = 1 + (f0 - 1) * tAn; f1 = 1 + (f1 - 1) * tAn; f2 = 1 + (f2 - 1) * tAn;
        // inverse transform G = R diag(1/f) R^T (rows of V^T scaled)
        const fi = [1 / f0, 1 / f1, 1 / f2];
        for (let r0 = 0; r0 < 3; r0++)
          for (let c0 = 0; c0 < 3; c0++) {
            let sum = 0;
            for (let k = 0; k < 3; k++) sum += V[r0 * 3 + k] * fi[k] * V[c0 * 3 + k];
            AN[m + r0 * 3 + c0] = sum;
          }
        AF[i] = Math.max(f0, f1, f2);
      }
    }
    // robust bounds of the coherent water (ignore stray clumps)
    const xs = [], ys = [], zs = [];
    for (let i = 0; i < N; i++) if (alive[i] && nbSmooth[i] >= 7) { xs.push(pos[i * 3]); ys.push(pos[i * 3 + 1]); zs.push(pos[i * 3 + 2]); }
    const sprites = this.looseParticles || (this.looseParticles = []);
    sprites.length = 0;
    if (xs.length < 8) {
      this.grid = { has: false };
      for (let i = 0; i < N; i++) if (alive[i]) sprites.push(i);
      return;
    }
    const pct = (a, f) => { a.sort((u, v) => u - v); return a[Math.min(a.length - 1, Math.max(0, Math.floor(f * (a.length - 1))))]; };
    // full extent of the coherent water when it fits the texture at a sane resolution;
    // only fall back to trimming outliers when a stray clump would blow up the voxel size
    const maxExt = (TEX_MAX - 12) * 1.0 * s;
    let lo = [pct(xs, 0), pct(ys, 0), pct(zs, 0)], hi = [pct(xs, 1), pct(ys, 1), pct(zs, 1)];
    for (const f of [0.002, 0.006, 0.015]) {
      if (Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) <= maxExt) break;
      lo = [pct(xs, f), pct(ys, f), pct(zs, f)]; hi = [pct(xs, 1 - f), pct(ys, 1 - f), pct(zs, 1 - f)];
    }
    const center = [xs[xs.length >> 1], ys[ys.length >> 1], zs[zs.length >> 1]];
    for (let a = 0; a < 3; a++) { lo[a] -= 2.5 * s; hi[a] += 2.5 * s; }
    const ext = [hi[0] - lo[0] + 2 * r, hi[1] - lo[1] + 2 * r, hi[2] - lo[2] + 2 * r];
    let vox = Math.max(0.5 * s, Math.cbrt((ext[0] * ext[1] * ext[2]) / 420000));
    vox = Math.max(vox, Math.max(ext[0], ext[1], ext[2]) / (TEX_MAX - 3));
    const dims = ext.map((e) => Math.min(TEX_MAX, Math.ceil(e / vox) + 3));
    const org = [lo[0] - r - vox, lo[1] - r - vox, lo[2] - r - vox];
    const nx = dims[0], ny = dims[1], nz = dims[2];
    const D = this.densData;
    D.fill(0, 0, nx * ny * nz);
    const inv = 1 / vox;
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      const px = SP[i * 3], py = SP[i * 3 + 1], pz = SP[i * 3 + 2];
      if (px < lo[0] || px > hi[0] || py < lo[1] || py > hi[1] || pz < lo[2] || pz > hi[2]) { sprites.push(i); continue; }
      const fx = (px - org[0]) * inv - 0.5, fy = (py - org[1]) * inv - 0.5, fz = (pz - org[2]) * inv - 0.5;
      const m = i * 9;
      const g00 = AN[m], g01 = AN[m + 1], g02 = AN[m + 2], g11 = AN[m + 4], g12 = AN[m + 5], g22 = AN[m + 8];
      // axis-aligned half-extents of the ellipsoid: r*sqrt(diag(S^2)), S = G^-1 (via adjugate)
      const c00 = g11 * g22 - g12 * g12, c01 = g02 * g12 - g01 * g22, c02 = g01 * g12 - g02 * g11;
      const c11 = g00 * g22 - g02 * g02, c12 = g01 * g02 - g00 * g12, c22 = g00 * g11 - g01 * g01;
      const det = g00 * c00 + g01 * c01 + g02 * c02, idt = 1 / det;
      const ex = r * inv * Math.sqrt((c00 * c00 + c01 * c01 + c02 * c02)) * Math.abs(idt);
      const ey = r * inv * Math.sqrt((c01 * c01 + c11 * c11 + c12 * c12)) * Math.abs(idt);
      const ez = r * inv * Math.sqrt((c02 * c02 + c12 * c12 + c22 * c22)) * Math.abs(idt);
      const x0 = Math.max(0, Math.ceil(fx - ex)), x1 = Math.min(nx - 1, Math.floor(fx + ex));
      const y0 = Math.max(0, Math.ceil(fy - ey)), y1 = Math.min(ny - 1, Math.floor(fy + ey));
      const z0 = Math.max(0, Math.ceil(fz - ez)), z1 = Math.min(nz - 1, Math.floor(fz + ez));
      const qa = (g00 * g00 + g01 * g01 + g02 * g02) * vox * vox;
      for (let z = z0; z <= z1; z++) {
        const dz = (z - fz) * vox;
        for (let y = y0; y <= y1; y++) {
          const dy = (y - fy) * vox;
          const uy = g01 * dy + g02 * dz, vy = g11 * dy + g12 * dz, wy = g12 * dy + g22 * dz;
          // solve the quadratic in x for the exact span of the row inside the ellipsoid
          const qb = 2 * (g00 * uy + g01 * vy + g02 * wy) * vox, qc = uy * uy + vy * vy + wy * wy - r2;
          const disc = qb * qb - 4 * qa * qc;
          if (disc <= 0) continue;
          const sq = Math.sqrt(disc), t0 = (-qb - sq) / (2 * qa), t1 = (-qb + sq) / (2 * qa);
          const xa = Math.max(x0, Math.ceil(fx + t0)), xb = Math.min(x1, Math.floor(fx + t1));
          const row = (z * ny + y) * nx;
          for (let x = xa; x <= xb; x++) {
            const dx = (x - fx) * vox;
            const u = g00 * dx + uy, v = g01 * dx + vy, w_ = g02 * dx + wy;
            const d2 = u * u + v * v + w_ * w_;
            if (d2 < r2) { const w = 1 - d2 / r2; D[row + x] += w * w * w; }
          }
        }
      }
    }
    // separable [1 2 1] smoothing: removes particle-lattice bumps from the surface normals
    {
      const n = nx * ny * nz;
      const T = this.densTmp && this.densTmp.length >= n ? this.densTmp : (this.densTmp = new Float32Array(D.length));
      const pass = (src, dst, stride, len, idxOf) => {
        for (let i = 0; i < n; i++) {
          const c = idxOf(i);
          const a = c > 0 ? src[i - stride] : 0, b = c < len - 1 ? src[i + stride] : 0;
          dst[i] = 0.25 * a + 0.5 * src[i] + 0.25 * b;
        }
      };
      pass(D, T, 1, nx, (i) => i % nx);
      pass(T, D, nx, ny, (i) => ((i / nx) | 0) % ny);
      pass(D, T, nx * ny, nz, (i) => (i / (nx * ny)) | 0);
      D.set(T.subarray(0, n));
    }
    // foam: coarse isotropic splat of the bubble fraction (normalised by the local water amount)
    {
      const fx2 = Math.ceil(nx / 2), fy2 = Math.ceil(ny / 2), fz2 = Math.ceil(nz / 2);
      const Fd = this.foamData, W = this.foamW && this.foamW.length === Fd.length ? this.foamW : (this.foamW = new Float32Array(Fd.length));
      Fd.fill(0, 0, fx2 * fy2 * fz2); W.fill(0, 0, fx2 * fy2 * fz2);
      const fv = 2 * vox, finv = 1 / fv, fr = 2.2 * s, fr2 = fr * fr, fk = fr * finv;
      const foam = sim.foam;
      let any = false;
      for (let i = 0; i < N; i++) {
        if (!alive[i]) continue;
        const f = foam ? foam[i] : 0;
        const px = SP[i * 3], py = SP[i * 3 + 1], pz = SP[i * 3 + 2];
        if (px < lo[0] || px > hi[0] || py < lo[1] || py > hi[1] || pz < lo[2] || pz > hi[2]) continue;
        if (f > 0.01) any = true;
        const gx = (px - org[0]) * finv - 0.5, gy = (py - org[1]) * finv - 0.5, gz = (pz - org[2]) * finv - 0.5;
        const x0 = Math.max(0, Math.ceil(gx - fk)), x1 = Math.min(fx2 - 1, Math.floor(gx + fk));
        const y0 = Math.max(0, Math.ceil(gy - fk)), y1 = Math.min(fy2 - 1, Math.floor(gy + fk));
        const z0 = Math.max(0, Math.ceil(gz - fk)), z1 = Math.min(fz2 - 1, Math.floor(gz + fk));
        for (let z = z0; z <= z1; z++) for (let y = y0; y <= y1; y++) {
          const dyz = ((y - gy) ** 2 + (z - gz) ** 2) * fv * fv; if (dyz >= fr2) continue;
          const row = (z * fy2 + y) * fx2;
          for (let x = x0; x <= x1; x++) {
            const d2 = dyz + (x - gx) * (x - gx) * fv * fv;
            if (d2 < fr2) { const w = 1 - d2 / fr2; Fd[row + x] += w * f; W[row + x] += w; }
          }
        }
      }
      for (let k = 0, n2 = fx2 * fy2 * fz2; k < n2; k++) Fd[k] = W[k] > 0.05 ? Fd[k] / W[k] : 0;
      this.hasFoam = any;
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_3D, this.foamTex);
      gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0, fx2, fy2, fz2, gl.RED, gl.FLOAT, Fd.subarray(0, fx2 * fy2 * fz2));
    }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_3D, this.dens);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0, nx, ny, nz, gl.RED, gl.FLOAT, D.subarray(0, nx * ny * nz));
    this.grid = {
      has: true, vox, org, dims, kr: r, center,
      boxMin: [org[0] + vox, org[1] + vox, org[2] + vox],
      boxMax: [org[0] + (nx - 1) * vox, org[1] + (ny - 1) * vox, org[2] + (nz - 1) * vox],
    };
  }

  uploadWet(sim) {
    if (!sim.wetDirty) return;
    const gl = this.gl;
    // surface tension levels a thin film: blur the deposited water before shading it
    const n = WET_RES, A = this.wetA || (this.wetA = new Float32Array(n * n)), B = this.wetB || (this.wetB = new Float32Array(n * n));
    const k = [1, 4, 6, 4, 1];
    const src = sim.wet;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      let a = 0, ws = 0;
      for (let t = -2; t <= 2; t++) { const x = i + t; if (x < 0 || x >= n) continue; a += src[j * n + x] * k[t + 2]; ws += k[t + 2]; }
      A[j * n + i] = a / ws;
    }
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      let a = 0, ws = 0;
      for (let t = -2; t <= 2; t++) { const y = j + t; if (y < 0 || y >= n) continue; a += A[y * n + i] * k[t + 2]; ws += k[t + 2]; }
      B[j * n + i] = a / ws;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.wetTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, WET_RES, WET_RES, gl.RED, gl.FLOAT, B);
    sim.wetDirty = false;
  }

  buildSprites(sim) {
    const S = this.spriteData;
    let n = 0;
    const put = (x, y, z, r, type, a, age) => {
      if (n >= this.spriteCap) return;
      const o = n * 8;
      S[o] = x; S[o + 1] = y; S[o + 2] = z; S[o + 3] = r; S[o + 4] = type; S[o + 5] = a; S[o + 6] = age; S[o + 7] = 0;
      n++;
    };
    // loose fluid parcels (outside the volume)
    const rr = 0.62 * sim.s;
    for (const i of this.looseParticles || []) put(sim.pos[i * 3], sim.pos[i * 3 + 1], sim.pos[i * 3 + 2], rr, 0, 1, 0);
    const f = sim.fx;
    let nd = n;
    for (let i = 0; i < f.alive.length; i++) {
      if (!f.alive[i] || f.type[i] !== 0) continue;
      put(f.pos[i * 3], f.pos[i * 3 + 1], f.pos[i * 3 + 2], f.size[i], 0, 1, f.age[i]);
    }
    this.nDrops = n;
    for (let i = 0; i < f.alive.length; i++) {
      if (!f.alive[i] || f.type[i] !== 1) continue;
      const age = f.age[i], life = f.life[i];
      const fade = Math.min(1, age / 0.004) * Math.pow(Math.max(0, 1 - age / life), 1.6);
      put(f.pos[i * 3], f.pos[i * 3 + 1], f.pos[i * 3 + 2], f.size[i], 1, 0.13 * fade * Math.min(1.5, sim.p.mist || 0), age);
    }
    this.nSprites = n;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, S.subarray(0, n * 8));
  }

  setCommon(p, V) {
    const gl = this.gl;
    const U = (n) => this.u(p, n);
    gl.uniform3fv(U('uCamPos'), V.cam.pos); gl.uniform3fv(U('uCamRight'), V.cam.right);
    gl.uniform3fv(U('uCamUp'), V.cam.up); gl.uniform3fv(U('uCamFwd'), V.cam.fwd);
    gl.uniform1f(U('uTanHalf'), V.cam.tanHalf); gl.uniform1f(U('uAspect'), V.cam.aspect);
    gl.uniformMatrix4fv(U('uViewProj'), false, V.cam.viewProj);
    gl.uniform3fv(U('uL'), V.light.L); gl.uniform1f(U('uLInt'), V.light.I);
    gl.uniform3fv(U('uFillDir'), V.light.fill);
    gl.uniform3fv(U('uKeyC'), V.light.keyC); gl.uniform3fv(U('uKeyU'), V.light.keyU); gl.uniform3fv(U('uKeyV'), V.light.keyV);
    gl.uniform2fv(U('uKeyHalf'), V.light.keyHalf);
    gl.uniform3fv(U('uFillC'), V.light.fillC); gl.uniform3fv(U('uFillU'), V.light.fillU); gl.uniform3fv(U('uFillV'), V.light.fillV);
    gl.uniform3fv(U('uFocus'), V.focus);
    gl.uniform1f(U('uWetExt'), WET_EXT);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.wetTex); gl.uniform1i(U('uWet'), 1);
  }

  render(sim, V, scale) {
    const gl = this.gl, c = this.canvas;
    const w = Math.max(64, Math.round(c.width * scale)), h = Math.max(64, Math.round(c.height * scale));
    this.ensureTargets(w, h);
    this.uploadWet(sim);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, w, h);
    // ---- pass 1: ray trace
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.ALWAYS); gl.depthMask(true);
    gl.disable(gl.BLEND);
    const p = this.progTrace;
    gl.useProgram(p);
    this.setCommon(p, V);
    const U = (n) => this.u(p, n);
    const G = this.grid;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, this.dens); gl.uniform1i(U('uDens'), 0);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_3D, this.foamTex); gl.uniform1i(U('uFoam'), 5); gl.uniform1f(U('uFoamOn'), this.hasFoam ? 1 : 0);
    { const G = this.grid, m = sim.mem, age = m && m.tIn >= 0 ? sim.t - m.tIn : -1;
      gl.uniform3fv(U('uBlobC'), G && G.center ? G.center : sim.C);
      gl.uniform1f(U('uRip'), age > 0.002 ? 0.4 * Math.exp(-(age - 0.002) / 0.12) + 0.22 : 0); } gl.activeTexture(gl.TEXTURE0);
    gl.uniform1f(U('uHasWater'), G.has ? 1 : 0);
    if (G.has) {
      gl.uniform3fv(U('uGridMin'), G.org);
      gl.uniform3f(U('uTexWorld'), TEX_MAX * G.vox, TEX_MAX * G.vox, TEX_MAX * G.vox);
      gl.uniform3fv(U('uBoxMin'), G.boxMin); gl.uniform3fv(U('uBoxMax'), G.boxMax);
      gl.uniform1f(U('uVoxel'), G.vox); gl.uniform1f(U('uKR'), G.kr);
    }
    gl.uniform1f(U('uIso'), 0.56); gl.uniform1i(U('uDbg'), this.dbg|0); gl.uniform1f(U('uPixA'), 2 * V.cam.tanHalf / h);
    gl.uniform1f(U('uAerate'), V.aerate);
    const b = sim.bullet;
    gl.uniform3f(U('uBulletTip'), b.x, b.y, b.z);
    gl.uniform1f(U('uBulletR'), b.r); gl.uniform1f(U('uBulletLen'), b.len);
    gl.uniform1f(U('uBulletOn'), b.x - b.len < 4 ? 1 : 0);
    const m = sim.mem;
    gl.uniform3fv(U('uMemC'), sim.C); gl.uniform1f(U('uMemR'), sim.R); gl.uniform1f(U('uMemRy'), sim.ry);
    gl.uniform1f(U('uMemOn'), m.gone ? 0 : 1);
    gl.uniform3fv(U('uE1'), sim.e1); gl.uniform3fv(U('uE2'), sim.e2);
    gl.uniform1f(U('uTh1'), m.tIn >= 0 ? m.th1 : -1); gl.uniform1f(U('uTh2'), m.tOut >= 0 ? m.th2 : -1);
    gl.uniform1f(U('uAlpha'), m.alpha);
    gl.uniform3fv(U('uLatex'), V.latex);
    const rem = m.tIn >= 0 ? Math.min(1, Math.max(0, (sim.t - m.tIn - 0.002) / 0.006)) : 0;
    gl.uniform1f(U('uRemnant'), rem);
    gl.uniform1f(U('uStringTop'), sim.C[1] + 3.0);
    gl.bindVertexArray(this.vaoFull);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // ---- pass 2: spray + mist
    if (this.nSprites > 0 && this.dbg !== 5) {
      const ps = this.progSprite;
      gl.useProgram(ps);
      this.setCommon(ps, V);
      gl.uniform1f(this.u(ps, 'uPixelWorld'), (2 * V.cam.tanHalf) / h);
      gl.depthFunc(gl.LESS);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(this.vaoSprite);
      gl.depthMask(true);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.nDrops);
      gl.depthMask(false);
      if (this.nSprites > this.nDrops) {
        // mist instances live after the drops in the buffer
        const off = this.nDrops * 32;
        const la = gl.getAttribLocation(ps, 'aPS'), li = gl.getAttribLocation(ps, 'aInfo');
        gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
        gl.vertexAttribPointer(la, 4, gl.FLOAT, false, 32, off);
        gl.vertexAttribPointer(li, 4, gl.FLOAT, false, 32, off + 16);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.nSprites - this.nDrops);
        gl.vertexAttribPointer(la, 4, gl.FLOAT, false, 32, 0);
        gl.vertexAttribPointer(li, 4, gl.FLOAT, false, 32, 16);
      }
      gl.depthMask(true);
      gl.disable(gl.BLEND);
    }
    // ---- pass 3: tone map to screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, c.width, c.height);
    gl.disable(gl.DEPTH_TEST);
    const pc = this.progComp;
    gl.useProgram(pc);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.hdr); gl.generateMipmap(gl.TEXTURE_2D);
    gl.uniform1i(this.u(pc, 'uHdr'), 0);
    gl.uniform1f(this.u(pc, 'uExposure'), V.exposure);
    gl.uniform2f(this.u(pc, 'uRes'), w, h);
    gl.uniform1f(this.u(pc, 'uSeed'), (this.frame++ % 64) * 0.37);
    gl.bindVertexArray(this.vaoFull);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }
}
