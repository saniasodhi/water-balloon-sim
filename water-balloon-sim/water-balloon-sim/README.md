# Water balloon ballistics — ray-traced simulation

A bullet pierces a hanging water balloon: the latex tears away, the water bursts,
sprays and mists at the holes, falls and splashes into a puddle — simulated and
ray traced in real time in the browser. **No libraries** (no Three.js, no frameworks):
one self-contained HTML file, plain JavaScript and WebGL2.

Open `dist/water-balloon.html` in any modern browser (Chrome, Edge, Firefox, Safari 15+).

## Features

- **Freeze and scrub** to any moment (−1 ms … 1 s after impact) and **orbit / pan / zoom** from any angle.
- **Sliders**: bullet speed & calibre, impact height, drag, balloon diameter and hanging height,
  rubber snap speed, gravity, cavity collapse, spray/mist amount, particle count, light direction,
  intensity, exposure, render scale, slow-motion factor. Balloon colour swatches, camera presets.
- Keyboard: `Space` play/pause, `→` step, `R` restart.

## How it works

**Simulation (`src/sim.js`)** — SI units, adaptive time step.
- Water: particle fluid (double-density relaxation, Clavet et al.) with XSPH viscosity, limited
  tension for cohesion, rest-density calibration and a gravity pre-roll so the balloon starts at rest.
  Stiffness and damping are normalised to a reference particle spacing, so the flow is (approximately)
  independent of the particle count.
- Bullet: drag-based energy deposition per unit length (F = ½ρC<sub>d</sub>Av²), radial cavity flow,
  back-splash at entry, jet at exit, bullet deceleration, momentum transfer to the water.
- Temporary cavity: atmospheric pressure decelerates the outward flow and collapses the tunnel
  (Rayleigh-type), with closure detection.
- Latex skin: analytic ellipsoid that holds the water until torn; tears propagate from the holes at
  the rubber snap speed with ragged edges; the torn skirt stays on the knot.
- Break-up: fast particles atomise into drops and mist; drops have quadratic air drag; the floor impact
  redirects momentum into a spreading sheet, lifts a low crown and throws droplets; water that comes to
  rest turns into a thin film (wetness map).
- Entrained air: a per-particle bubble fraction, created along the bullet path and at the floor impact.

**Rendering (`src/render.js`)**
- Water surface: particles are Laplacian-smoothed and splatted with **anisotropic kernels**
  (Yu & Turk 2013) into a 3D density texture; the fragment shader ray-marches the iso-surface.
- Ray tracing: reflection + refraction with total internal reflection, Beer–Lambert absorption,
  bubble scattering, caustic-lit floor under the water, soft shadows through water, water film with
  meniscus and ripples, refraction through the intact balloon (the water-filled sphere acts as a lens).
- Latex membrane shading, SDF bullet (copper jacket), knot & neck, anti-aliased string.
- Spray and mist as depth-tested sprites; ACES tone mapping, mip-chain bloom, vignette, grain.

## Build

```bash
python3 build.py          # inlines src/*.js into src/app.html -> dist/water-balloon.html
```

## Tools (optional, for development)

```bash
node tools/bench.js 6000              # simulation cost per step for N particles
node tools/resolution_check.js 6000   # blob spread vs. particle count
pip install playwright && playwright install chromium
python3 tools/shoot.py '[["home",-0.2],["barrel",5],["floor",470]]' shots   # headless screenshots
python3 tools/contact_sheet.py shots
```

## Known limitations

- Detail is limited by the particle spacing (≈7 mm at the default 6 000 particles); fine ripples are
  a shading effect, not simulated. A GPU solver would be needed for much finer resolution.
- The physics is a plausible engineering model, not a validated CFD solver.
