"""Inline src/sim.js and src/render.js into src/app.html -> dist/water-balloon.html (one self-contained file)."""
import os
here = os.path.dirname(os.path.abspath(__file__))
src = lambda f: open(os.path.join(here, 'src', f), encoding='utf-8').read()
html = src('app.html').replace('/*__SIM__*/', src('sim.js')).replace('/*__RENDER__*/', src('render.js'))
os.makedirs(os.path.join(here, 'dist'), exist_ok=True)
out = os.path.join(here, 'dist', 'water-balloon.html')
open(out, 'w', encoding='utf-8').write(html)
print(f'wrote {out} ({len(html) // 1024} KB)')
