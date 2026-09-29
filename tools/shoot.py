"""Headless screenshot harness: python tools/shoot.py '[["home",10],["floor",470]]' outdir [preJS]
Views: home, side, barrel, above, floor. Times in ms relative to impact."""
import sys, json, time, os, pathlib
HTML = pathlib.Path(__file__).resolve().parent.parent.joinpath("dist", "water-balloon.html").as_uri()
from playwright.sync_api import sync_playwright
shots = json.loads(sys.argv[1])
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
pre = sys.argv[3] if len(sys.argv)>3 else ""
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader","--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width":1280,"height":720})
    logs=[]
    pg.on("console", lambda m: logs.append(m.type+": "+m.text))
    pg.on("pageerror", lambda e: logs.append("PAGEERROR: "+str(e)))
    pg.goto(HTML)
    pg.wait_for_timeout(800)
    pg.evaluate("()=>WB.halt()")
    pg.wait_for_timeout(300)
    if pre: pg.evaluate(pre)
    for l in logs: print(l[:3000])
    for i,(view,ms,*extra) in enumerate(shots):
        t0=time.time()
        pg.evaluate("v=>WB.view(v[0], v[1])", [view, extra[0] if extra else None])
        r=pg.evaluate("ms=>WB.shot(ms)", ms)
        fn=f"{out}/{i:02d}_{view}_{ms}.png"
        pg.screenshot(path=fn, timeout=120000)
        print(fn, f"{time.time()-t0:.1f}s gpu {r:.0f}ms")
    for l in logs[-20:]: print(l[:800])
    b.close()
