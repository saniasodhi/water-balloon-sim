import sys,glob
from PIL import Image
d=sys.argv[1]; fs=sorted(glob.glob(d+'/[0-9]*.png'))
W,H=560,420; cols=2
rows=(len(fs)+1)//2
S=Image.new('RGB',(W*cols,H*rows))
for i,f in enumerate(fs):
    im=Image.open(f).crop((0,0,960,720)).resize((W,H))
    S.paste(im,((i%cols)*W,(i//cols)*H))
S.save(d+'/sheet.jpg',quality=85)
