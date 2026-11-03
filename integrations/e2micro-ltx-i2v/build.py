import json, textwrap, pathlib
root=pathlib.Path("/home/ubuntu/e2mpt-death/ltx_upgrade")
cells=[]
def add(code):
    cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":textwrap.dedent(code).strip()+"\n"})
add("""
import sys,subprocess,os,time
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_XET']='1'
print('INSTALL_BEGIN',flush=True)
subprocess.run([sys.executable,'-m','pip','install','-q',
 'diffusers>=0.35.0,<0.42.0','transformers>=4.47.0,<4.52.0',
 'accelerate','sentencepiece','imageio[ffmpeg]','safetensors','pillow','numpy'],
 check=True)
import torch, diffusers
print('INSTALL_COMPLETE','torch',torch.__version__,'diffusers',diffusers.__version__,flush=True)
print('GPU',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT_AVAILABLE',flush=True)
assert torch.cuda.is_available(),'Kaggle accelerator required'
""")
add("""
from pathlib import Path
from PIL import Image, ImageOps
import json, numpy as np, torch, gc, traceback, os, subprocess, time, math
from diffusers import LTXConditionPipeline
from diffusers.utils import export_to_video
ASSETS = list(Path('/kaggle/input').rglob('speech_00.wav'))
assert ASSETS,'Missing private VoxCPM2 assets dataset'
SRC=ASSETS[0].parent
IMAGES=[SRC / ('scene_%02d.jpg'%i) for i in range(8)]
assert all(p.exists() for p in IMAGES),[(p.name,p.exists()) for p in IMAGES]
SCENES=json.loads((SRC/'script.json').read_text())
assert len(SCENES)==8
OUT=Path('/kaggle/working/ltx_death_documentary');OUT.mkdir(exist_ok=True)
print('INPUTS_OK',SRC,'narration', (SRC/'narration.wav').stat().st_size,flush=True)
MOTION=[
 'Cinematic night sky with subtle drifting clouds and twinkling stars, distant silhouetted figure stays still, gentle slow dolly forward, stable documentary motion, no text.',
 'Cinematic close view of a symbolic anatomical heart model, soft red light pulses gently around the model, slow cinematic camera orbit, stable composition, educational documentary, no text.',
 'Scientific visualization of interconnected neurons, glowing electric blue pulses move along pathways then slowly fade, subtle light transitions, stable and precise, no text.',
 'Empty hospital corridor, camera slowly moves forward, soft morning light changes gently, subtle airborne dust motes, no people, cinematic documentary, no text.',
 'Nature scene with fallen autumn leaves, delicate mist slowly drifts and leaves flutter in a light breeze, calm photographic realism, no text.',
 'Beautiful deep indigo celestial stars, slow drifting distant particles and very subtle cinematic zoom toward a warm distant light, no text.',
 'Abstract transparent human profile silhouette with gentle neural shimmer and subtle floating particles, static composition with organic luminous movement, no text.',
 'Tranquil ocean sunrise, golden clouds slowly shift, gentle wave movement and expanding warm sunrise, peaceful cinematic finale, no text.'
]
""")
add("""
# Prefer the 2B 0.9.8 distilled checkpoint. Never silently label a fallback "distilled".
MODEL_USED='Lightricks/ltxv-2b-0.9.8-distilled.safetensors'
MODEL_ERROR=None
pipe=None
torch.cuda.empty_cache()
try:
    from diffusers import LTXVideoTransformer3DModel
    print('TRY_DISTILLED_2B',flush=True)
    transformer=LTXVideoTransformer3DModel.from_single_file(
      'https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.8-distilled.safetensors',
      torch_dtype=torch.float16
    )
    pipe=LTXConditionPipeline.from_pretrained('Lightricks/LTX-Video-0.9.5',
      transformer=transformer,torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe.vae,'enable_tiling'):pipe.vae.enable_tiling()
    print('DISTILLED_2B_LOADED',flush=True)
except Exception as e:
    MODEL_ERROR=str(e)[:500]
    print('DISTILLED_2B_LOAD_FAILED',MODEL_ERROR,flush=True)
    for key in ['transformer','pipe']:
        if key in globals():globals()[key]=None
    gc.collect();torch.cuda.empty_cache()
    print('FALLBACK_STABLE_2B_0.9.5',flush=True)
    pipe=LTXConditionPipeline.from_pretrained('Lightricks/LTX-Video-0.9.5',
      torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe.vae,'enable_tiling'):pipe.vae.enable_tiling()
    MODEL_USED='Lightricks/LTX-Video-0.9.5 (2B non-distilled fallback)'
print('MODEL_USED',MODEL_USED,flush=True)
""")
add("""
from diffusers.utils import export_to_video
results=[]
FPS=24
NEG='flickering objects, glitches, artificial transformations, deformation, low quality, jittery, unreadable captions, watermark'
for i,src_image in enumerate(IMAGES):
    target=OUT / ('motion_%02d.mp4'%i)
    if target.exists() and target.stat().st_size>20000:
        results.append({'scene':i,'status':'ai','path':str(target),'bytes':target.stat().st_size})
        print('REUSED_AI_CLIP',i,flush=True)
        continue
    image=ImageOps.fit(Image.open(src_image).convert('RGB'),(512,512),method=Image.Resampling.LANCZOS)
    image.save(OUT/('input_%02d.jpg'%i))
    prompt=MOTION[i]
    print('I2V_START',i+1,'/8',MODEL_USED,flush=True)
    try:
        import inspect
        kwargs=dict(image=image,prompt=prompt,negative_prompt=NEG,width=512,height=512,
           num_frames=49,num_inference_steps=8 if 'distill' in MODEL_USED.lower() and 'non-distilled' not in MODEL_USED.lower() else 28,
           guidance_scale=1.0 if 'distill' in MODEL_USED.lower() and 'non-distilled' not in MODEL_USED.lower() else 4.5,
           decode_timestep=0.05,decode_noise_scale=0.025,
           generator=torch.Generator('cpu').manual_seed(20261103+i))
        frames=pipe(**kwargs).frames[0]
        export_to_video(frames,str(target),fps=FPS)
        assert target.exists() and target.stat().st_size>10000
        results.append({'scene':i,'status':'ai','path':str(target),'bytes':target.stat().st_size})
        print('I2V_SUCCESS',i+1,target.stat().st_size,flush=True)
        del frames;gc.collect();torch.cuda.empty_cache()
    except Exception as exc:
        print('I2V_SCENE_FAILED',i+1,repr(exc)[:550],flush=True)
        results.append({'scene':i,'status':'fallback','path':None,'error':str(exc)[:350]})
        gc.collect();torch.cuda.empty_cache()
    (OUT/'generation_report.json').write_text(json.dumps({'model':MODEL_USED,'model_load_error':MODEL_ERROR,'scenes':results},indent=2))
print('I2V_RESULTS',json.dumps({'ai':sum(x['status']=='ai' for x in results),'fallback':sum(x['status']!='ai' for x in results)}),flush=True)
""")
add("""
# MoneyPrinterTurbo-style final assembly: clips -> scene subtitles -> soundtrack -> MP4.
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from pathlib import Path
import subprocess, wave, json, math, numpy as np, shutil
W,H,FPS=720,1280,24
FINAL=OUT/'what_happens_when_we_die_AI_animated.mp4'
def ff(*args):
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*list(map(str,args))],check=True)
def texwrap(draw,text,font,max_width):
    words=text.split();lines=[];line=''
    for word in words:
        c=(line+' '+word).strip()
        if line and draw.textbbox((0,0),c,font=font)[2]>max_width:
            lines.append(line);line=word
        else:line=c
    if line:lines.append(line)
    return lines
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',26)
fonttitle=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',34)
fontsmall=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
for i,row in enumerate(SCENES):
    dest=OUT / ('overlay_%02d.png'%i)
    layer=Image.new('RGBA',(W,H),(0,0,0,0))
    d=ImageDraw.Draw(layer,'RGBA')
    d.rectangle((0,0,W,165),fill=(4,10,26,205))
    d.rectangle((0,H-360,W,H),fill=(2,7,20,205))
    d.text((31,35),'WHAT HAPPENS WHEN WE DIE?',font=fonttitle,fill=(247,241,224,255))
    d.text((33,109),'SCIENCE DOCUMENTARY  •  %02d/08'%(i+1),font=fontsmall,fill=(190,210,221,255))
    lines=texwrap(d,row['text'],font,W-58)
    h=38*len(lines)
    sy=H-178-h//2
    for ix,line in enumerate(lines):
        box=d.textbbox((0,0),line,font=font);dx=(W-(box[2]-box[0]))//2
        d.text((dx,sy+ix*38),line,font=font,fill=(255,255,249,255),stroke_width=1,stroke_fill=(5,10,13,240))
    d.text((250,H-45),'E2MICRO  •  VOXCPM2',font=fontsmall,fill=(174,196,204,255))
    layer.save(dest)
segments=[]
for i,row in enumerate(SCENES):
    duration=float(row['duration'])
    visual=next((p for p in results if p['scene']==i),None)
    out=OUT / ('segment_%02d.mp4'%i)
    overlay=OUT/('overlay_%02d.png'%i)
    img=IMAGES[i]
    if visual and visual['status']=='ai':
        source=visual['path']
        # Loop the real AI motion softly to fill the VoxCPM2 narration duration.
        ff('-stream_loop','-1','-i',source,'-loop','1','-i',overlay,
           '-filter_complex',f'[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},format=yuv420p[v];[v][1:v]overlay=0:0:shortest=1[out]',
           '-map','[out]','-t',str(duration),'-an','-r',str(FPS),'-c:v','libx264','-preset','veryfast','-crf','23',out)
    else:
        count=math.ceil(duration*FPS)
        ff('-loop','1','-i',img,'-loop','1','-i',overlay,
           '-filter_complex',f'[0:v]scale=768:768,zoompan=z=\\'min(zoom+0.0002,1.10)\\':x=\\'(iw-iw/zoom)/2\\':y=\\'(ih-ih/zoom)/2\\':d={count}:s={W}x{H}:fps={FPS},format=yuv420p[v];[v][1:v]overlay=0:0:shortest=1[out]',
           '-map','[out]','-frames:v',str(count),'-an','-r',str(FPS),'-c:v','libx264','-preset','veryfast','-crf','23',out)
    assert out.stat().st_size>15000
    segments.append(out)
    print('ASSEMBLED_SCENE',i+1,visual['status'] if visual else 'missing',out.stat().st_size,flush=True)
listing=OUT/'segments.txt'
listing.write_text(''.join("file '"+str(p)+"'\\n" for p in segments))
ff('-f','concat','-safe','0','-i',listing,'-c','copy',OUT/'silent.mp4')
# Existing VoxCPM2 narration and restrained original music.
import soundfile as sf
speech,SR=sf.read(SRC/'narration.wav',dtype='float32')
if speech.ndim>1:speech=speech.mean(axis=1)
t=np.arange(len(speech),dtype=np.float64)/SR
tot=len(speech)/SR
music=np.zeros(len(speech),dtype=np.float64)
for k,notes in enumerate([(146.83,196.00,246.94),(174.61,220.0,261.63),(130.81,164.81,196.0)]):
    lo=k*tot/3;hi=(k+1)*tot/3
    envelope=np.clip((t-lo)/1.2,0,1)*np.clip((hi-t)/1.6,0,1)
    for j,f in enumerate(notes):
        music+=.009/(j+1)*envelope*np.sin(2*np.pi*f*t)
sf.write(OUT/'soundtrack.wav',np.clip(.98*speech+music,-.97,.97).astype(np.float32),SR)
ff('-i',OUT/'silent.mp4','-i',OUT/'soundtrack.wav','-c:v','copy','-c:a','aac','-b:a','144k',
   '-shortest','-movflags','+faststart',FINAL)
res=subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration,size','-of','json',str(FINAL)],text=True)
print('FINAL_SUCCESS',FINAL,'probe',res.strip(),flush=True)
(OUT/'generation_report.json').write_text(json.dumps({'model':MODEL_USED,'model_load_error':MODEL_ERROR,'scenes':results,
    'output':str(FINAL),'probe':json.loads(res)},indent=2))
""")
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
(root/"ltx_i2v_documentary.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=1))
meta={"id":"mgcyaw/e2micro-death-ltx2b-i2v-20261103","title":"E2micro Death Documentary LTX 2B Image to Video 20261103",
"code_file":"ltx_i2v_documentary.ipynb","language":"python","kernel_type":"notebook","is_private":True,
"enable_gpu":True,"enable_tpu":False,"enable_internet":True,
"keywords":["ltx-video","image-to-video","voxcpm2","moneyprinterturbo"],
"dataset_sources":["mgcyaw/e2micro-death-documentary-final-assets-20261103"],
"kernel_sources":[],"competition_sources":[],"model_sources":[]}
(root/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
for i,cell in enumerate(cells):
    compile(cell['source'],f'cell{i+1}','exec')
print('BUILT',len(cells),'cells', (root/"ltx_i2v_documentary.ipynb").stat().st_size)
