#!/usr/bin/env python3
"""E2micro controller: retrieve completed Kaggle animation and deliver to Telegram once."""
from pathlib import Path
import subprocess,json,re,urllib.request,uuid,sys
ROOT=Path(__file__).resolve().parent
KERNEL="mgcyaw/e2micro-death-ltx2b-i2v-20261103"
STATE=ROOT/"telegram_delivery.json"
OUTPUT=ROOT/"results"
TARGET="what_happens_when_we_die_AI_animated.mp4"
CONFIG=Path("/home/ubuntu/e2-telegram-controller/secrets.env")
KAGGLE="/home/ubuntu/.local/bin/kaggle"
def run(*a,timeout=120):
    return subprocess.run([KAGGLE,*a],capture_output=True,text=True,timeout=timeout)
def envfile():
    d={}
    for line in CONFIG.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k,v=line.split("=",1);d[k.strip()]=v.strip().strip('"').strip("'")
    return d
def sendvideo(path):
    creds=envfile()
    boundary="----e2micro-"+uuid.uuid4().hex
    def part(key,val):
        return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{val}\r\n").encode()
    caption=("WHAT HAPPENS WHEN WE DIE? Animated Edition | Cloudflare AI images, "
        "LTX video on Kaggle, VoxCPM2 narration and documentary subtitles.")
    body=part("chat_id",creds["TG_OWNER_ID"])+part("caption",caption)+part("supports_streaming","true")
    body+=(f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"{TARGET}\"\r\nContent-Type: video/mp4\r\n\r\n").encode()
    body+=path.read_bytes()+b"\r\n"+f"--{boundary}--\r\n".encode()
    req=urllib.request.Request("https://api.telegram.org/bot"+creds["TG_BOT_TOKEN"]+"/sendVideo",
        data=body,method="POST",headers={"Content-Type":f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req,timeout=150) as response: result=json.load(response)
    if not result.get("ok"):raise RuntimeError("Telegram did not confirm delivery")
    return result["result"]["message_id"]
def main():
    if STATE.exists():
        info=json.loads(STATE.read_text())
        if info.get("status")=="delivered":
            print("ALREADY_DELIVERED",info["message_id"]);return 0
    st=run("kernels","status",KERNEL)
    mt=re.search(r"KernelWorkerStatus\.([A-Z_]+)",st.stdout+" "+st.stderr)
    status=mt.group(1) if mt else "UNKNOWN"
    print("KAGGLE_STATUS",status,flush=True)
    if status!="COMPLETE":return 2
    OUTPUT.mkdir(exist_ok=True)
    down=run("kernels","output",KERNEL,"-p",str(OUTPUT),"-q",
        "--file-pattern",r".*(AI_animated[.]mp4|generation_report[.]json|[.]log)$",timeout=300)
    print("DOWNLOAD_RC",down.returncode)
    if down.returncode:
        print("DOWNLOAD_ERROR",(down.stderr+down.stdout)[-500:]);return 3
    videos=list(OUTPUT.rglob(TARGET))
    if not videos:
        print("FINAL_MP4_MISSING")
        print("FILES",[p.name for p in OUTPUT.rglob("*") if p.is_file()][:30])
        return 4
    video=videos[0]
    if video.stat().st_size<50_000:
        print("FINAL_MP4_TOO_SMALL",video.stat().st_size);return 5
    reports=list(OUTPUT.rglob("generation_report.json"))
    report=json.loads(reports[0].read_text()) if reports else {}
    statuses=[x.get("status") for x in report.get("scenes",[])]
    if statuses and not any(x=="ai" for x in statuses):
        print("NO_AI_MOTION_SCENES; NO DELIVERY");return 6
    message_id=sendvideo(video)
    result={"status":"delivered","message_id":message_id,"file":str(video),
        "bytes":video.stat().st_size,"ai_scenes":sum(x=="ai" for x in statuses),
        "fallback_scenes":sum(x!="ai" for x in statuses),"model":report.get("model","not reported")}
    STATE.write_text(json.dumps(result,indent=2))
    print("TELEGRAM_DELIVERY_SUCCESS",json.dumps(result,indent=2),flush=True)
    return 0
if __name__=="__main__":sys.exit(main())
