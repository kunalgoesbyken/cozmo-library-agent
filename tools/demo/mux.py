import json, subprocess, sys
d = sys.argv[1]
t = json.load(open(f"{d}/timing.json"))
v = t["videoStart"]
a = t["taps"][0]["start"] - v
u = t["gumAt"] - v
dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f"{d}/screen.webm"]).decode().strip() or 0)
fc = f"[1:a]adelay={a}|{a}[ag];[2:a]atrim=0:100,adelay={u}|{u},volume=0.8[us];[ag][us]amix=inputs=2:duration=longest:normalize=0,apad[a]"
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", f"{d}/screen.webm", "-i", f"{d}/tap0.webm", "-i", "user_voice.wav",
    "-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
    "-r", "25", "-c:a", "aac", "-b:a", "160k", "-t", str(dur) if dur else "360", f"{d}/demo.mp4"], check=True)
print(a, u, dur)
