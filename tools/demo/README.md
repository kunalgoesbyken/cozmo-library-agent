# Demo recording harness

How the demo video was made without a physical shelf. A real Chromium session runs the real web app against the real server. Only the camera and the user's voice are fakes, and the video says so on screen.

1. Fake camera: an MJPEG pan over the two shelf photos at native resolution (`ground_truth/SOURCE.md`). Chromium plays MJPEG files at 30 fps, so encode at 30 fps or the pan runs at the wrong speed.

```
ffmpeg -loop 1 -framerate 15 -i pelican.jpg -vf "crop=1920:1080:'if(lt(t,4),0,min(1920,(t-4)*30))':430" -t 74 -an seg1.mp4
ffmpeg -loop 1 -framerate 15 -i library.jpg -vf "crop=1920:1080:0:'min(200,t*8)'" -t 34 -an seg2.mp4
printf "file seg1.mp4\nfile seg2.mp4\n" > list.txt
ffmpeg -f concat -i list.txt -c:v mjpeg -q:v 4 -pix_fmt yuvj420p -r 30 camera.mjpeg
```

2. User voice: four lines from Windows text-to-speech (System.Speech, 16 kHz mono), placed at 8, 42, 78 and 91 s in `user_voice.wav`.
3. Record: `npm i playwright`, then `node record.cjs "<tunnel url with ?t=token>" 98 take` (set `CHROMIUM_PATH` if Playwright's own browser is not installed). It starts the sweep, opens the inventory sheet once, ends the sweep if the agent has not, waits for the packet, then scrolls the report. The agent's audio is captured inside the page from the playback AudioContext.
4. Mux: `python mux.py take` joins the screen video, the agent audio and the user voice at their measured offsets into `take/demo.mp4`. This is one continuous take with no cuts.
