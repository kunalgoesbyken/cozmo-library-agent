# Demo recording harness

How the demo video was made without a physical shelf. A real Chromium session runs the real web app against the real server and the live Gemini session. Only the camera and the user's voice are fakes, and an on-screen banner says so for the whole take.

1. Fake camera: `python make_camera.py pelican.jpg library.jpg` writes `camera.mjpeg`, 30 fps at 1920x1080 (Chromium plays MJPEG files at 30 fps). The camera pans over the two shelf photos named in `ground_truth/SOURCE.md` at their native resolution. At 30 s it whips too fast with motion blur, then goes back slowly over the blurred stretch, so the take shows the agent catching bad capture and the sweep recovering in the same pass.
2. User voice: five lines from Windows text-to-speech (System.Speech, 16 kHz mono) placed at 8, 35, 58, 82 and 103 s in `user_voice.wav`. Chromium feeds it to the page as the microphone (`--use-file-for-fake-audio-capture`).
3. Record: `npm i playwright`, then `node record.cjs "<tunnel url with ?t=token>" 116 take` (set `CHROMIUM_PATH` if Playwright's own browser is not installed).
   - A second tab captures the app tab live with `getDisplayMedia` (tab video plus tab audio). It records them with one `MediaRecorder` into `take/take.webm`.
   - The microphone is monitored into the tab's audio output, so both voices are in the same live recording. Nothing is mixed in afterwards.
   - The script starts the sweep, opens the inventory sheet once, ends the sweep if the agent has not, waits for the packet, then scrolls the report in the same tab.
4. Convert only: `ffmpeg -i take/take.webm -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac demo.mp4`. This is one continuous take with no cuts and no added audio.
