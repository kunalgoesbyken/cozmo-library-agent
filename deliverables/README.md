# Submission

| # | Deliverable | Where |
|---|---|---|
| 1 | Repository | This private repo. Setup in the [README](../README.md), `.env.example`, no secrets committed |
| 2 | Demo video | [demo.mp4](demo.mp4): one unedited take, 4 min 6 s, recorded live (screen and audio captured together, nothing added afterwards). The live sweep with the agent catching a blurred pass and the redo, the inventory filling in, and the final packet and report |
| 3 | Claim packet | [claim_packet/claim_packet.json](claim_packet/claim_packet.json), [claim_packet/report.html](claim_packet/report.html), every saved frame the packet references under [claim_packet/frames/](claim_packet/frames/), and the per-stage outputs under [claim_packet/stages/](claim_packet/stages/) |
| 4 | Ground truth and results | Hand-made sheet in [ground_truth/](../ground_truth/) with its source in [SOURCE.md](../ground_truth/SOURCE.md). System numbers against each pass bar are in [docs/results.md](../docs/results.md) |
| 5 | Architecture note | [docs/architecture.md](../docs/architecture.md) |
| 6 | Failure log | [docs/failure-log.md](../docs/failure-log.md): the three worst errors, cost per sweep and latency per stage |

## Read this first

No physical shelf was available. The demo is a real Chromium session running the real app against the real server and the live Gemini session. The camera is a fake device panning over two public shelf photos. The user's voice is text-to-speech fed into the microphone. A second tab records the app tab's screen and audio live, so nothing was mixed in afterwards. The video says this on screen for the whole take. How it was recorded: [tools/demo/README.md](../tools/demo/README.md).

Results of this run, sweep `e5bee844d3f74a4b836be8b350e71d93`:

| Bar | Result |
|---|---|
| Book count | PASS: 62 counted against 60 (3.3%) |
| Title identification | FAIL: 30 of 54 legible spines (55.6%). Confidently wrong 1 of 31 (3.2%), arguable ("Die Mosella" against "Mosella") |
| Time to packet | PASS: 58 s after the sweep ended |
| Spine dimensions, prices, items, floor, wall | NOT EVALUATED: photos cannot be measured with a ruler or tape, and only 2 books were priced |

Cost: USD 0.106 at paid-tier rates (USD 0 billed on the free tier).
