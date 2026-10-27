# Submission

| # | Deliverable | Where |
|---|---|---|
| 1 | Repository | This private repo. Setup in the [README](../README.md), `.env.example`, no secrets committed |
| 2 | Demo video | [demo.mp4](demo.mp4): one unedited take, 4 min 34 s. The live sweep with the agent's voice, the inventory filling in, and the final packet and report |
| 3 | Claim packet | [claim_packet/claim_packet.json](claim_packet/claim_packet.json), [claim_packet/report.html](claim_packet/report.html), every saved frame the packet references under [claim_packet/frames/](claim_packet/frames/), and the per-stage outputs under [claim_packet/stages/](claim_packet/stages/) |
| 4 | Ground truth and results | Hand-made sheet in [ground_truth/](../ground_truth/) with its source in [SOURCE.md](../ground_truth/SOURCE.md). System numbers against each pass bar are in [docs/results.md](../docs/results.md) |
| 5 | Architecture note | [docs/architecture.md](../docs/architecture.md) |
| 6 | Failure log | [docs/failure-log.md](../docs/failure-log.md): the three worst errors, cost per sweep and latency per stage |

## Read this first

No physical shelf was available. The demo is a real Chromium session running the real app against the real server and the live Gemini session. The camera is a fake device panning over two public shelf photos, and the user's voice is text-to-speech. The video says this on screen for the whole take. How it was recorded: [tools/demo/README.md](../tools/demo/README.md).

Results of this run, sweep `2cfa238aa7524fbeaebaf3f84598bf3b`:

| Bar | Result |
|---|---|
| Book count | PASS: 61 counted against 60 (1.7%) |
| Title identification | FAIL: 13 of 54 legible spines (24%). Confidently wrong 0 of 13 |
| Time to packet | PASS: 78.7 s after the sweep ended |
| Spine dimensions, prices, items, floor, wall | NOT EVALUATED: photos cannot be measured with a ruler or tape, and only 2 books were priced |

Cost: USD 0.083 at paid-tier rates (USD 0 billed on the free tier).
