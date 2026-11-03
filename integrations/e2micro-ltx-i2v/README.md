# E2micro × MoneyPrinterTurbo × Cloudflare × VoxCPM2 × Kaggle LTX image-to-video

A two-stage video production adapter that follows MoneyPrinterTurbo's
script → assets → narration → video → subtitle → delivery workflow.

This runs through the E2micro controller but does the expensive work on free
Kaggle compute. The existing MoneyPrinterTurbo fork, all pre-existing VoxCPM2
notebooks, and the running Telegram bot remain unchanged.

## Inputs
- Eight Cloudflare Workers AI scenes in the private dataset:
  mgcyaw/e2micro-death-documentary-final-assets-20261103
- Voice recordings generated earlier with VoxCPM2 in the same dataset,
  including narration.wav, speech_00.wav ... speech_07.wav and script.json.

## Stage 1 — Kaggle GPU
- Notebook: ltx_i2v_documentary.ipynb
- Kaggle slug: mgcyaw/e2micro-death-ltx2b-i2v-20261103
- Tries official LTX 2B 0.9.8 distilled native checkpoint, with an explicit
  non-distilled LTX 0.9.5 fallback if loading fails.
- Generates 8 short image-conditioned motion MP4s and generation_report.json.
- Only performs GPU inference. Video encoding is separated so font/FFmpeg
  errors cannot waste the generated model output.

## Stage 2 — Kaggle CPU (successfully verified)
- Notebook: render_ai_motion.ipynb
- Kernel slug: mgcyaw/e2micro-ai-motion-final-20261103
- Consumes a private clips dataset (currently
  mgcyaw/e2micro-ltx-motion-clips-20261103) plus the original VoxCPM2 assets.
- Applies subtitles with a portable font fallback, repeats short AI clips
  to match spoken segments, mixes quiet instrumental audio, and exports a
  720x1280 MP4 named what_happens_when_we_die_AI_animated.mp4.

## Stage 3 — E2micro Telegram
- Script: deliver_ltx.py
- Reads TG_BOT_TOKEN and TG_OWNER_ID from the existing private E2micro
  secrets.env at runtime.
- Downloads the completed CPU render from Kaggle, verifies that motion
  scenes were genuinely AI generated, then delivers exactly once to the bot.
- Delivery result for the first successful 0.9.5 motion run: message #138,
  8/8 actual AI animation scenes, 62.24 seconds, 4.49 MB.

## Notes
- This is an adapter of MPT's workflow stages, not a deployment of the full
  MoneyPrinterTurbo web app on E2micro's 1 GiB RAM.
- The original GPU version generated all 8 images successfully but failed
  when its final caption cell requested a missing font. We recovered all
  eight files and successfully rendered them on a separate CPU notebook.
  The code here now deliberately ends the GPU job after motion generation.
- Model choice and per-scene success are recorded in generation_report.json.
- Private datasets and Telegram credentials must never be committed to GitHub.
- Kaggle and Cloudflare free quotas apply; there is no automatic paid upgrade.
