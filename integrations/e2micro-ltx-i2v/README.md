# E2micro × MoneyPrinterTurbo × Cloudflare × VoxCPM2 × Kaggle LTX image-to-video

Reusable adaptation of the MoneyPrinterTurbo stages. A separate Kaggle notebook preserves all existing
working notebooks and does not require the full MoneyPrinterTurbo app on the E2micro's 1 GB RAM.

Pipeline:
1. MoneyPrinterTurbo-style scripted scenes (script.json)
2. Cloudflare Workers AI generated images (scene_00.jpg to scene_07.jpg)
3. VoxCPM2 narration generated earlier on Kaggle (speech_*.wav, narration.wav)
4. LTX 2B model converts the *original* images into short motion clips (motion_*.mp4)
5. FFmpeg matches motion and captions to narration; encodes vertical MP4
6. E2micro sends the MP4 to the configured Telegram owner chat.

New Kaggle notebook: ltx_i2v_documentary.ipynb
Kernel slug: mgcyaw/e2micro-death-ltx2b-i2v-20261103
Input: mgcyaw/e2micro-death-documentary-final-assets-20261103 (private)
GPU: Nvidia Tesla T4; internet enabled.

To run again from E2micro:
    kaggle kernels push -p /home/ubuntu/e2mpt-death/ltx_upgrade --accelerator NvidiaTeslaT4
    kaggle kernels status mgcyaw/e2micro-death-ltx2b-i2v-20261103
    python3 /home/ubuntu/e2mpt-death/ltx_upgrade/deliver_ltx.py

The notebook first tries the official LTX 2B 0.9.8 distilled checkpoint. If it fails, it
explicitly labels LTX-Video-0.9.5 fallback (non-distilled). If scene generation fails, a
Ken Burns scene is substituted so original VoxCPM2 narration is preserved. Inspect
generation_report.json for the model and which scenes achieved actual AI motion.

Private Telegram credentials are read from E2micro's existing secrets.env; none are
stored in the notebook, committed to GitHub or printed. Existing working notebooks
remain unchanged.
