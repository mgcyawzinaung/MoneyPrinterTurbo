# E2micro MoneyPrinterTurbo-style Kaggle pipeline

This Kaggle adapter follows the stage structure of the forked app/services/task.py from MoneyPrinterTurbo. It does not import the full MPT web app, which is too heavy for an Oracle E2 Micro with 1 GiB RAM.

Stages: script and shot plan, Cloudflare AI image generation, VoxCPM2 GPU narration, scene-aligned subtitles, zoom/fade and MP4 rendering with ffmpeg, and Telegram delivery.

The Cloudflare image scenes are in a private Kaggle dataset:
mgcyaw/e2micro-death-documentary-scenes-20261103

Reusable notebook: death_documentary.ipynb
Kaggle push: kaggle kernels push -p . --accelerator NvidiaTeslaT4
Kaggle run: mgcyaw/e2micro-moneyprinter-voxcpm2-death-20261103

To reuse: edit SCENES, update the Cloudflare-generated images in your Kaggle dataset, and run this notebook again. The separate existing VoxCPM2 notebook remains untouched.

Medical fact sources:
- https://www.cdc.gov/heart-disease/about/cardiac-arrest.html
- https://my.clevelandclinic.org/health/articles/23144-what-happens-when-you-die
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10203028/

Claims about life after irreversible death are explicitly labeled unknown, not scientific fact.
