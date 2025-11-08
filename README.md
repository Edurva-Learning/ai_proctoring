# ai_proctoring (proct)

This repository contains the proctoring scripts and assets found in the local folder `proct` (face/eye/mouth tracking, object detection, etc.).

Notes for the initial commit:

- Large model or weight files (for example `*.pt`, `*.weights`) are intentionally excluded via `.gitignore`. If you want to track those files in GitHub, use Git LFS or host them externally (S3, release assets, etc.).
- This initial commit was created from the local `proct` folder on the user's machine.

How to push (if the automated push fails):

1. Ensure you have permission to push to `https://github.com/Edurva-Learning/ai_proctoring.git`.
2. If you need to track large models, install Git LFS and run `git lfs install` and `git lfs track "*.pt"` before committing the model files.
