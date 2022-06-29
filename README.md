# DETR-5-nuclio

DETR-5, a modification of [end-to-end object detection with transformers (DETR)](https://github.com/facebookresearch/detr) with 5 classes, ready to deploy on Nuclio.

## Usage

Run the following code to deploy to nuclio.

```
nuctl create project cvat
nuctl deploy --project-name cvat --path ./ --platform local
```

---

This project was developed as part of Nodeflux Internship x Kampus Merdeka.
