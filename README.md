# Invisibility Cloak – Video Processing Version

This project takes an **input video**, detects a cloak in batches of **30 frames at a time**, replaces the cloak region with a saved background, and writes a **smooth invisible‑effect video** as output.

---


##  Features

* **Batch processing (30‑frame windows)** for better throughput than naïve frame‑by‑frame loops.
* Custom‑trained **YOLOv5** model (`best.py`) to detect the cloak class.
* Background capture & masking with **OpenCV** for a seamless invisibility effect.
* Works with any `.mp4`, `.avi`, or `.mov` input.

---

##  Requirements

| Tool          | Version (tested)             |
| ------------- | ---------------------------- |
| Python        | 3.9+                         |
| OpenCV‑Python | ≥ 4.7                        |
| PyTorch       | ≥ 1.13                       |
| YOLOv5 repo   | *only* `best.py` script kept |

Install dependencies:

```bash
pip install -r requirements.txt  # generate via pip freeze > requirements.txt
```

---

##  Running the Script

```bash
python best.py \
  --source input_video.mp4 \
  --output output_video.mp4 \
  --batch_size 30            # number of frames processed per pass
```

**Parameters**

| Flag           | Default   | Description                         |
| -------------- | --------- | ----------------------------------- |
| `--source`     | ―         | Path to input video                 |
| `--output`     | `out.mp4` | Path to save processed video        |
| `--batch_size` | `30`      | Frames processed together for speed |
| `--conf_thres` | `0.25`    | YOLO confidence threshold           |

---


## How It Works

1. **Background capture**: first few frames without the cloak are averaged.
2. **Batch inference**: YOLOv5 detects cloak masks for 30‑frame chunks.
3. **Mask & replace**: cloak region replaced with background for each frame.
4. **Writer**: processed frames are written sequentially into `output_video.mp4`.

Batching reduces model‑inference overhead, yielding a smoother final video.

---

## Author

**Aqsa Saleem**
---

