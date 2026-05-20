<div align="center">
  <h1>👁️ Face Recognition</h1>
  <p>
    <strong>Real-time face detection, recognition & tracking pipeline</strong>
  </p>
  <p>
    Built with <strong>PyTorch</strong> · <strong>MTCNN</strong> · <strong>FaceNet</strong> · <strong>OpenCV</strong>
  </p>
  <br>
</div>

## ✨ Highlights

- **⚡ High FPS** – Separate camera thread eliminates I/O bottlenecks
- **🧠 Deep Learning** – MTCNN detection + InceptionResnetV1 (vggface2) for 512‑dim embeddings
- **💾 Smart Caching** – Embeddings computed once, saved as `.pt` files, loaded instantly
- **🎯 Temporal Smoothing** – IoU tracker with sliding‑window voting eliminates flicker
- **🚀 GPU Ready** – Auto‑detects CUDA, falls back to CPU gracefully

## 🧩 How It Works

```
Camera Feed  →  MTCNN Detection  →  Face Crop (160×160)  →  FaceNet Embedding
     ↓
Cosine Similarity  →  Database Match  →  IoU Tracker  →  Temporal Smoothing
     ↓
OpenCV HUD Overlay
```

## 📁 Structure

```
├── face recog/dataset/       # Your face images (one folder per person)
├── embeddings/               # Auto‑generated .pt embedding cache
├── face name.py              # Main application
└── README.md
```

## ⚙️ Setup

```bash
# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python facenet-pytorch numpy tqdm pillow
```

## 🗂️ Dataset

Create folders inside `face recog/dataset/` named after each person. Place **1+ clear, front‑facing photos** in each folder.

```
face recog/dataset/
├── Alice/
│   ├── alice1.jpg
│   └── alice2.jpg
└── Bob/
    └── bob1.png
```

## ▶️ Run

```bash
python "face name.py"
```

**Controls:** Press `q` on the camera window to exit.

## 🎛️ Configuration

| Constant | Default | Description |
|----------|---------|-------------|
| `THRESHOLD` | `0.65` | Similarity threshold (higher = stricter) |
| `iou_thresh` | `0.3` | IoU threshold for track association |
| `max_missing` | `3` | Frames before dropping a track |
| `smooth_win` | `5` | Temporal voting window size |

---

<div align="center">
  <sub>Built with ❤️ using PyTorch & OpenCV</sub>
</div>
