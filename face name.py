import os, cv2, time, torch, numpy as np, threading
from collections import deque
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

DATASET_PATH = 'face recog/dataset'
EMBEDDING_PATH = 'embeddings'
THRESHOLD = 0.65
IMG_SIZE = 160

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using {DEVICE}")
torch.backends.cudnn.benchmark = True

print("[INFO] Loading models...")
mtcnn = MTCNN(image_size=IMG_SIZE, margin=10, keep_all=True, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
os.makedirs(EMBEDDING_PATH, exist_ok=True)


def build_embeddings():
    embeddings_dict = {}
    print("[INFO] Building embedding database...")
    for person in tqdm(os.listdir(DATASET_PATH)):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue
        saved = os.path.join(EMBEDDING_PATH, f"{person}.pt")
        if os.path.exists(saved):
            embeddings_dict[person] = torch.load(saved, map_location='cpu')
            continue
        embs = []
        for img_name in os.listdir(person_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            try:
                img = cv2.cvtColor(cv2.imread(os.path.join(person_path, img_name)), cv2.COLOR_BGR2RGB)
            except:
                continue
            faces = mtcnn(img)
            if faces is None:
                continue
            if isinstance(faces, torch.Tensor):
                faces = [faces] if faces.ndim == 3 else list(faces)
            for face in faces:
                with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True):
                    emb = resnet(face.unsqueeze(0).to(DEVICE, non_blocking=True)).squeeze()
                embs.append((emb / emb.norm()).cpu())
        if embs:
            mean_emb = torch.stack(embs).mean(0)
            torch.save(mean_emb, saved)
            embeddings_dict[person] = mean_emb
    return embeddings_dict


embeddings = build_embeddings()
db_names = list(embeddings.keys())
db_tensor = torch.stack([embeddings[n] for n in db_names]).to(DEVICE) if db_names else torch.empty(0, 512, device=DEVICE)
print(f"[INFO] DB has {len(db_names)} persons")


class App:
    def __init__(self):
        self.frame = None
        self.running = True
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        threading.Thread(target=self._capture, daemon=True).start()
        self.prev_time = time.time()

        self.tracks = {}
        self.next_id = 0
        self.iou_thresh = 0.3
        self.max_missing = 3
        self.smooth_win = 5

    def _capture(self):
        while self.running:
            ret, f = self.cap.read()
            if ret:
                self.frame = f

    def iou(self, a, b):
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a_area = (a[2] - a[0]) * (a[3] - a[1])
        b_area = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (a_area + b_area - inter + 1e-6)

    def update_tracks(self, dets):
        self.tracks = {k: v for k, v in self.tracks.items() if v['missed'] < self.max_missing}
        for t in self.tracks.values():
            t['missed'] += 1
        used = set()
        for box, name, face_tensor in dets:
            best_id, best_iou = None, 0
            for tid, t in self.tracks.items():
                if tid not in used:
                    i = self.iou(box, t['box'])
                    if i > best_iou:
                        best_iou, best_id = i, tid
            if best_id is not None and best_iou > self.iou_thresh:
                t = self.tracks[best_id]
                t['box'] = box
                t['missed'] = 0
                t['tensor'] = face_tensor
                t['history'].append(name)
                if len(t['history']) > self.smooth_win:
                    t['history'].popleft()
                used.add(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'box': box, 'missed': 0, 'history': deque([name], maxlen=self.smooth_win), 'tensor': face_tensor}
                used.add(tid)

    def recognize(self, face_crops, boxes):
        if not face_crops:
            return
        batch = torch.stack(face_crops).to(DEVICE, non_blocking=True)
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True):
            embs = resnet(batch)
        embs = embs / embs.norm(dim=1, keepdim=True)
        sims = torch.mm(embs, db_tensor.T)
        vals, idxs = sims.max(dim=1)
        for i, (box, val, idx) in enumerate(zip(boxes, vals, idxs)):
            name = db_names[idx.item()] if val.item() > THRESHOLD else "Unknown"
            self.update_tracks([(box, name, embs[i].cpu())])

    def run(self):
        print("[INFO] Starting...")
        detect_interval = 2
        frame_count = 0

        while self.running:
            if self.frame is None:
                time.sleep(0.001)
                continue

            frame = self.frame.copy()
            frame_count += 1
            do_detect = (frame_count % detect_interval == 0)

            if do_detect:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(img)

                if boxes is not None:
                    face_crops = []
                    det_boxes = []
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(x1, 0), max(y1, 0)
                        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        try:
                            f = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                            t = torch.from_numpy(f).permute(2, 0, 1).float().div(255.0).sub(0.5).div(0.5)
                            face_crops.append(t)
                            det_boxes.append((x1, y1, x2, y2))
                        except:
                            continue

                    if face_crops:
                        self.recognize(face_crops, det_boxes)

                for t in self.tracks.values():
                    if t['missed'] > 0:
                        continue
                    name = max(set(t['history']), key=t['history'].count)
                    if name != "Unknown":
                        x1, y1, x2, y2 = t['box']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                for t in self.tracks.values():
                    if t['missed'] > 0:
                        continue
                    name = max(set(t['history']), key=t['history'].count)
                    if name != "Unknown":
                        x1, y1, x2, y2 = t['box']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            curr = time.time()
            fps = 1 / (curr - self.prev_time)
            self.prev_time = curr
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cap.release()
        cv2.destroyAllWindows()


App().run()

