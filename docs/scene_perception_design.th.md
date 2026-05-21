# ระบบ Scene Embedding และ Background Perception — Phase 1 + 2 + 3

Branch: `feat/scene-perception`
สถานะ: **Phase 3 implementation เสร็จเรียบร้อย** โมดูลอยู่ที่ `perception/` ผ่าน test 42 รายการ ดูสรุปได้ที่หัวข้อ "Phase 3 — สรุปการ implement" ท้ายเอกสาร

> **มี Addendum เรื่อง motion + dedup behavior รวมเข้ามาแล้ว** หุ่นเคลื่อนที่ได้ loop ทำงานต่อเนื่องไม่จำกัด การเห็นซ้ำเป็นเรื่องปกติ ส่วนที่แก้จาก addendum จะมีแท็ก **[Addendum]** กำกับ

> เอกสารฉบับนี้เป็น **เวอร์ชันภาษาไทย** แปลและขยายความจาก `scene_perception_design.md` (ภาษาอังกฤษ) ซึ่งเป็น source of truth ถ้ามีจุดไหนขัดกัน ให้ยึดเวอร์ชันอังกฤษ

## TL;DR

`perception/` คือ Python package ที่รัน async loop ทำงานตลอดเวลา เรียก AI server เพื่อทำ object detection + image caption + (ในอนาคต) CLIP embedding เรียก walkie-sdk `Tools.bboxes_to_positions` เพื่อยก detection แต่ละชิ้นเป็น 3D world-frame position แล้ว upsert ผลลัพธ์ลง ChromaDB collection เดียว query แบบต่างๆ (semantic / spatial / recency / diff) ก็มาจาก collection เดียวกันนี้

**Phase 3 implementation เสร็จและผ่าน test ครบแล้ว** (42 test ผ่าน รัน ~20 วินาที) blocker ของ production ตอนนี้เหลือเรื่องเดียวคือ CLIP — walkie-ai-server มี CLIP embedding service implement ไว้แล้วแต่ comment route ทิ้งไว้ โค้ดเรารองรับผ่าน `Embedder` Protocol แล้ว เพียงแค่ server เปิด blueprint นั้นก็พร้อมเสียบใช้งานได้ทันที

## แผนที่ของเอกสาร

- **Phase 1 — ผลการสำรวจ (Discovery)** — สิ่งที่อ่านมาจาก repo ต้นทางสองตัว, API ที่เราพึ่งพา, capability ที่ขาดหายไป
- **Phase 2 — Design** — schema, dedup strategy, query API, retention policy ส่วนที่มีแท็ก `[Addendum]` คือส่วนที่เขียนใหม่หลังได้รับ addendum
- **คำถามที่ยังเปิดอยู่** — 4 คำตอบที่ต้องตัดสินใจ พร้อมสถานะของ Phase 3 ในแต่ละข้อ
- **Phase 3 — สรุปการ implement** — รายการไฟล์ในโมดูล, รายการ test, งานต่อเนื่อง

---

## Phase 1 — ผลการสำรวจ (Discovery)

### โค้ดอยู่ที่ไหนบนเครื่อง

| สิ่งที่ใช้ | path | หมายเหตุ |
|---|---|---|
| Consumer (repo นี้) | `/home/hextex/Documents/GitHub/walkie-agent-v2/` | `services/perception.py` + `services/explore.py` ทำ version ที่ง่ายกว่าของเรื่องนี้อยู่แล้ว ทั้งสองไฟล์จะถูกแทนที่ด้วยโมดูลใหม่ |
| source ของ walkie-sdk | uv sync จะ resolve ไป commit `025ee9b` (`walkie-sdk==0.2.0`) ใน `.venv/lib/python3.12/site-packages/walkie_sdk/` | `main` branch บน GitHub มี API ครบ — `arm/camera/tools/multi_camera/visualization` ครบหมด clone ที่ `/home/hextex/Documents/GitHub/Walkie-SDK/` เป็นเวอร์ชันเก่า (มีแค่ 3 commit) ทำให้เข้าใจผิดตอนสำรวจครั้งแรก `pyproject.toml` ประกาศ git source แบบไม่ pin ref แต่ `uv.lock` pin commit ที่แน่นอนไว้ ดังนั้นถ้า commit lockfile ไปด้วย reproducibility ก็ยังโอเค |
| source ของ walkie-ai-server | `/home/hextex/Documents/GitHub/walkie-ai-server/` | Flask app ที่ port 5000 |

### walkie-sdk API ที่เราจะใช้

ทุก signature ตรวจสอบกับ `walkie_sdk/modules/` แล้ว:

```python
WalkieRobot(
    ip: str,
    ros_protocol: str = "rosbridge" | "zenoh",
    ros_port: int = 9090,
    camera_protocol: str = "webrtc" | "zenoh" | "shm" | "none",
    camera_port: int = 8554,
    timeout: float = 10.0,
    namespace: str = "",
)

bot.camera.get_frame() -> np.ndarray | None        # BGR HxWx3 uint8, frame ล่าสุดจาก cache, ไม่ block
bot.camera.is_streaming -> bool
bot.camera.frame_shape -> (H, W, C) | None

bot.status.get_position() -> {"x": float, "y": float, "heading": float} | None  # heading หน่วยเป็น radian; แม้ชื่อ method จะเป็น "position" แต่ payload มี orientation รวมอยู่ด้วย
bot.status.get_velocity() -> {"linear": float, "angular": float} | None

bot.tools.bboxes_to_positions(
    coords: list[list[float]],   # แต่ละ bbox เป็น [cx, cy, w, h] ในพิกัด pixel
    timeout: float = 5.0,
) -> list[list[float]] | None    # แต่ละจุดเป็น [x, y, z] ใน frame ของ YOLO-3D ต้นทาง (มักเป็น `map`); คืน None ถ้า timeout
```

`bboxes_to_positions` ทำงานแบบ **pub/sub request-reply**: publish ข้อความ `vision_msgs/Detection2DArray` ไปยัง topic `/yolo/detections_2d` แล้วรอ `geometry_msgs/PoseArray` กลับมาที่ topic `/ob_detection/poses` ภายใน `timeout` วินาที จากนั้นแปลง poses กลับมาเป็น list ของ `[x,y,z]` ลำดับของ output ตรงกับลำดับ input คืน `None` ถ้าไม่ได้รับ response ทันเวลา

**เรื่องการตั้งชื่อใน Telemetry (ไม่ใช่ bug แต่ควรรู้ไว้)**: method ชื่อ `get_position()` แต่ payload มี `heading` รวมอยู่ด้วย — จริงๆ มันคือ *pose* 2D (position + orientation) ที่แปะ label ว่า position SDK เคยใช้ชื่อ `get_pose()` ในเวอร์ชันก่อน แล้ว rename เป็น `get_position()` ก่อน commit `025ee9b` consumer ที่ `agents/actuator_agent/tools.py:24` เรียก method ปัจจุบันถูกต้องแล้ว ถ้าเจอ `get_pose` ใน doc/branch เก่าให้ถือว่าเป็น call เดียวกัน

### walkie-ai-server endpoint ที่เราจะใช้

ตรวจสอบกับ `walkie-ai-server/api/routes/` แล้ว:

| Capability | Endpoint | Request | Wrapper ใน repo นี้ | Return |
|---|---|---|---|---|
| Object detection | `POST /object-detection/detect` | multipart `image` | `client.ObjectDetectionClient.detect(img)` | `list[DetectedObject(bbox=(x1,y1,x2,y2), class_name, confidence, area_ratio, mask=None)]` |
| Image captioning | `POST /image-caption/caption` | multipart `image`, form `prompt?` | `client.ImageCaptionClient.caption(img, prompt=…)` | `str` |
| Image captioning (batch) | `POST /image-caption/caption-batch` | multipart `images[]`, form `prompts[]?` | `client.ImageCaptionClient.caption_batch(imgs, …)` | `list[str]` |
| Pose estimation | `POST /pose-estimation/estimate` | multipart `image` | `client.PoseEstimationClient.estimate(img)` | `list[PersonPose(bbox, confidence, keypoints=17×COCO)]` |
| STT / TTS | `/stt/transcribe`, `/tts/synthesize{,-stream}` | — | — | ระบบ perception ไม่ใช้ |

response ทุกตัว wrap อยู่ใน `{"success": bool, "data": …}` `client/base.py::_unwrap` ลอกเปลือกออกให้ และ raise `WalkieAPIError` ถ้า server ตอบ failure

### Capability ที่ขาด — **ต้องตัดสินใจ**

`walkie-ai-server/services/image_embed/` มี CLIP provider ครบเรียบร้อย (`openai/clip-vit-base-patch16` ขนาด 512 มิติ ทำได้ทั้ง image embedding, text embedding และ cosine similarity) ไฟล์ route `api/routes/image_embed.py` expose:

- `POST /image-embed/embed-image` → `{embedding, dim}`
- `POST /image-embed/embed-text` → `{embedding, dim}`
- `POST /image-embed/similarity` → `{similarity}`

**แต่ใน `api/__init__.py` blueprint ถูก comment ทิ้งไว้** (บรรทัด 16: `# app.register_blueprint(image_embed.bp)`) ตอนนี้ endpoint จะตอบ 404

มีสองตัวเลือก:

1. **เปิด blueprint ใน AI server** แก้แค่บรรทัดเดียวใน `walkie-ai-server/api/__init__.py` + deploy ก็จะได้ joint image+text embedding มาเลย semantic query จะ match ได้ทั้ง caption และ visual content และ cross-modal `similarity` endpoint ก็จะเป็นกระดูกสันหลังตามธรรมชาติของ query แบบ "ขอดูที่ X ที"
2. **Embed ที่ฝั่ง agent เอง** ใส่โมเดล `sentence-transformers` (เช่น `all-MiniLM-L6-v2`, 384 มิติ) เข้ามาใน repo นี้แล้ว embed เฉพาะ *ข้อความ caption* query จะ match แค่ฝั่ง caption ไม่ได้ visual half ของ CLIP ขนาด deployment เล็กลง แต่ช้าลงต่อ tick (~30–80 ms ของ CPU ถ้า batch caption หลายๆ ตัว) และเสียความสามารถ "หาภาพด้วยภาพ" ไป

**แนะนำตัวเลือกที่ 1** ฟรี โมเดลก็ wire ไว้แล้ว และ CLIP image+text ที่อยู่ใน space เดียวกันคือ primitive ที่ถูกต้องสำหรับปัญหานี้ smoke test `tests/perception/test_smoke_image_embed.py` ก็ pin contract ที่เราจะใช้ไว้แล้ว

---

## Phase 2 — Design

### โครงสร้างของโมดูล

```
perception/                       (จะแทนที่ services/perception.py + services/explore.py เมื่อ migrate main.py)
├── __init__.py    # public re-exports
├── loop.py        # async background loop (cancel ได้, ปรับ rate ได้)
├── pipeline.py    # 1 frame → list[Detection] (เรียก AI server + walkie-sdk)
├── store.py       # SceneStore (wrapper ของ ChromaDB: upsert, dedup, queries)
├── dedup.py       # decision logic แบบ pure-function: classify() + ฟังก์ชันดึง threshold
├── types.py       # SceneEntry, Detection, DedupDecision, SceneDiff, TickReport + Protocols
└── mocks.py       # FakeCamera, FakeDetector, FakeCaptioner, FakeEmbedder, FakePositionLifter (เฉพาะ test)
```

6 module ไม่มี inheritance ประกอบกันใน `loop.py` แต่ละ module unit-test ได้แยกกัน Protocol ใน `types.py` ทำให้ mock สามารถ swap แทน collaborator จริงได้แบบ drop-in

### ChromaDB schema — collection เดียวชื่อ `scene_entries`

```python
chroma_client.get_or_create_collection(
    name="scene_entries",
    metadata={"hnsw:space": "cosine"},
)
```

| Chroma field | เก็บอะไร | เพราะอะไร |
|---|---|---|
| `ids` | `f"{class_name}:{spatial_bucket_x}:{spatial_bucket_y}:{spatial_bucket_z}:{uuid4_short}"` | UUID ส่วนท้ายทำให้ insert idempotent ข้าม restart bucket prefix แค่ *เพื่อ debug* — dedup decision ไม่ได้อ่าน id แต่ query ใหม่ทุกครั้ง |
| `embeddings` | CLIP `embed_image` ของ **crop จาก bbox region** ของ frame ต้นทาง, 512 มิติ, L2-normalized | visual match อยู่รอดเมื่อ caption เปลี่ยนคำพูด ใช้ crop (ไม่ใช่ full frame) เพื่อให้สองวัตถุที่อยู่ติดกันได้ vector ต่างกัน |
| `documents` | `f"{class_name}. {caption}"` | ทำให้ default text-embedding fallback ของ Chroma ใช้งานได้แม้ visual embedding หายไป (graceful degradation) และทำให้ `.peek()` อ่านง่าย |
| `metadatas` (dict ต่อหนึ่ง record) | ดูด้านล่าง | |

#### Metadata fields

```jsonc
{
  // Identity
  "class_name":      "chair",        // YOLO label
  "class_id":        56,
  "first_seen_ts":   1716240000.0,   // epoch seconds, set ตอน insert
  "last_seen_ts":    1716240320.0,   // epoch seconds, set ทุกครั้งที่ update
  "sightings":       7,              // ตัวนับขาขึ้น (ใช้กรอง transient noise)

  // Position (frame map จาก bboxes_to_positions)
  "x": 1.23, "y": -0.45, "z": 0.78,
  "position_frame":  "map",          // pin ไว้เผื่อ upstream เปลี่ยน
  "position_conf":   0.91,           // ค่าเฉลี่ยรันนิ่ง

  // Visual metadata สำหรับผลของ query
  "caption":         "เก้าอี้กินข้าวไม้ เอียงเล็กน้อย",
  "bbox_last":       "[120,80,260,400]",   // (x1,y1,x2,y2) ของการเห็นล่าสุด JSON-encode เพราะ Chroma เก็บ list ใน metadata ไม่ได้
  "frame_ref":       "frames/2026-05-21T18-22-13Z_chair_abc123.jpg",  // path บน disk; None ถ้าไม่ได้เก็บ frame

  // Embedding hygiene
  "embedding_model": "clip-vit-base-patch16",  // ใช้ invalidate ถ้าวันหลังเปลี่ยน model
  "embedding_dim":   512,
}
```

ค่าใน metadata ของ ChromaDB ต้องเป็น scalar (str/int/float/bool) เท่านั้น — list ต้อง JSON-encode เป็น string trade-off คือ spatial filter ใช้ column `x`/`y`/`z` ที่เป็น scalar (Chroma `where=` รับ numeric range filter ได้) และเราไม่เคยต้อง filter ด้วย raw bbox อยู่แล้ว

#### Collection เดียว หรือแบ่งตามห้อง/zone?

**Collection เดียว** เหตุผล:
- เรายังไม่มี room classifier ที่เชื่อถือได้ ถ้าจะ partition ต้อง label ด้วยมือ หรือมี `RoomMembership` classifier (ยังไม่มี)
- Spatial filter ผ่าน `where={"x": {"$gte": ...}, ...}` เร็วพอที่ scale ที่เราคาด (~10k record)
- Query "ในครัว" กลายเป็น metadata range query ไม่ใช่การเลือก collection — evolve ได้ถูกกว่า

วันที่มี zone metadata จริงๆ ก็แค่เพิ่ม field `zone: str` ใน metadata แล้ว filter ไม่ต้อง migrate record

### Dedup / upsert strategy

เมื่อ detection ใหม่มาถึง เรามี:

```
new = (class_name, position=(x, y, z), embedding, caption, confidence, bbox, ts, frame_ref)
```

Decision tree (อยู่ใน `perception/dedup.py`):

```
1. CANDIDATES = scene_store.find_nearby(
       class_name=new.class_name,
       position=new.position,
       radius=SPATIAL_RADIUS,           # 0.5 m (default)
   )

2. ถ้า CANDIDATES ว่าง:
       → INSERT new

3. สำหรับ candidate c แต่ละตัวใน CANDIDATES (เรียงตาม L2 distance น้อย→มาก):
       cos_sim = dot(c.embedding, new.embedding)         # ทั้งคู่ L2-normalized
       ถ้า cos_sim >= EMB_SIM_HIGH:                       # 0.85
           → UPDATE c (running-mean position, sightings +1, refresh caption, last_seen_ts)
           return
       ถ้า cos_sim >= EMB_SIM_LOW และ L2(c, new) <= TIGHT_RADIUS:   # 0.65 และ 0.2 m
           → UPDATE c   (เหมือนกันมาก AND อยู่ใกล้กันมาก — ถือว่าตัวเดียวกัน)
           return

4. → INSERT new (ผ่าน merge test ทั้งหมดไม่ได้)
```

#### ทำไม threshold ถึงเป็นค่าเหล่านี้

**[Addendum]** addendum ขอให้อธิบาย `τ_pos` เทียบกับ noise ของ converter ของ walkie-sdk `bboxes_to_positions` คือ YOLO-3D pipeline ที่ฝั่งหุ่น — ความแม่นยำขึ้นกับ depth sensor และ TF chain ที่นำหน้ามา เท่าที่สังเกต (และจากที่ทีม EIC เคยทำมา): jitter ของวัตถุเดียวจาก viewpoint คงที่อยู่ราว **5–15 cm**; ความไม่ตรงกันของวัตถุเดียวจาก viewpoint ต่างกันที่ระยะ 2–3 ม. อยู่ราว **15–30 cm** ถ้า TF calibrate ดี **0.5 ม. ให้ margin ~2–3 เท่าของ noise ข้าม viewpoint แบบทั่วไป** ขณะที่ยังเล็กกว่าระยะห่างปกติระหว่างวัตถุในห้อง (เก้าอี้รอบโต๊ะมักห่างกันอย่างน้อย 0.6 ม. center-to-center) ถ้า deployment ไหน 3D noisy กว่านี้ override ผ่าน `SCENE_DEDUP_RADIUS_M` ได้

| Threshold | ค่า default | เหตุผล |
|---|---|---|
| `SPATIAL_RADIUS` (`τ_pos`) | **0.5 m** | margin 2–3 เท่าของ noise converter ข้าม viewpoint (ดูด้านบน) เล็กกว่านี้เสี่ยงแยกเก้าอี้ตัวเดียวจากสองมุมมองเป็นสอง record ใหญ่กว่านี้เสี่ยง merge เก้าอี้สองตัวที่โต๊ะกินข้าวเดียวกัน |
| `EMB_SIM_HIGH` (`τ_sim`) | **0.85** (cosine) | CLIP ViT-B/16 instance เดียวข้าม viewpoint คะแนนตกที่ 0.80–0.95; instance ต่างกันแต่ class เดียวกันตกที่ 0.55–0.80 0.85 คือ cut conservative ที่เลือก "แยกไว้ก่อน" ดีกว่า "merge ผิด" |
| `EMB_SIM_LOW` + `TIGHT_RADIUS` | **0.65 + 0.2 m** | กลไก failsafe สำหรับกรณีที่แสงหรือ occlusion ทำให้ similarity ตกต่ำกว่า 0.85 แต่ bbox lift มาอยู่ในระยะ 20 cm ของ entry เดิม spatial gate ที่แคบนี้กันไม่ให้วัตถุต่างชิ้นที่อยู่บนโต๊ะเดียวกันถูก fuse |

ทั้ง 4 ค่าเป็น constant ใน `perception/dedup.py` override ได้ผ่าน env var (`SCENE_DEDUP_RADIUS_M`, `SCENE_EMB_SIM_HIGH`, ...) ตามแบบที่ `services/explore.py` ใช้อยู่

#### Semantic ของ UPDATE (เมื่อ UPDATE ไม่ใช่ INSERT)

**[Addendum]** addendum ขอให้ระบุพฤติกรรมของ position smoothing เราใช้ **running mean ถ่วงด้วย sightings** ไม่ใช่ EMA เหตุผล: ในฉาก static variance ของ running mean จะหดตามอัตรา 1/N ให้ค่าประมาณตำแหน่งที่นิ่งที่สุดเมื่อเห็นซ้ำเป็นร้อยครั้ง — ซึ่งเป็นพอดี case ที่ test 200-tick stare ทดสอบอยู่ EMA (α < 1) ตอบสนองการเคลื่อนที่จริงได้เร็วกว่า แต่ในสภาพแวดล้อม static ความ "ไว" นั้นคือ noise sensitivity เปล่าๆ การเคลื่อนที่จริงจัดการโดย branch "moved by human" (ด้านล่าง): เมื่อวัตถุย้ายไปไกลกว่า `τ_pos` spatial gate ก็ปฏิเสธการ merge แล้ว insert record ใหม่ — ซึ่งเป็นพฤติกรรม tracking ที่ถูกต้องในเมื่อเราไม่มี object-permanence model

- `position` ← running mean ถ่วงด้วย sightings: `new_pos = (old_pos * n + new_pos) / (n+1)`
- `position_conf` ← running mean ของ detection confidence (สูตรเดียวกัน)
- `sightings` ← `n + 1` (ตรงกับ `observation_count` ใน addendum)
- `last_seen_ts` ← `ts` ของ detection
- `first_seen_ts` ← **คงไว้** (ไม่ทับตอน update)
- `caption` ← caption ใหม่แทน caption เก่า (ล่าสุดชนะ)
- `bbox_last` ← bbox ล่าสุด
- `frame_ref` ← **ไม่ update** ตอน UPDATE archive frame ครั้งเดียวต่อ INSERT เท่านั้น เพื่อให้พื้นที่ disk โตตามจำนวนวัตถุที่ต่างกัน ไม่ใช่ตามจำนวนครั้งที่เห็น trade-off: คนที่อยากได้ frame ล่าสุดของวัตถุที่อยู่นานๆ ต้อง capture เอง; `frame_ref` ที่เก็บไว้คือการเห็น *ครั้งแรก*
- `embedding` ← **คงค่าเดิม** (อย่าเฉลี่ย vector — มันจะ drift เข้าหา mean ของ class แล้วทำให้ dedup ในอนาคตแย่ลง)

### วัตถุย้ายระหว่าง session

**[Addendum]** กรณี "คนย้ายวัตถุ" *ไม่ใช่* branch พิเศษใน dedup tree เป็นผลที่ตามมาตามธรรมชาติของ spatial gate:

1. Detection ที่ตำแหน่ง B เข้ามา โดย `dist(B, R_at_A) > τ_pos`
2. `find_nearby` ไม่คืน `R_at_A` เป็น candidate
3. `classify` ตอบ `INSERT`
4. record ใหม่ที่ B ลงไปใน store **`R_at_A` ไม่ถูกแตะ** — `last_seen_ts` ของมันหยุดเดินหน้า ก็เก่าไปตามเวลาผ่าน TTL หรือไปโผล่เป็น "disappeared" ใน `diff(since=now − minutes)`

ไม่มี flag "stale" ชัดเจน ผู้เรียกตัดสินจากการแก่ของ `last_seen_ts` (`SceneDiff.disappeared` partition ก็ surface เรื่องนี้) ถ้าอยากได้ flag boolean ใน metadata เพื่อ filter ที่ถูกกว่า ก็เพิ่มได้บรรทัดเดียว — แต่ตอนนี้การ filter ตาม timestamp ครอบ use case ที่ agent ใช้อยู่ (`recency_query`, `diff`)

### หายแล้วกลับมา (วัตถุกลับมาตำแหน่งเดิม)

วัตถุหายไปแล้วกลับมาที่ตำแหน่ง world-frame *เดิม* *ไม่ใช่* case พิเศษ: เมื่อกลับมา dedup เจอ record ที่นอนรออยู่ (ยังอยู่ใน store แค่ `last_seen_ts` เก่า) cosine match ผ่าน `EMB_SIM_HIGH` แล้ว UPDATE — `sightings` เพิ่ม `last_seen_ts` refresh `first_seen_ts` คงไว้ ไม่มี record ซ้ำ test `test_08_disappear_then_reappear_merges` pin พฤติกรรมนี้ไว้

### Class equivalence — ตอนนี้ strict

**[Addendum]** addendum พูดถึง "Class agreement (หรือทั้งคู่อยู่ใน class-equivalence group เดียวกัน)" ตอนนี้ `classify()` raise ถ้า candidate ข้าม class — ต้อง match class ตรงเป๊ะ default นี้ปลอดภัยกว่าเพราะ label noise ข้าม class ของ YOLO เกิดน้อยกว่า embedding drift ข้าม viewpoint มาก ถ้าวันหลังอยาก merge `"mug"` กับ `"cup"` จาก checkpoint ต่างกัน จุดที่ควรเพิ่มคือ mapping `CLASS_EQUIVALENCE: dict[str, str]` ที่ `find_nearby` ปรึกษา (canonicalize ก่อน where-clause) store ยังเก็บ class *ดิบ* เอาไว้ audit เหมือนเดิม ยังไม่ทำ — flag ไว้ว่าเป็น follow-up ถ้าโมเดลผลิต label ขัดแย้งกัน

### Audit logging ของทุก decision

**[Addendum]** upsert ทุกครั้ง emit log INFO line เดียวที่มีโครงสร้างชัดเจน ผ่าน logger ชื่อ `perception.store`:

```
scene.dedup action=INSERT id=chair:0:0:0:abc12345 matched_id=null    dist=nan   sim=nan   reason=no candidates within radius
scene.dedup action=UPDATE id=chair:0:0:0:abc12345 matched_id=chair:0:0:0:abc12345 dist=0.080 sim=0.987 reason=cosine 0.987 ≥ EMB_SIM_HIGH (0.85); dist 0.08m
scene.dedup action=INSERT id=mug:5:5:0:xyz78901   matched_id=null    dist=nan   sim=nan   reason=2 candidate(s) failed both merge gates
```

field: `action` (INSERT/UPDATE), `id` (record ที่ได้รับผล), `matched_id` (candidate ที่ใกล้ที่สุด แม้กรณี INSERT ก็มี — ใช้ audit near-merge ได้), `dist` (L2 ถึง candidate ที่ใกล้ที่สุด เป็น NaN ถ้าไม่มี candidate ในรัศมี `τ_pos`), `sim` (cosine กับ candidate ที่ใกล้ที่สุด), `reason` (gate ไหนทำงาน หรืออะไร fail) ใน production pipe ออก JSONL file ไว้ทำ audit behavior offline

### Query API surface — `perception.store.SceneStore`

```python
class SceneStore:
    # Writes
    def upsert(self, entry: SceneEntry) -> str: ...
        # คืน chroma id ของ record ที่ได้รับผล log dedup decision ด้วย

    # Reads
    def semantic_query(
        self,
        text: str,
        n_results: int = 5,
        min_last_seen_ts: float | None = None,    # ขอบล่างของความใหม่ (epoch s)
        within_radius_of: tuple[float, float, float] | None = None,
        max_distance_m: float | None = None,
    ) -> list[SceneEntry]: ...
        # CLIP text embed → cosine knn บน Chroma + filter หลังด้วย spatial / recency

    def visual_query(
        self,
        image: PIL.Image,
        n_results: int = 5,
        # … filter เดียวกับ semantic_query
    ) -> list[SceneEntry]: ...

    def spatial_query(
        self,
        center: tuple[float, float, float],
        radius_m: float,
        class_name: str | None = None,
        n_results: int | None = None,
    ) -> list[SceneEntry]: ...
        # ไม่ใช้ vector search — filter ด้วย metadata range อย่างเดียว แล้วเรียงตามระยะ

    def recency_query(
        self,
        since_ts: float,
        class_name: str | None = None,
        n_results: int | None = None,
    ) -> list[SceneEntry]: ...

    def diff(
        self,
        since_ts: float,
        within: tuple[tuple[float, float, float], float] | None = None,  # (center, radius_m)
    ) -> SceneDiff:
        """คืน: appeared (first_seen > since_ts), refreshed (last_seen > since_ts
        และ first_seen <= since_ts), disappeared (last_seen <= since_ts)"""

    # Maintenance
    def prune(self, *, ttl_sec: float | None = None, max_records: int | None = None) -> int: ...
        # คืนจำนวนที่ถูกลบ ดู retention policy ด้านล่าง
```

ทุก read return `list[SceneEntry]` (frozen dataclass) — ผู้เรียกไม่เคยเห็น raw dict ของ Chroma เลย tool `find_object_from_memory` ที่ agent มีอยู่จะถูก rewire ให้เรียก `semantic_query`

### Retention policy

มี 2 ปุ่ม ปิดไว้ default แต่ config ได้:

- **TTL บน `last_seen_ts`** (`SCENE_TTL_SEC` เช่น `86400` คือ 24 ชม.) `prune()` ลบ record ที่ `last_seen_ts` เก่ากว่า cutoff query "ครั้งล่าสุดที่เห็น remote อยู่ไหน?" ควรยังตอบได้บนข้อมูลอายุเป็นสัปดาห์ ดังนั้น TTL ยัง optional ไว้ก่อน
- **เพดานจำนวน record แข็ง** (`SCENE_MAX_RECORDS` เช่น `5000`) `prune()` ลบ record ที่ `last_seen_ts` เก่าที่สุดเมื่อเกินเพดาน เพื่อให้ HNSW index ของ Chroma เร็วอยู่
- **นโยบายเก็บ frame**: ไม่เก็บ full frame ใน Chroma pipeline เขียน frame ต้นทางลง `frames/{ts}_{class}_{id8}.jpg` เมื่อ (และเฉพาะเมื่อ) เกิด INSERT update ไม่ archive frame ใหม่ ฟังก์ชัน `prune_frames()` แยกต่างหากสำหรับลบ frame ที่กำพร้า (ไม่มี record ชี้หา) ทำให้ disk footprint โตตามจำนวนวัตถุที่ต่างกัน ไม่ใช่ตามเวลารัน

### Background loop semantics

```python
async def run_scene_perception(
    *,
    camera: CameraSource,       # มี .capture_pil() — wrap walkie.camera.get_frame()
    detector: Detector,         # client object detection ของ walkie-ai-server
    captioner: Captioner,
    embedder: Embedder,         # CLIP image embed
    lifter: PositionLifter,     # walkie.tools.bboxes_to_positions
    store: SceneStore,
    interval_sec: float = 2.0,
    on_tick: Callable[[TickReport], None] | None = None,
) -> None:
    ...
```

- เป็น `asyncio.Task` driver ล้วน — `await asyncio.sleep(interval_sec)` ระหว่าง tick **ห้าม** `time.sleep`
- inference call ทั้งหมดใน 1 tick รัน concurrent ผ่าน `asyncio.gather` รอบๆ HTTP client ที่ wrap ใน `asyncio.to_thread` (`client/` เดิมใช้ `requests` แบบ sync เรา wrap ไม่ rewrite)
- 1 tick *ไม่เคย* block main loop ของ agent เพราะ task รันบน event loop เดียวกับ agent และ yield ทุก `await` ถ้า tick เกิน `interval_sec` tick ถัดไปเริ่มทันที (ไม่ทับซ้อน — loop รัน sequential strictly ต่อ task)
- shutdown แบบ graceful: `task.cancel()` → `CancelledError` raise ที่ `await` ถัดไป → finally block ปิดงาน upsert ที่ค้างอยู่ก่อน exit
- structured logging: `logging.getLogger("perception")` emit JSON 1 บรรทัดต่อ tick (`ts, frame_age, n_detections, n_inserts, n_updates, n_skips, latency_ms_per_stage`)

### การเชื่อมกับ repo ปัจจุบัน

`services/perception.py` กับ `services/explore.py` implement version ที่ง่ายกว่าของเรื่องนี้อยู่แล้ว (track-and-promote, ใช้ threading) module ใหม่จะ *แทนที่* — สองไฟล์เก่าจะหายไปเมื่อ Phase 3 lands จนถึงตอนนั้นรันคู่ขนานไป agent ยังทำงานได้ปกติ

`agents/walkie_agent/tools.py::find_object_from_memory` และตัวเดียวกันใน `agents/vision_agent/tools.py` จะถูก rewire ไปเรียก `SceneStore.semantic_query` ไม่ต้องแก้ prompt ของ agent — tool surface เหมือนเดิมจากมุมมองของ agent

---

## Test plan — ตามที่ส่งใน Phase 3

### Dedup unit test (`tests/perception/test_dedup.py` — 11 tests)

แต่ละ test ป้อน `Detection` สังเคราะห์เข้า `SceneStore` ที่ backed ด้วย ChromaDB แยกต่อ test แล้ว assert decision เป้าหมาย: catch regression ของ merge threshold

1. Store ว่าง → INSERT
2. วัตถุเดียวกัน drift นิดเดียว (0.1 ม. embedding เดิม) → UPDATE, position เป็น running mean, `sightings == 2`
3. วัตถุที่หน้าตาเหมือนกันสองตัวห่างกัน 2 ม. → INSERT ทั้งคู่ (spatial gate แยก)
4. ตำแหน่งเดียวกัน class ต่าง → INSERT (ไม่ merge ข้าม class)
5. spatial เกือบชน (0.4 ม.) embedding เหมือนกันทุกประการ → UPDATE ผ่าน gate `EMB_SIM_HIGH`
6. spatial เกือบชน (0.4 ม.) embedding ตั้งฉาก → INSERT (fail ทั้งสอง gate)
7. tight-radius failsafe: ห่างกัน 0.15 ม. cosine ≈ 0.70 → UPDATE ผ่าน `EMB_SIM_LOW + TIGHT_RADIUS`
8. หาย + กลับมา (1 ชม. ต่อมา ตำแหน่งเดิม) → record เดียว `sightings == 2` `first_seen_ts` คงไว้
9. มีหลาย candidate ตัวที่ใกล้ที่สุดชนะ: ถ้า probe ใกล้ A กว่า B → A update B ไม่แตะ
10. env-var override (`SCENE_DEDUP_RADIUS_M=0.05`) พลิก UPDATE เดิมเป็น INSERT
11. `classify()` raise เมื่อ candidate ข้าม class (precondition guard)

### Query API unit test (`tests/perception/test_queries.py` — 12 tests)

seed store ใหม่ด้วย 10 entry ที่รู้ค่า (mix class, position, caption, timestamp) แล้ว:

1. `semantic_query("coffee mug")` จัดอันดับ mug-class ขึ้นบน
2. `semantic_query` พร้อม `within_radius_of` + `max_distance_m` ตัด mug ตัวไกลออก
3. `semantic_query` พร้อม `min_last_seen_ts` ตัด record ที่เห็นก่อน cutoff ออก
4. `visual_query` return ครบ `n_results` และมี `distance` field
5. `spatial_query` return ทุกอย่างใน ball โดยไม่สนใจ class
6. `spatial_query` พร้อม `class_name` กรองต่อ
7. `recency_query` return เฉพาะ entry `last_seen_ts > since_ts`
8. `diff` แบ่ง partition `appeared / refreshed / disappeared` ถูกต้อง
9. partition `refreshed` ของ `diff` surface การเห็นซ้ำ (ไม่ใช่ insert ใหม่)
10. `prune(ttl_sec=…)` ลบ record ที่คาดหวัง
11. `prune(max_records=N)` เก็บ N ตัวที่สดที่สุด
12. `upsert` หลัง `prune` ทำงานสะอาด (ไม่มี state ค้าง)

### Background loop integration test (`tests/perception/test_loop.py` — 8 tests)

ใช้ fake ใน `perception/mocks.py` ทั้งหมด ทำงาน end-to-end ผ่าน loop, pipeline, store

1. **Happy path** 5 tick ของฉาก static → 1 record `sightings ≥ 5` insert พอดี 1 ครั้ง
2. **Mid-scene change** วัตถุใหม่ปรากฏ tick ที่ 4 → 2 record `diff` จัดวัตถุใหม่เป็น `appeared`
3. **Graceful cancel** `task.cancel()` → shutdown ใน 200 ms ไม่มี record เขียนค้าง
4. **Detector error ไม่ฆ่า loop** raise เฉพาะ tick 3 → tick 1, 2, 4, 5 ยัง upsert; tick 3 มี field `error` ใน report
5. **Tick ช้าไม่กองทับ** captioner delay 50 ms vs interval 5 ms → inter-tick gap วัดได้ ≥ delay (sequential)
6. **[Addendum] Stare 200 tick** มอง mug ตัวเดียว 200 tick → 1 record `sightings ≥ 200` insert พอดี 1 ครั้ง (กัน regression "DB โตไม่หยุด")
7. **[Addendum] วัตถุถูกคนย้าย** bbox เดิมแต่ position world-frame ต่าง (> τ_pos) → 2 record คนละตัว ตำแหน่ง original ไม่ถูกดึงไปทาง B
8. **[Addendum] Long-run patrol** 320 tick วนผ่าน 4 จุดที่ FOV ทับซ้อนกัน → 4 record (ตัวต่อวัตถุที่ต่างกัน) insert พอดี 4 ครั้ง `sightings` ต่อวัตถุ ≈ N_TICKS/2

### Phase 1 smoke test (`tests/perception/test_smoke_*.py` — 11 tests)

pin contract ของ external API ทุกตัวที่เราใช้:

- `test_smoke_object_detection.py` — `ObjectDetectionClient.detect()` parse YOLO payload, raise เมื่อ server error
- `test_smoke_image_caption.py` — `ImageCaptionClient.caption()` / `caption_batch()` unwrap envelope
- `test_smoke_pose_estimation.py` — `PoseEstimationClient.estimate()` return `PersonPose` ที่มี keypoint COCO 17 จุด
- `test_smoke_image_embed.py` — client ชั่วคราว mirror shape ของ `/image-embed/*` (route ที่ disable อยู่) เพื่อ pin contract ตั้งแต่ตอนนี้
- `test_smoke_bboxes_to_positions.py` — request-reply ของ walkie-sdk `Tools.bboxes_to_positions` ทำงาน timeout ถูกต้อง return `[x,y,z]` ตรงลำดับ input

smoke test ทุกตัว mock network/transport boundary และรันเสร็จใน <1 วินาที

### Mock layer (`perception/mocks.py`)

- `FakeCamera(frames)` — วน PIL frame ทุก `capture_pil()`
- `FakeDetector(scripted, raise_on_idx=…)` — detection แบบ scripted ต่อ tick injected exception ก็ได้
- `FakeCaptioner(captions, delay=0)` — fixed text หรือ map ต่อ prompt; delay กำหนดได้เพื่อ test sequential tick
- `FakeEmbedder(dim, override_text, override_image)` — embedding deterministic ผ่าน SHA-256; override hook ให้ test คุม cosine ได้แน่นอน
- `FakePositionLifter(scripted, default, timeout_after=…)` — bbox → (x,y,z) lookup; return `None` หลัง N call ได้
- `FakeDetectedObject`, `make_tiny_image(seed)` — fixture สนับสนุน

---

## คำถามที่ยังเปิดอยู่ / ต้องการจากคุณ

1. **การตัดสินใจเรื่อง CLIP endpoint** เปิด `/image-embed/*` บน AI server (แนะนำ) หรือ embed ที่ฝั่ง agent? ดู "Capability ที่ขาด" ด้านบน **สถานะ Phase 3**: โค้ดรับ `Embedder` Protocol ใดๆ ก็ได้ `FakeEmbedder` มาให้สำหรับ test แล้ว **สถานะ Phase 3.1**: `client.ImageEmbedClient` + `perception.RemoteCLIPEmbedder` ลงไปแล้วใน branch `feat/perception-clip-client` **ยัง block ที่ฝั่ง server** — ต้อง uncomment `walkie-ai-server/api/__init__.py:16` (`app.register_blueprint(image_embed.bp)`) แล้ว redeploy AI server เมื่อเสร็จแล้วฝั่งเราไม่ต้องแก้โค้ดอะไรเพิ่ม สร้าง `RemoteCLIPEmbedder(walkieAI.image_embed)` แล้วส่งเข้า loop ได้เลย
2. **Position frame** `bboxes_to_positions` คืนตำแหน่งใน frame ที่ YOLO-3D node ต้นทาง publish ตอนนี้สมมติว่า `map` ควรยืนยันกับคนคุม ROS graph ของหุ่น ไม่อย่างนั้น design ต้องเพิ่มขั้นตอน `tf` lookup **สถานะ Phase 3**: เก็บ `position_frame: "map"` ใน metadata ไว้ตรวจ drift ภายหลังได้
3. **การเก็บ frame** default คือ "เก็บ JPEG ต่อ INSERT 1 ใบลง `frames/`" — โอเค หรืออยากเก็บทุก tick (history ละเอียดขึ้น) หรือไม่เก็บเลย (disk เล็กลง)? **สถานะ Phase 3**: implement เป็น "เก็บตอน INSERT เท่านั้น" คุมผ่าน `SceneStore(frames_dir=...)` และ argument `archive_source_frame` ของ loop
4. **Threading model** design นี้ใช้ `asyncio` end-to-end `PerceptionService` / `ExploreService` ของเดิมใช้ `threading.Thread` agent เองเป็น sync-with-async-tool-grouping จะทำ loop เป็น async แต่ต้องมี `run_perception_in_thread()` adapter สำหรับ `main.py` จนกว่าจะ migrate ทั้งหมด โอเคไหม? **สถานะ Phase 3**: loop ขับด้วย `asyncio.Task` adapter เข้า `main.py` เก็บไว้เป็น commit ติดตาม

---

## Phase 3 — สรุปการ implement

Module: `perception/` (package ใหม่)

| ไฟล์ | บรรทัด | หน้าที่ |
|---|---|---|
| `types.py` | ~160 | Frozen dataclass + Protocol ไม่มี logic |
| `dedup.py` | ~135 | pure-function `classify(new, candidates) → DedupDecision` + ฟังก์ชันดึง threshold ที่ override ผ่าน env var ได้ + helper สำหรับ merge position/confidence |
| `store.py` | ~400 | `SceneStore` — wrapper ของ ChromaDB `upsert / find_nearby / semantic_query / visual_query / spatial_query / recency_query / diff / prune / get_by_id / clear` |
| `pipeline.py` | ~140 | `process_frame(frame, …)` — detect → ยก 3D → caption → embed คืน `(list[Detection], latency_ms)` |
| `loop.py` | ~160 | `run_scene_perception(...)` — async loop cancel ได้ tick ทำงาน sequential, structured logging, isolate error |
| `mocks.py` | ~200 | `FakeCamera / FakeDetector / FakeCaptioner / FakeEmbedder / FakePositionLifter / FakeDetectedObject / make_tiny_image` |
| `__init__.py` | ~50 | public re-export |

### Test ที่ส่งมอบ (รวม 42 รายการ ~20 วินาทีบน cache อุ่น)

| ไฟล์ | จำนวน test | ครอบคลุม |
|---|---|---|
| `test_dedup.py` | 11 | ทุก branch ของ decision (store ว่าง, drift merge, ห่างกัน split, ข้าม class split, HIGH gate, fail ทั้งสอง gate, TIGHT failsafe, หาย/กลับ, multi-candidate closest-wins, env-var override, precondition ข้าม class) |
| `test_queries.py` | 12 | ทุก read path (semantic / visual / spatial / recency / diff) พร้อมการ combine filter + prune ตาม TTL/max-records + upsert หลัง prune สะอาด |
| `test_loop.py` | 8 | Happy path, mid-scene change, graceful cancel, error recovery, sequential tick, **[Addendum]** 200-tick stare, **[Addendum]** วัตถุถูกคนย้าย, **[Addendum]** long-run patrol FOV ทับซ้อน (320 tick → 4 record) |
| `test_smoke_*.py` | 11 | Phase 1 smoke สำหรับ `client/*` + contract `bboxes_to_positions` ของ walkie-sdk |

### Follow-up ที่ทราบ (อยู่นอกขอบเขต branch นี้)

- Wire loop เข้า `main.py` (sync adapter + แทนที่ `services/perception.py` กับ `services/explore.py` แบบ threading)
- เพิ่ม `client/image_embed.py` เมื่อ server เปิด `/image-embed/*` smoke test `test_smoke_image_embed.py` pin contract ไว้แล้ว
- Rewire `agents/walkie_agent/tools.py::find_object_from_memory` และ tool เดียวกันของ vision agent ให้ใช้ `SceneStore.semantic_query` tool surface เหมือนเดิมจากมุม agent
- Pillow 14 deprecation: `mocks.py` ใช้ `Image.Image.getdata` (warning เฉยๆ; เปลี่ยนเป็น `get_flattened_data` ก่อน Pillow 14 ออกในปี 2027)
