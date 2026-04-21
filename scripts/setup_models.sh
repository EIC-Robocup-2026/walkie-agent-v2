#!/bin/bash

# setup_models.sh: ดาวน์โหลดโมเดลสำหรับ walkie-vision (RTX 5090 Ready)
# ตำแหน่งจัดเก็บ: walkie-master/data/models/

MODEL_DIR="data/models"
mkdir -p "$MODEL_DIR"

echo "🚀 กำลังเริ่มต้นดาวน์โหลดโมเดลสำหรับ walkie-vision..."

# 1. SAM 2.1 Large (Segment Anything 2.1)
# อ้างอิงจากแผน Image Encoder Lab ที่ใช้ SAM 2.1 Large
echo "📥 กำลังดาวน์โหลด SAM 2.1 Large..."
if [ ! -f "$MODEL_DIR/sam2.1_h14.pt" ]; then
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072424/sam2.1_h14.pt -O "$MODEL_DIR/sam2.1_h14.pt"
else
    echo "✅ SAM 2.1 Large มีอยู่ในระบบแล้ว"
fi

# แก้ไข URL ให้ถูกต้อง
echo "📥 กำลังดาวน์โหลด SAM 2.1 Config..."
mkdir -p "data/models/configs/sam2.1"
wget https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1/sam2.1_h14.yaml -O "data/models/configs/sam2.1/sam2.1_h14.yaml"

# 2. YOLO11x-cls (Classification)
# อ้างอิงจากแผนที่ใช้ YOLO11x (Extra Large) สำหรับ precise object naming
echo "📥 กำลังดาวน์โหลด YOLO11x-cls..."
if [ ! -f "$MODEL_DIR/yolo11x-cls.pt" ]; then
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-cls.pt -O "$MODEL_DIR/yolo11x-cls.pt"
else
    echo "✅ YOLO11x-cls มีอยู่ในระบบแล้ว"
fi

# 3. ArcFace (InsightFace - buffalo_l)
# อ้างอิงจากแผนการใช้ ArcFace สำหรับ Face Embedding
echo "📥 กำลังดาวน์โหลด ArcFace (buffalo_l)..."
if [ ! -d "$MODEL_DIR/insightface/models/buffalo_l" ]; then
    mkdir -p "$MODEL_DIR/insightface/models"
    wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -O "$MODEL_DIR/buffalo_l.zip"
    unzip "$MODEL_DIR/buffalo_l.zip" -d "$MODEL_DIR/insightface/models/buffalo_l"
    rm "$MODEL_DIR/buffalo_l.zip"
else
    echo "✅ ArcFace buffalo_l มีอยู่ในระบบแล้ว"
fi

echo "✨ ดาวน์โหลดโมเดลทั้งหมดเรียบร้อยแล้วที่: $MODEL_DIR"
