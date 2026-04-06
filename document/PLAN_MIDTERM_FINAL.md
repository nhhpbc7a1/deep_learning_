# Kế hoạch chi tiết: giữa kỳ (M1 + dataset D) và cuối kỳ (M2 + CIFAR)

Tài liệu khớp `document/requirement.txt`. **Pipeline chính (M1/M2):** `train_student_cnn.py` + `models/student_cnn.py` (StudentCNN), log `train_log.txt`, vẽ `plot_train_log.py`, xác minh `verify_student_cnn.py`. (Có thể dùng thêm `train_pdpnet.py` / PDPNet nếu cần so sánh — không bắt buộc.)

---

## 1. Tổng quan luồng làm việc

| Giai đoạn | Việc cần làm |
|-----------|----------------|
| **Giữa kỳ** | Thu thập dataset D (≥ 5 lớp), chia `train/` và `val/`. Huấn luyện **StudentCNN (M1)** **hai lần** trên D: **224×224** và **32×32**, mỗi lần ~**100 epoch**. Log, đồ thị, checkpoint, báo cáo. |
| **Cuối kỳ** | **M2 = cùng StudentCNN** (đề cho phép). Train **CIFAR-10** và **CIFAR-100**, mỗi tập ~**200 epoch**; vấn đáp thiết kế / siêu tham số. |

---

## 2. Dataset D cho giữa kỳ (ImageFolder)

### 2.1 Cấu trúc thư mục (bắt buộc với code hiện tại)

Đặt gốc dữ liệu ví dụ `data_custom/` (tên tùy bạn, trỏ bằng `-r`):

```text
data_custom/
  train/
    class_a/   *.jpg hoặc *.png
    class_b/
    ...
  val/
    class_a/
    class_b/
    ...
```

- **Cùng tên thư mục lớp** giữa `train` và `val` (cùng thứ tự lớp).
- **Tối thiểu 5 lớp** (đề bài).
- Nên có **đủ ảnh mỗi lớp** (tránh lớp chỉ vài ảnh); ghi trong báo cáo: nguồn, số ảnh/lớp, cách chia train/val (ví dụ 80/20 ngẫu nhiên).

### 2.2 Nguồn gợi ý (tải về rồi tự gán nhãn thư mục / chia train-val)

1. **Kaggle Datasets** — tìm “multiclass image classification”, “5 classes”, “10 classes”: ví dụ bộ ảnh đồ vật, thực phẩm, phương tiện. Tải file ZIP, giải nén, gom ảnh vào cấu trúc `train/class_name/...`.
2. **Caltech-101 / 256** — nhiều lớp; có thể **chọn 5–10 lớp** cố định, copy vào `data_custom`.
3. **Oxford 102 Flowers** — nếu muốn bài “hoa” (nhiều lớp); có thể gộp hoặc chọn subset.
4. **Google Open Images** / **Roboflow Universe** — lọc theo lớp, export ảnh theo thư mục.
5. **Tự chụp / tự thu thập** — 5 loại đồ vật trong phòng; ghi rõ trong báo cáo (điểm “chất lượng dataset”).

**Lưu ý bản quyền và trích dẫn** nguồn trong báo cáo.

---

## 3. Lệnh train (sau khi có `data_custom/`)

Thư mục gốc project (nơi có `train_student_cnn.py`).

### 3.1 Giữa kỳ — dataset D, 224×224 và 32×32

```text
# ~100 epoch, 224x224
python train_student_cnn.py -d imagefolder -r ./data_custom --image-size 224 -e 100 -b 32 -g 0 --run-tag D_224

# ~100 epoch, 32x32
python train_student_cnn.py -d imagefolder -r ./data_custom --image-size 32 -e 100 -b 32 -g 0 --run-tag D_32
```

- Log/checkpoint: `checkpoints/StudentCNN_imagefolder/D_224/` và `.../D_32/`. Có thể đổi tên `train_log.txt` khi nộp báo cáo.

### 3.2 Cuối kỳ — CIFAR-10 và CIFAR-100 (~200 epoch)

Gợi ý **batch 128** và **schedule** phù hợp 200 epoch (ví dụ giảm LR sớm hơn PDPNet):

```text
python train_student_cnn.py -d cifar10 -r ./data_cifar --download -e 200 -b 128 -g 0 --image-size 32 -s 60 120 160

python train_student_cnn.py -d cifar100 -r ./data_cifar --download -e 200 -b 128 -g 0 --image-size 32 -s 60 120 160
```

(`-r` thư mục CIFAR; `--download` tải nếu chưa có. `weight_decay` mặc định **5e-4** — phổ biến với CNN + BN trên CIFAR.)

---

## 4. Đồ thị và bảng số

Sau khi có file log (copy từ `train_log.txt` hoặc đổi tên như `FIE02.txt`):

```text
python plot_train_log.py --log checkpoints/StudentCNN_imagefolder/D_224/train_log.txt --out checkpoints/StudentCNN_imagefolder/D_224/plots.png --csv checkpoints/StudentCNN_imagefolder/D_224/metrics.csv
```

- Chèn **PNG** và **bảng best accuracy** (từ log hoặc CSV) vào Word/PDF.
- Đề yêu cầu **training và testing** — trong code, metric trên tập held-out được gọi là `val`; bạn có thể gọi là **test** trong báo cáo nếu `val` chính là tập đánh giá cố định (không dùng để học trọng số).

---

## 5. Xác minh checkpoint (yêu cầu đề)

```text
python verify_student_cnn.py -c checkpoints/StudentCNN_imagefolder/D_224/checkpoint_best_XX.XX.pth -d imagefolder -r ./data_custom -g 0 --image-size 224
```

Đổi `-d`, `-r`, `--image-size` cho đúng lần train. Kết quả loss/accuracy phải **khớp hợp lý** với log epoch tốt.

---

## 6. Thiết kế StudentCNN (M1/M2) — giải thích báo cáo / vấn đáp

**Ý tưởng:** CNN “kinh điển” — chồng **Conv → BatchNorm → ReLU**, giảm kích thước không gian bằng **stride 2** (và MaxPool ở nhánh 224), cuối cùng **Global Average Pooling** rồi **FC**; **Dropout** nhẹ trước FC để giảm overfit. Không dùng attention hay depthwise — dễ giải thích, vẫn đủ mạnh nếu train đúng (SGD + MultiStepLR, label smoothing khi train, weight decay 5e-4 trên CIFAR).

**Hai nhánh (trong `models/student_cnn.py`):**

1. **`image_size <= 32` (CIFAR / midterm 32×32):**  
   - Hai conv 3×3 giữ độ phân giải 32, rồi các cặp conv với **stride 2** lần lượt: 32→16→8→4.  
   - Kênh tăng dần 64 → 128 → 256 → 512 (kiểu VGG gọn).  
   - **AdaptiveAvgPool2d(1)** → vector 512 chiều.

2. **`image_size > 32` (224×224):**  
   - **Stem** Conv 7×7 stride 2 + **MaxPool** (giống nhiều mạng ImageNet nhẹ).  
   - Tiếp tục các khối conv stride 2 và một conv giữ kích thước cho đến khi feature map ~7×7, rồi GAP → 512 → FC.

**Train (`train_student_cnn.py`):** SGD + momentum, **MultiStepLR**, train loss = **Label Smoothing CE**, val = **CrossEntropy** chuẩn — giống pipeline bạn đã quen; `weight_decay` mặc định **5e-4** (thường hợp CIFAR + BN).

**Gợi ý tối ưu accuracy (vẫn gọn):**  
- CIFAR: batch 128 nếu GPU đủ; điều chỉnh `-s` (milestones) theo tổng epoch.  
- Dataset tự thu: cân lớp, đủ ảnh; giữ augmentation trong code (crop/flip/jitter tùy 32 vs 224).

---

## 7. Checklist nộp bài

**Giữa kỳ:** Dataset D mô tả rõ; hai lần train (224, 32); log đủ epoch; đồ thị; best val/test; `verify_student_cnn.py`; bảng so sánh hai độ phân giải.

**Cuối kỳ:** Train CIFAR-10 và CIFAR-100; log + đồ thị + verify; **vấn đáp**: Conv+BN, GAP, dropout; LR schedule; augmentation; weight decay.

---

## 8. File liên quan StudentCNN

| File | Mục đích |
|------|----------|
| `models/student_cnn.py` | Kiến trúc StudentCNN |
| `train_student_cnn.py` | Huấn luyện (CIFAR / ImageFolder / dogs) |
| `verify_student_cnn.py` | Xác minh checkpoint StudentCNN |
| `check_student_cnn.py` | In model + `torchsummary` (GPU) |
| `plot_train_log.py` | Vẽ đồ thị từ `train_log.txt` |
| `requirements.txt` | Dependencies |
| `data_custom/README_DATASET.md` | Cấu trúc thư mục ảnh |

Có thêm `train_pdpnet.py` / `verify_checkpoint.py` (PDPNet) nếu cần đối chiếu.
