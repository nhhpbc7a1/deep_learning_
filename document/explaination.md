# Giải thích StudentCNN cho người mới (`models/student_cnn.py`)

File này chỉ để **bạn đọc cho dễ hiểu**: từng **khâu** của mạng chọn **cái gì**, đặt **chỗ nào**, **vì sao** — không gắn với bất kỳ tài liệu nội bộ nào là “chuẩn tham khảo”. Code nằm ở `models/student_cnn.py`.

---

## Mạng làm việc theo thứ tự nào?

Ảnh vào → **trích đặc trưng** (nhiều lớp conv) → **gom lại một vector** → **dropout** → **lớp tuyến tính** ra số lớp.

Bạn có thể nghĩ: “lớp conv tìm cạnh / họa tiết → sau đó gom ý toàn ảnh → rồi đoán nhãn”.

---

## Khâu 1 — Đầu vào: vì sao có **hai kiểu** mạng (32 vs 224)?

- Ảnh **nhỏ** (32×32, kiểu CIFAR hoặc midterm 32): nếu đầu mạng dùng conv **rất lớn + stride 2** như ảnh ImageNet, ảnh **bị tụt kích thước quá nhanh**, mất chi tiết. Nên ta dùng **conv 3×3, stride 1** vài lần trước, **giữ** gần nguyên 32×32 một chút rồi mới **giảm dần** xuống 16 → 8 → 4.
- Ảnh **lớn** (224×224): ngược lại, cần **hạ độ phân giải sớm** cho nhẹ và hợp lý, nên đầu dùng **conv 7×7 stride 2** và **MaxPool** giống nhiều mạng ảnh lớn quen thuộc.

**Tóm lại:** cùng một “ý tưởng” (chồng conv), nhưng **đoạn đầu** chỉnh theo **cỡ ảnh** — không phải hai mạng khác hẳn, chỉ là **nhánh khác nhau** trong code.

---

## Khâu 2 — Thân mạng (giữa): chọn **conv chồng nhau** + **BN + ReLU**

Sau stem, phần giữa làm việc lặp kiểu:

**Conv → BatchNorm → ReLU**

- **Conv:** trượt filter trên ảnh / bản đồ đặc trưng, học pattern cục bộ.
- **BatchNorm:** coi như “cân lại” số liệu trong batch cho dễ train, loss đỡ nhảy loạn — rất hay gặp trong CNN hiện đại.
- **ReLU:** bật/tắt (phi tuyến), không ReLU thì chồng nhiều tầng tuyến tính cũng như một tầng.

**Giảm kích thước bản đồ:** dùng conv **stride 2** (đôi khi kèm conv stride 1 ngay sau để “chỉnh” lại đặc trưng). Mỗi lần hạ cỡ, **số kênh tăng** (64 → 128 → 256 → 512): thường làm vậy để **bù** cho việc ô ảnh lớn hơn (mỗi pixel trên bản đồ “nhìn” vùng rộng hơn trên ảnh gốc).

**Vì sao không thêm ResNet (nhánh tắt + cộng)?**  
ResNet giúp mạng **rất sâu** train ổn hơn. Ở đây mạng **không quá sâu**, ưu tiên **dễ đọc code và dễ giải thích** — chỉ chồng conv thường là đủ cho bài tập.

**Vì sao không thêm SE / attention?**  
Là phần “xịn” thêm; có thể cải thiện nhưng **không bắt buộc**. Bỏ đi cho đỡ rối khi bạn mới học.

**Vì sao không dropout giữa các conv?**  
Dropout giữa conv đôi khi làm mạng **khó học feature** nếu bật quá mạnh. Ở đây chỉ **dropout ngay trước lớp cuối** (xem khâu 3).

---

## Khâu 3 — Cuối: **pooling trung bình toàn cục** → **dropout** → **FC**

- **`AdaptiveAvgPool2d(1)`:** mỗi kênh còn lại lấy **trung bình cả bản đồ** → được **một vector cố định** (512 số). Cách này **ít tham số** hơn là flatten hết 4×4×512 rồi nối FC khổng lồ, thường **đỡ overfit** hơn cho phân loại.
- **Dropout:** lúc train, tắt ngẫu nhiên một phần số trong vector, bắt mạnh không “ôm khư khư” vài chiều — chủ yếu tác động **trước** lớp phân loại.
- **`Linear(512, n_class)`:** nhân ma trận ra **đúng số lớp** (5 hoa, 10 CIFAR-10, 100 CIFAR-100, …).

Thứ tự trong `forward`: **features** → **flatten** → **dropout** → **fc**.

---

## Khâu phụ — `bias=False` trên conv và khởi tạo trọng số

- Conv **không bias** vì sau conv đã có **BatchNorm** (BN có tham số dịch/thang riêng) — tránh trùng vai trò.
- Conv khởi tạo kiểu **Kaiming** (phù hợp ReLU); BN và Linear khởi tạo kiểu **an toàn** để bước đầu train không nổ số.

---

## Bảng nhìn nhanh

| Khâu | Trong code (gợi ý) | Chọn gì, vì sao ngắn gọn |
|------|---------------------|---------------------------|
| Ảnh nhỏ 32 | `_make_small_image_backbone` | Đầu 3×3 s1, sau đó mới stride 2 dần — khỏi “nuốt” hết ảnh |
| Ảnh lớn 224 | `_make_large_image_backbone` | 7×7 + MaxPool đầu — hạ cỡ hợp lý cho ảnh lớn |
| Giữa | Cả hai nhánh | Conv–BN–ReLU, tăng kênh khi giảm kích thước |
| Cuối | `AdaptiveAvgPool2d` → dropout → `fc` | Gom vector gọn, dropout nhẹ, ra logits lớp |

---

## Đọc thêm (tùy chọn, khi cần trích báo cáo)

Đây chỉ là **nguồn bài báo gốc** cho từng ý lớn, không bắt buộc đọc hết để hiểu mạng:

- BatchNorm: https://arxiv.org/abs/1502.03167  
- Khởi tạo conv + ReLU (Kaiming): https://arxiv.org/abs/1502.01852  
- Chồng conv 3×3 (VGG): https://arxiv.org/abs/1409.1556  
- Global average pooling (ý tưởng gom kênh): https://arxiv.org/abs/1312.4400  
- Dropout: https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf  
- Residual (nếu sau này bạn nâng cấp): https://arxiv.org/abs/1512.03385  

---

**Một câu nhớ:** StudentCNN = **CNN chồng conv có BN**, **đầu chỉnh theo 32 hay 224**, **cuối pool trung bình + dropout + FC** — đủ làm bài, dễ kể lại từng khâu khi thầy hỏi.
