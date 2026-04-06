Một “thiết kế CNN net tốt” thường không có 1 công thức tuyệt đối kiểu “đúng padding/đúng flatten thì chắc accuracy cao”, mà là tổng hoà của 2 thứ: (1) **thiên kiến (inductive bias) đúng cho dữ liệu** và (2) **khả năng train/stabilize tốt để hiện thực hóa inductive bias đó**. Vì vậy, đúng như bạn nói: “train xong mới biết” là có cơ sở, nhưng bạn vẫn có thể dự đoán khá nhiều dựa trên các đúc kết nghiên cứu.

## 1) Thế nào là “tốt”: nhìn vào cái gì ngoài accuracy?
Trong thực tế/điểm số, người ta thường coi mô hình “tốt” nếu đồng thời ổn ở vài mặt:

1. **Tổng quát hóa**: train accuracy/loss không chỉ đẹp, mà **val/test cũng tăng và gap không nở quá**.
2. **Ổn định huấn luyện**: loss giảm đều, không văng/không dao động quá mạnh; gradient không bị “tắt”.
3. **Hiệu năng**: không chỉ accuracy cao mà còn cân bằng **compute/params/latency** (đặc biệt nếu bài có ràng buộc).
4. **Hiệu quả theo dữ liệu & độ khó**: CIFAR-10 dễ hơn CIFAR-100; cùng một thiết kế có thể cho kết quả khác nhau—“tốt” là “phù hợp bài toán”.

Nên hiểu “loss/accuracy sau training” là thước đo cuối cùng, còn kiến trúc là thứ bạn chọn để làm cho thước đo đó có cơ hội tốt.

## 2) Có quy chuẩn/đúc kết tồn tại không?
Có, nhưng thường ở dạng **pattern + best practice**, không phải “luật cứng”. Các nghiên cứu nhiều năm tạo ra những “khung thiết kế” mà đa số trường hợp dùng lên là train được và có baseline mạnh:

- **Backbone mạnh có lịch sử** (ResNet/WRN/WideResNet, EfficientNet, MobileNet, v.v.) thường “an toàn” hơn thiết kế mới hoàn toàn.
- **Khối residual (ResNet/WRN)** giúp train sâu dễ hơn (giải quyết vấn đề suy giảm gradient).
- **BatchNorm + activation đúng chỗ** (thường pre-activation trong WRN) giúp convergence ổn.
- **Attention như SE** thường có chỗ “đúng tinh thần” (channel attention; đặt sau một số conv để tái cân kênh).
- **Depthwise separable conv** thường xuất hiện trong kiến trúc tối ưu compute (MobileNet-style), và nếu bạn đưa nó vào sai chỗ hoặc thiếu pointwise/thiếu đủ nonlinearity thì chất lượng có thể giảm.

Những “kết luận từ nhiều nghiên cứu” vì vậy tồn tại, nhưng bạn cần nhớ: nó không đảm bảo 100% vì còn phụ thuộc dữ liệu, augmentation, optimizer, LR schedule, regularization, cách chọn kích thước đầu vào…

## 3) Các “quy chuẩn” thường gặp khi thiết kế CNN (đặc biệt hợp CIFAR 32×32)
Dưới đây là các nguyên tắc rất hay gặp (dùng như checklist khi bạn thiết kế/đổi block):

### (a) Khâu đầu (stem) và padding
- Dùng **conv 3×3 padding=1** để giữ kích thước không gian (với kernel lẻ như 3×3).
- Nếu stride=2 thì kích thước giảm một nửa; với CIFAR hay xuống dần kiểu: 32 → 16 → 8 (thường là đúng “kịch bản” cho receptive field).

### (b) Các block giữa (body)
- Với bài toán phân loại ảnh, backbone tối ưu thường là:
  - **residual blocks** (basic/bottleneck)
  - hoặc variants như **WideResNet**: tăng “widen factor” để cải thiện biểu diễn.
- Nếu có dropout: thường chỉ thêm ở mức hợp lý để regularize, tránh làm block mất khả năng học feature.

### (c) Attention (SE/những attention tương tự)
- SE thường là dạng: *global avg pool → MLP (reduction) → sigmoid → nhân kênh*.
- Thực hành phổ biến: gắn attention **sau các conv trong residual block** (đúng logic: attention điều chỉnh feature đã được trích xuất), rồi mới cộng skip connection.

### (d) Khâu cuối: pooling/flatten
- “Chuẩn” trong nhiều kiến trúc hiện đại là **Global Average Pooling** (AdaptiveAvgPool to 1×1) rồi đưa vào FC.
- **Flatten** theo kiểu “bê nguyên feature map rồi flatten” không hẳn sai, nhưng thường kém hiệu quả hơn global pooling vì global pooling giảm tham số/giảm overfit và phù hợp với classification.

### (e) Nếu dùng depthwise
- Depthwise conv thường đi kèm **pointwise (1×1 conv)** để trộn kênh; đây là đặc trưng MobileNet-style.
- Depthwise thường giúp giảm compute, nhưng bạn vẫn phải đảm bảo mạng vẫn đủ khả năng trộn thông tin giữa các kênh (nếu không, accuracy có thể tụt).

## 4) Vậy “điều bạn cần học” có phải chỉ là block nào dùng strict thế nào không?
Không. Thứ bạn cần học (để làm bài cuối kiểu Final) thường là:

1. **Biết chọn kiến trúc hợp lý để train được tốt** (baseline mạnh + hiểu tại sao).
2. **Biết các quy tắc thiết kế phổ biến** để tránh thiết kế “lệch” gây khó train hoặc giảm generalization.
3. **Biết dùng thực nghiệm để xác nhận**: theo dõi loss/acc theo epoch, xem overfit/underfit, kiểm tra best checkpoint, hiểu trade-off train vs val/test.
4. **Biết giải thích oral exam**: bạn chọn block/attention/regularization vì mục tiêu gì, nó liên quan thế nào đến cơ chế học (ví dụ SE làm gì, residual giúp gì, depthwise tối ưu compute thế nào).

Kết luận: các đúc kết tồn tại và rất đáng học, nhưng “chuẩn” trong bài cuối không chỉ là kiến trúc đúng sách; mà là kiến trúc đó + training protocol + giải thích hợp lý tạo ra kết quả test/val tốt và train ổn.

Nếu bạn muốn, bạn gửi cho mình biết bạn định thiết kế cho Final theo hướng nào (WRN/ResNet hay PDPNet/depthwise/MobileNet-like). Mình sẽ giúp bạn biến các “quy chuẩn” trên thành một checklist cụ thể: stem–downsample–block–attention–head, và những điểm nào oral exam thường hỏi.

---

## Thuật ngữ — định nghĩa ngắn (theo nội dung trên + `keyword.md`)

### A. Chỉ số khi train / thống kê theo epoch

| Thuật ngữ | Định nghĩa ngắn |
|-----------|-----------------|
| **Loss** | Hàm mất mát: đo “sai bao nhiêu” giữa dự đoán và nhãn thật (ví dụ CrossEntropy). Giảm loss thường đi cùng học tốt hơn, nhưng cần xem cả val/test để tránh **chỉ khớp tập train**. |
| **Chỉ khớp tập train / ghi nhớ train** | Mô hình học “thuộc lòng” nhãn và nhiễu trên tập train nên **train acc / train loss rất tốt**, nhưng **val/test kém** — tức **overfitting**. Ngược lại, nếu train và val/test cùng cải thiện hợp lý thì thường tốt hơn. |
| **Accuracy** | Tỷ lệ dự đoán đúng (thường %). “Accuracy value” là giá trị số của độ chính xác tại một bước (một batch, một epoch, hoặc trên cả tập). |
| **Metric trên train** | Loss/accuracy (hoặc F1, …) tính **chỉ trên các mẫu đang dùng để train** (thường trung bình qua batch rồi gộp cả epoch). Phản ánh mức “khớp” dữ liệu huấn luyện; **không thay thế** metric trên val/test khi đánh giá tổng quát. |
| **Metric trên val / test** | Loss/accuracy tính trên tập **không dùng để cập nhật gradient** trong bước đó (val thường mỗi epoch chạy một lần ở chế độ `eval`; test tương tự). Đây là thước đo “ra đề mới” gần hơn. |
| **Epoch** | Một **vòng lặp** trong đó mô hình đã xem qua **toàn bộ** tập train **một lần** theo nghĩa: **mỗi mẫu trong tập train được đưa vào forward (và backward) đúng một lần** trong epoch đó (sau khi chia **batch**). Không có nghĩa “mỗi ảnh chỉ xử lý một phép tính duy nhất”: thực tế train theo **lô (batch)** — ví dụ 50.000 ảnh, batch 128 → khoảng \(50000/128\) bước cập nhật mỗi epoch; cộng hết các batch thì đủ 50.000 mẫu một vòng. |
| **Batch (lô)** | Một nhóm mẫu đưa vào GPU cùng lúc; mỗi batch tính loss rồi **một lần** backward (cập nhật trọng số). **Batch size** = số mẫu mỗi batch. **Iteration / step** = một lần cập nhật như vậy. Trong một epoch: số step ≈ \(\lceil N_{\text{train}} / \text{batch size} \rceil\) (có thể có batch cuối nhỏ hơn). |
| **Siêu tham số (hyperparameters)** | Các thiết lập **không** học trực tiếp bằng gradient trên tập train, mà do người chọn hoặc tìm thử: **learning rate**, **batch size**, **số epoch**, **weight decay**, kiểu **optimizer**, kiểu **scheduler**, mức **dropout**, độ mạnh **augmentation**, đôi khi cả **kiến trúc** (số block, số kênh) khi bạn coi là cố định trong một thí nghiệm. Thường tinh chỉnh dựa trên **validation** (hoặc cross-validation), không tinh trên test để tránh “rò” thông tin. |
| **Train / validation / test** | **Train**: dữ liệu để **tính gradient** và cập nhật trọng số. **Validation**: tách ra để **chọn siêu tham số**, **early stopping**, **chọn best checkpoint** — **không** dùng gradient từ val để học trọng số (chỉ **đo** metric). **Test**: tập **đánh giá cuối cùng** sau khi đã khóa mọi lựa chọn; lý tưởng nhất chỉ chạm **một lần** khi báo cáo kết quả thật. |
| **Val và test: có dùng chung / gọi lẫn được không?** | **Về vai trò**: khác nhau — val phục vụ **điều chỉnh trong quá trình phát triển mô hình**; test phục vụ **ước lượng không thiên vị** sau cùng. **Trong thực hành (CIFAR, nhiều notebook)**: đôi khi chỉ có **một** tập “held-out” và tác giả gọi là `val` hoặc `test` — **cùng một tập** thì **tên khác nhau nhưng dữ liệu giống nhau**; khi đó coi như “một vai trò”. **Không nên** dùng chung một tập vừa để chọn epoch tốt nhất vừa tự nhận đó là “test thuần túy” rồi báo cáo như chưa nhìn — tốt nhất ghi rõ: “chúng em dùng split X làm validation để chọn checkpoint; test chính thức là Y” (nếu có). |
| **Val/test (cách gọi chung)** | Tập **không dùng để train** (gradient), dùng để đo khả năng tổng quát; cần biết code đang gọi tập nào. |
| **Gap (khoảng cách train–val/test)** | Chênh lệch giữa metric trên train và trên val/test (ví dụ train acc cao, test acc thấp → gap lớn, thường gợi ý overfitting). |
| **Các cột trong bảng log train** | Thường gồm: epoch, `train_loss`, `train_acc`, `val_loss`, `val_acc` (hoặc `test_*` tương tự), đôi khi learning rate. `train_*` = metric trên tập train; `val_*` / `test_*` = metric trên tập held-out. Mỗi dòng = tóm tắt sau một epoch. |

**Gợi ý đọc nhanh — epoch vs batch:** Một **epoch** = “đủ một vòng qua **hết** mẫu train”; mỗi vòng đó thực hiện bằng nhiều **bước** (mỗi bước = một **batch**). Vậy “mỗi hình một lần trong epoch” đúng nghĩa **mỗi mẫu xuất hiện đúng một lần trong phần train của epoch** (trừ khi code có lỗi hoặc oversampling), không phải “chỉ một phép nhân cho cả dataset”.

### B. Ổn định huấn luyện & tối ưu

| Thuật ngữ | Định nghĩa ngắn |
|-----------|-----------------|
| **Ổn định huấn luyện** | Loss không nhảy loạn, không NaN; trọng số cập nhật mượt; mô hình học được xu hướng chung (giảm loss theo thời gian) thay vì dao động vô nghĩa. |
| **Dao động mạnh** | Loss hoặc accuracy thay đổi đột ngột giữa các bước/epoch (do LR quá lớn, batch nhỏ, augmentation quá mạnh, v.v.). |
| **Gradient không bị “tắt”** | Gradient vẫn đủ lớn để lan truyền ngược (không bị **vanishing** quá mức làm layer sâu hầu như không học). Ngược lại **exploding gradient** là gradient quá lớn làm cập nhật vỡ. Residual, BatchNorm, khởi tạo tốt giúp giảm vanishing. |
| **Convergence (hội tụ)** | Quá trình huấn luyện đi đến trạng thái loss/metric ổn định (ít cải thiện thêm dù train thêm). |
| **Optimizer** | Thuật toán cập nhật trọng số (SGD, Adam, …) từ gradient. |
| **LR schedule** | Cách thay đổi learning rate theo epoch/bước (CosineAnnealing, StepLR, …) để hội tụ tốt hơn ở cuối train. |
| **Augmentation** | Biến đổi ảnh input khi train (crop, flip, AutoAugment, Cutout, …) để tăng đa dạng, giảm overfit. |
| **Regularization** | Kỹ thuật giảm overfitting: weight decay, dropout, label smoothing, augmentation, … |

### C. Hiệu năng & tài nguyên

| Thuật ngữ | Định nghĩa ngắn |
|-----------|-----------------|
| **Compute** | Lượng tính toán (phép nhân–cộng, FLOPs) — thường liên quan thời gian train/suy luận. |
| **Params (parameters)** | Số lượng trọng số cần học; nhiều params → mô hình “lớn”, dễ overfit nếu dữ liệu ít. |
| **Latency** | Độ trễ khi chạy (ví dụ ms/ảnh); quan trọng khi triển khai thực tế. |
| **Cân bằng compute / params / latency** | Không chỉ nhắm accuracy: chọn kiến trúc vừa đủ nhanh/nhẹ so với yêu cầu (edge device vs server). |

### D. Tư duy thiết kế

| Thuật ngữ | Định nghĩa ngắn |
|-----------|-----------------|
| **Inductive bias (thiên kiến)** | Giả định cấu trúc mà kiến trúc “ép” lên dữ liệu (ví dụ locality qua conv, translation equivariance). Giúc học hiệu quả hơn khi giả định trùng bài toán. |
| **Pattern** | Mô hình lặp lại trong thiết kế (ví dụ cứ N block lại downsample một lần). |
| **Best practice** | Cách làm được cộng đồng/nghiên cứu thấy thường ổn (không phải định luật, nhưng giảm thử sai). |
| **Baseline mạnh** | Một kiến trúc + recipe train đã được báo cáo kết quả tốt trên dataset tương tự; dùng làm mốc so sánh. |
| **Khó train** | Loss không giảm, hội tụ chậm, hoặc cần tuning rất kỹ LR/initialization — thường gặp khi mạng quá sâu/rộng hoặc thiết kế lệch. |
| **Training protocol** | Toàn bộ “công thức” train: optimizer, LR, batch size, augmentation, số epoch, cách chọn checkpoint, … |

### E. Kiến trúc CNN — khối & luồng dữ liệu

| Thuật ngữ | Định nghĩa ngắn |
|-----------|-----------------|
| **Backbone** | Phần CNN trích đặc trưng (nhiều conv/block), trước khi gắn **head** phân loại. |
| **Stem (khâu đầu)** | Vài layer đầu (thường conv) nhận ảnh gốc và tạo feature ban đầu; đôi khi giảm kích thước không gian. |
| **Body** | Phần giữa: chồng nhiều block, tăng độ sâu / kênh đặc trưng. |
| **Residual (phần dư / kết nối tắt)** | Đường **skip connection** cộng input vào output của khối conv: \(y = F(x) + x\), giúp gradient đi thẳng hơn, train mạng sâu dễ hơn. |
| **Residual blocks (basic / bottleneck)** | **Basic**: hai conv 3×3 (hoặc tương đương) trong một block. **Bottleneck**: 1×1 giảm kênh → 3×3 → 1×1 tăng lại, ít tính hơn ở tầng giữa sâu. |
| **Bottleneck** | Kiểu block “eo” kênh ở giữa để giảm chi phí tính toán. |
| **Variants** | Biến thể của một họ mạng (WRN = ResNet “rộng hơn”; SE-ResNet = thêm SE, …). |
| **Widen factor** | Hệ số nhân số kênh trong WideResNet: mạng “rộng” hơn thay vì chỉ sâu hơn. |
| **Receptive field** | Vùng trên ảnh đầu vào mà một neuron ở layer sau “nhìn thấy” (phụ thuộc kernel, stride, chồng layer). Càng sâu thường receptive field càng lớn. |

### F. Chuẩn hóa & kích hoạt

| Thuật ngữ | Định nghĩa ngắn |
|-----------|-----------------|
| **BatchNorm (Batch Normalization)** | Chuẩn hóa theo batch theo từng kênh để train ổn định, cho phép LR cao hơn; có tham số scale/shift học được. |
| **Activation (hàm kích hoạt)** | ReLU, GELU, … — thêm phi tuyến để mạng học được hàm phức tạp hơn tuyến tính. |
| **Pre-activation** | Thứ tự BN → ReLU → Conv (trước khi conv) trong một block; một kiểu thiết kế ResNet/WRN giúp train tốt. |
| **Nonlinearity** | Tính “phi tuyến” do activation; không có thì chồng nhiều tầng tuyến tính vẫn tương đương một tầng. |

### G. Convolution & attention

| Thuật ngữ | Định nghĩa ngắn |
|-----------|-----------------|
| **Padding** | Thêm pixel (thường 0) quanh ảnh/feature map để sau conv kích thước ra theo ý muốn (ví dụ padding=1 với kernel 3×3, stride 1 → giữ kích thước). |
| **Stride** | Bước nhảy của kernel; stride=2 thường làm giảm một nửa chiều rộng/cao. |
| **Depthwise convolution** | Mỗi kênh input được lọc riêng bởi một kernel (không trộn kênh trong bước này). |
| **Pointwise convolution** | Conv 1×1: trộn thông tin **giữa các kênh** (đổi số kênh). |
| **Depthwise separable conv** | Gồm depthwise rồi pointwise; ít phép tính hơn conv đầy đủ cùng kích thước (kiểu MobileNet). |
| **Khả năng trộn thông tin giữa các kênh** | Sau depthwise cần pointwise (hoặc conv thường) để các kênh “nói chuyện” với nhau; thiếu bước trộn kênh thì biểu diễn yếu. |
| **Attention SE (Squeeze-and-Excitation)** | Module: **squeeze** (global average pool theo không gian) → hai lớp FC nhỏ → **excitation** (sigmoid) → nhân từng kênh feature map. Là dạng **channel attention** (nhấn mạnh kênh quan trọng). |
| **Channel attention** | Trọng số học được cho từng kênh (khác spatial attention nhấn vùng không gian). |
| **Đưa đúng chỗ / sai chỗ** | Module (SE, depthwise, …) phụ thuộc vị trí trong mạng: đặt hợp lý thì cải thiện; đặt sai có thể phá luồng feature hoặc tăng khó train. |

### H. Regularization trong mạng

| Thuật ngữ | Định nghĩa ngắn |
|-----------|-----------------|
| **Dropout** | Tắt ngẫu nhiên một phần neuron/kết nối khi train để giảm phụ thuộc đồng phục, giảm overfit. |
| **Regularize (chuẩn hóa / điều chuẩn hóa)** | Làm mô hình không khớp noise trên train quá mức (xem dropout, weight decay, …). |
| **Mất khả năng học feature** | Mạng hoặc layer không còn cải thiện biểu diễn hữu ích (do vanishing gradient, dropout quá mạnh, kiến trúc tắc, …). |

### I. Đầu ra phân loại

| Thuật ngữ | Định nghĩa ngắn |
|-----------|-----------------|
| **Global Average Pooling (GAP)** | Trung bình mỗi kênh trên toàn bản đồ không gian → vector độ dài = số kênh (thường trước FC). Giảm tham số so với flatten toàn bộ feature map. |
| **AdaptiveAvgPool** | GAP với kích thước đầu ra cố định (ví dụ 1×1) bất kể input spatial size. |
| **Flatten** | Duỗi tensor nhiều chiều thành vector một chiều (thường trước FC). |
| **FC (Fully Connected)** | Lớp tuyến tính: mỗi output nối với mọi input; cuối mạng thường map sang số lớp. |
| **Classification (phân loại)** | Bài toán gán nhãn rời rạc (CIFAR-10: 10 lớp; CIFAR-100: 100 lớp). |

### J. Hiện tượng & đánh giá mô hình

| Thuật ngữ | Định nghĩa ngắn |
|-----------|-----------------|
| **Overfit** | Khớp train quá tốt nhưng val/test kém (học cả nhiễu); gap train–test lớn. |
| **Underfit** | Cả train và val/test đều kém — mô hình quá đơn giản hoặc train chưa đủ. |
| **Giảm generalization (tổng quát hóa kém)** | Khả năng áp dụng lên dữ liệu mới kém; thường liên quan overfit hoặc train sai protocol. |
| **Giảm tham số / giảm overfit** | Dùng ít params hơn, hoặc regularization, để mô hình không học quá chi tiết noise. |
| **Checkpoint** | File lưu trạng thái mô hình (và đôi khi optimizer) tại một epoch. |
| **Best checkpoint** | Checkpoint tại epoch có val/test metric tốt nhất (theo tiêu chí bạn chọn), dùng để báo cáo hoặc deploy. |
| **Trade-off train vs val/test** | Cân nhắc: tăng capacity giúp train acc nhưng có thể hại val; augmentation làm train acc thấp hơn nhưng val tốt hơn — v.v. |

### K. Tên mạng (tham chiếu nhanh)

| Thuật ngữ | Định nghĩa ngắn |
|-----------|-----------------|
| **ResNet** | Mạng residual nhiều tầng, giải quyết khó train mạng sâu. |
| **WRN / WideResNet** | ResNet “mở rộng theo chiều kênh” (widen factor), thường mạnh trên CIFAR. |
| **EfficientNet** | Họ mạng scale đồng thời depth/width/resolution. |
| **MobileNet** | Kiến trúc nhẹ, dùng depthwise separable conv nhiều. |

### L. Khái niệm liên quan (có thể gặp thêm)

| Thuật ngữ | Định nghĩa ngắn |
|-----------|-----------------|
| **Head** | Phần cuối (thường pooling + FC) gắn sau backbone để ra logits từng lớp. |
| **Logits** | Đầu ra trước softmax; softmax biến thành xác suất. |
| **Softmax** | Chuẩn hóa vector thành phân phối xác suất trên các lớp. |
| **CrossEntropyLoss** | Loss phổ biến cho phân loại đa lớp kết hợp softmax + negative log-likelihood. |
| **Label smoothing** | Làm “mềm” nhãn one-hot (không cho 100% một lớp) để giảm overconfidence, thường giúp generalization. |
| **Weight decay** | Phạt trọng số lớn (thường trong optimizer), tương tự L2 regularization. |
| **FLOPs** | Số phép dấu phẩy động ước lượng độ phức tạp tính toán của mô hình. |