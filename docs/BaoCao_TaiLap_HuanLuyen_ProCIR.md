# Báo cáo nhanh: tái lập huấn luyện ProCIR ở quy mô thu gọn

Võ Thị Bích Phượng, ngày 10/10/2026

Kính gửi cô Châu,

Em đã huấn luyện lại ProCIR trên cả ba bộ dữ liệu của FashionMV ở quy mô thu gọn theo đề cương (25% dữ liệu, LoRA) và đánh giá trên toàn bộ tập validation. Ba kết quả chính:

1. **MT+Align cao hơn single-turn 1,2 điểm R@5 trung bình (72,5 so với 71,3)**, với cùng hình mẫu như Bảng 3 của bài báo: có lợi trên DeepFashion và FashionGen, gần như không đổi trên Fashion200K.
2. **Quy trình đánh giá của em khớp với bài báo trên DeepFashion và FashionGen:** checkpoint do tác giả công bố đạt R@5 = 89,3 và 75,8 trên bộ ảnh của em, so với 89,2 và 75,0 trong bài báo.
3. **Ảnh Fashion200K của em không tương đương với ảnh của tác giả:** cùng checkpoint đó chỉ đạt 67,7 thay vì 77,6. Trên bộ này, kết quả của em phải được so với 67,7.

## 1. Em đã làm gì

- **Dữ liệu:** có đủ ảnh cho 100% triplet của cả ba bộ (trước đây thiếu FashionGen và chỉ có ảnh DeepFashion 224×224). Tổng 188.015 triplet train, 32.718 triplet val.
- **Mã huấn luyện:** tác giả chưa phát hành, nên em cài đặt lại theo bài báo (GradCache, hội thoại hai lượt, hàm mất mát căn chỉnh), có bộ kiểm thử đi kèm.
- **Hai lần chạy trên cùng dữ liệu:** MT+Align và single-turn. Mỗi lần huấn luyện trên 41.235 triplet (25%, 644 bước, batch 64, 1 epoch), chọn checkpoint trên tập dev (5% sản phẩm), rồi đánh giá trên **toàn bộ** tập val.
- **Ba mốc tham chiếu trên cùng bộ ảnh:** checkpoint của tác giả (chấm bằng `evaluate.py` gốc), FashionCLIP và CLIP zero-shot.
- **Khác bài báo:** 25% dữ liệu thay vì 100%; LoRA (r = 16) thay vì tinh chỉnh toàn bộ; 3 ảnh ở 336² khi huấn luyện thay vì tối đa 5 ảnh ở 512²; không có bước tiền huấn luyện SFT. Vì vậy hàng so sánh đúng trong Bảng 3 là nhóm "khởi tạo Pretrained".

## 2. Kết quả

Giao thức đánh giá của bài báo (tối đa 5 ảnh, 512², giữ sản phẩm nguồn trong gallery), toàn bộ tập val. Giá trị là R@5 / R@10 (%).

| | DeepFashion | Fashion200K | FashionGen | TB R@5 |
|---|---|---|---|---|
| **Số liệu bài báo** | | | | |
| Single-turn, Pretrained | 77,0 / 87,8 | 61,4 / 73,6 | 59,7 / 72,5 | 66,0 |
| MT+Align, Pretrained | 79,5 / 89,0 | 61,8 / 73,8 | 62,7 / 74,9 | 68,0 |
| ProCIR đầy đủ (MT+Align, SFT) | 89,2 / 94,9 | 77,6 / 86,6 | 75,0 / 85,3 | 80,6 |
| **Đo trên bộ ảnh của em** | | | | |
| Checkpoint ProCIR của tác giả | 89,26 / 95,10 | 67,71 / 76,97 | 75,82 / 86,05 | 77,60 |
| **Em, MT+Align** (25%, LoRA) | 80,47 / 90,79 | 66,22 / 77,34 | 70,72 / 81,76 | 72,47 |
| **Em, single-turn** (25%, LoRA) | 78,41 / 89,51 | 66,18 / 77,48 | 69,19 / 80,67 | 71,26 |
| FashionCLIP, hợp nhất muộn (α = 0,25) | 59,83 / 71,59 | 30,09 / 38,27 | 39,01 / 50,45 | 42,98 |
| CLIP ViT-B/32, hợp nhất muộn (α = 0,5) | 27,49 / 41,79 | 10,43 / 16,14 | 14,17 / 22,36 | 17,36 |
| **MT+Align trừ single-turn, R@5** | | | | |
| Bài báo | +2,5 | +0,4 | +3,0 | +2,0 |
| Em | +2,06 | +0,04 | +1,53 | +1,21 |

R@1 (DeepFashion / Fashion200K / FashionGen): checkpoint của tác giả 51,04 / 30,92 / 39,59; MT+Align của em 29,36 / 22,36 / 29,39; single-turn của em 26,41 / 23,11 / 27,76.

## 3. Cách em đọc các con số

- **Xu hướng MT+Align tốt hơn single-turn lặp lại được**, ở cả giao thức của bài báo lẫn một giao thức thứ hai (3 ảnh, 336²) mà hai lần chạy được chấm bằng cùng một công cụ (+3,36 / +0,04 / +1,39 điểm R@5). Chi phí là mỗi bước huấn luyện chậm gấp khoảng 1,8 lần.
- **So với checkpoint của tác giả trên cùng bộ ảnh**, mô hình MT+Align của em thấp hơn 8,8 điểm R@5 trên DeepFashion, 1,5 điểm trên Fashion200K và 5,1 điểm trên FashionGen. Khoảng cách ở R@1 lớn hơn nhiều (29,4 so với 51,0 trên DeepFashion).
- **So với hàng "MT+Align, Pretrained" của bài báo**, mô hình của em tương đương trên DeepFashion và cao hơn 8 điểm R@5 trên FashionGen, dù dùng ít dữ liệu hơn và LoRA. Em chưa có giải thích được kiểm chứng cho điều này và mới có một seed, nên em chưa coi đây là kết luận.
- **Thứ tự CLIP < FashionCLIP < ProCIR** của báo cáo trước vẫn giữ nguyên trên cả ba gallery đầy đủ.

## 4. Những điểm cần thận trọng

- **Mới có một seed.** Sai số chuẩn của R@5 khoảng 0,3 đến 0,6 điểm, nên chênh lệch 1,5 đến 2 điểm trên DeepFashion và FashionGen là đáng kể ở mức vừa phải; +0,04 trên Fashion200K coi như bằng nhau.
- **Hai lần chạy được chấm ở hai bước khác nhau.** Tiêu chí trên dev (trung bình R@5) chọn bước 400 cho MT+Align và bước 644 cho single-turn. Gallery dev chỉ vài trăm sản phẩm nên R@5 đã ở mức 93 đến 94% và khó phân biệt các checkpoint. Em sẽ chấm lại MT+Align ở bước cuối để so với single-turn ở cùng bước, và đổi tiêu chí sang R@1 cho các lần chạy sau.
- **Hai công cụ chấm ở giao thức của bài báo.** MT+Align chấm bằng `evaluate.py` gốc; single-turn chấm bằng công cụ của em vì mã gốc chỉ hỗ trợ truy vấn hai lượt. Giao thức thứ hai nêu ở trên là phép đối chứng.
- **Ảnh Fashion200K.** Em dùng bản cắt theo hộp phát hiện của bản phát hành chính thức; kết quả ở Mục 2 cho thấy đây không phải ảnh tác giả dùng. Em sẽ chấm lại checkpoint của tác giả trên ảnh gốc chưa cắt để kiểm chứng.

## 5. Chi phí đo được

Máy chủ em đang dùng cấp một lát MIG `1g.35gb` của GPU H200 (khoảng 1/7 năng lực tính toán), dùng chung nên chỉ còn khoảng 11 GiB bộ nhớ trống.

| | MT+Align | Single-turn |
|---|---|---|
| Thời gian mỗi bước (trung vị) | 40,8 giây | 22,7 giây |
| Huấn luyện 644 bước | khoảng 7,3 giờ, chưa kể đánh giá trên dev | 5 giờ 17 phút, kể cả đánh giá trên dev |

Đánh giá toàn bộ tập val bằng `evaluate.py` gốc mất khoảng 1 giờ. Con số huấn luyện phù hợp với khoảng 7 đến 13 GPU-giờ mỗi epoch mà đề cương ước lượng. Huấn luyện trên 100% dữ liệu sẽ mất khoảng 30 giờ cho MT+Align trên lát GPU này.

## 6. Việc tiếp theo

1. Chấm lại lần chạy MT+Align ở bước cuối (644), và chấm checkpoint của tác giả trên ảnh Fashion200K chưa cắt (mỗi việc khoảng 1 giờ GPU; ảnh chưa cắt cần tải thêm lên máy chủ).
2. **Mốc go/no-go G5 của đề cương:** em đã kiểm tra toàn bộ annotation của FashionMV. Dữ liệu công bố **không kèm nhóm ứng viên bị loại** (triplet chỉ có sản phẩm nguồn, đích, bộ dữ liệu, `views_involved` và văn bản chỉnh sửa). Theo phương án dự phòng trong đề cương, em sẽ tự xây kho mẫu âm khó, bắt đầu bằng cách khai thác từ embedding của mô hình vừa huấn luyện.
3. Gán nhãn góc nhìn: DeepFashion có sẵn tên góc trong tên tệp và mỗi triplet có trường `views_involved`; Fashion200K và FashionGen cần gán bằng CLIP zero-shot như đề cương.
4. Thí nghiệm chẩn đoán VHR của baseline (Tháng 3).

## 7. Em xin ý kiến cô

1. **Số seed cho baseline:** em nên chạy thêm seed cho hai cấu hình này ngay bây giờ (mỗi lần chạy khoảng 10 giờ kể cả đánh giá), hay để dành ngân sách và chỉ chạy 3 seed cho mô hình cuối như đề cương?
2. **Quy mô dữ liệu:** giữ 25% cho các thí nghiệm tiếp theo, hay nên có ít nhất một lần chạy 100% (khoảng 30 giờ) làm mốc?
3. **Ảnh Fashion200K:** nếu ảnh chưa cắt cho kết quả khớp bài báo, em nên chuyển toàn bộ thí nghiệm sang ảnh chưa cắt (phải huấn luyện lại), hay giữ ảnh hiện tại và so sánh nội bộ với mốc 67,7?
4. **Pilot LoRA so với tinh chỉnh toàn bộ:** với 11 GiB bộ nhớ trống, tinh chỉnh toàn bộ không chạy được trên lát GPU hiện tại. Em đề xuất ghi nhận đây là giới hạn hạ tầng và giữ LoRA, trừ khi cô thấy cần xin thêm GPU cho phép thử này.

Chi tiết đầy đủ (thiết lập, các bảng R@1/R@5/R@10, diễn biến trên tập dev, các giới hạn) nằm ở Mục "Tái lập huấn luyện ProCIR ở quy mô thu gọn", Chương 6 của báo cáo; tệp kết quả gốc nằm ở `results/procir_train/` trong kho mã.

Em cảm ơn cô.
