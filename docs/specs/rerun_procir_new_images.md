# Spec: chạy lại đánh giá ProCIR với ảnh DeepFashion gốc (256×256 và high-res)

Ngày soạn: 2026-10-07. Trạng thái: bản nháp, chưa thực hiện.

## 1. Mục tiêu

Chạy lại `evaluate.py` của FashionMV với checkpoint `yuandaxia/ProCIR` trên **ảnh DeepFashion In-shop lấy từ nguồn gốc**, thay cho bản HF mirror `Marqo/deepfashion-inshop` (224×224) đã dùng. Sau đó so sánh:

| Lần chạy | Nguồn ảnh | Trạng thái |
|---|---|---|
| A (đã có) | HF mirror, 224×224 | R@1/5/10 = 33,58 / 74,42 / 85,25 |
| B | `img.zip` In-shop, 256×256 | cần chạy |
| C | `img_highres` In-shop (mật khẩu đã nhận) | cần chạy |

Câu hỏi cần trả lời: độ phân giải ảnh làm kết quả thay đổi bao nhiêu, và kết quả có gần số của paper hơn không.

## 2. Phạm vi

**Trong phạm vi**
- DeepFashion, toàn bộ 5.188 triplet của `val_triplets.jsonl`.
- Dùng `evaluate.py` nguyên bản, không sửa.

**Ngoài phạm vi (nêu rõ trong báo cáo)**
- FashionGen: không tải được nguồn dữ liệu.
- Fashion200K: chưa có ảnh gốc (`women.tar.gz` chưa tải), xem mục 8.
- Huấn luyện lại ProCIR: mã huấn luyện chưa được phát hành.

## 3. Đầu vào

| Thành phần | Vị trí | Ghi chú |
|---|---|---|
| Triplet val | `data/FashionMV_hf/data/data/val_triplets.jsonl` | 5.188 dòng `deepfashion` |
| Checkpoint | `data/FashionMV_hf/model/` | `model.safetensors` 1,6 GB |
| Ảnh low-res | `img.zip` nằm trong `data/raw/DeepFashion-...-004.zip` | 52.712 ảnh, 256×256, không mã hóa |
| Ảnh high-res | `data/raw/img_highres-005.zip` | 52.713 mục, mã hóa ZipCrypto, mật khẩu trong thư của tác giả |

Mật khẩu **không** được ghi vào file, mã nguồn hay log trong repo. Truyền qua biến môi trường hoặc nhập tay khi chạy.

## 4. Cấu trúc ảnh mà code yêu cầu

Đọc từ `procir/datasets.py`:

- `source_id` / `target_id` trong triplet, ví dụ `WOMEN/Tees_Tanks/id_00006581/12`, được ghép thành **một thư mục**: `<image_root>/deepfashion/WOMEN/Tees_Tanks/id_00006581/12/`.
- Trong thư mục đó, code lấy tối đa 5 ảnh (`MAX_VIEWS = 5`), sắp theo tên file. Với In-shop đó là `12_1_front`, `12_2_side`, `12_3_back`, `12_4_full`.
- Trường `views_involved` trong triplet **không được dùng** khi đánh giá. Mọi ảnh trong thư mục đều được đưa vào.
- Thư mục thiếu hoặc rỗng bị **bỏ qua im lặng**, không báo lỗi.

Trong zip gốc ảnh nằm phẳng: `img/WOMEN/Tees_Tanks/id_00006581/12_1_front.jpg`. Bước chuẩn bị phải gom thành `.../id_00006581/12/12_1_front.jpg`.

Quy tắc chuyển: với file `<thư_mục_sản_phẩm>/<NN>_<k>_<góc>.jpg`, thư mục đích là `<thư_mục_sản_phẩm>/<NN>/`.

## 5. Các bước

1. **Kiểm tra mật khẩu.** Giải nén thử một ảnh từ `img_highres-005.zip`, mở được bằng PIL thì đúng.
2. **Viết script chọn lọc** `src/data/extract_deepfashion_official.py`, nhận tham số `--zip`, `--out`, và `--password` (tùy chọn):
   - đọc `val_triplets.jsonl`, lấy tập ID cần dùng (source + target, dataset `deepfashion`);
   - chỉ giải nén ảnh thuộc các ID đó, xếp theo cấu trúc mục 4;
   - in thống kê: số ID cần, số ID tìm thấy, số ảnh ghi, danh sách ID thiếu.
3. **Chạy B** (low-res) rồi **C** (high-res), mỗi lần ra một thư mục ảnh riêng: `data/images_official256/deepfashion`, `data/images_official_hr/deepfashion`.
4. **Đóng gói** thư mục ảnh đã lọc thành một zip nhỏ, đưa lên nền tảng chạy GPU (xem mục 7).
5. **Chạy đánh giá:** `python evaluate.py --model_path <...> --image_root <...> --data_dir <...> --datasets deepfashion --batch_size <...>`.
6. **Lưu kết quả** vào `results/procir/` và cập nhật `results/procir/README.md`: bảng chạy, nguồn ảnh, thời gian, kết quả.

## 6. Tiêu chí hoàn thành

- Script chọn lọc báo `Ids covered: N / N` với N bằng số ID cần. Nếu thiếu ID, liệt kê và giải thích.
- Dòng log `[CIRValDataset] Loaded 5188 triplets` xuất hiện. Con số thấp hơn 5.188 nghĩa là còn ảnh thiếu.
- Dòng `Gallery: ... products` của mỗi lần chạy xấp xỉ 2.791 (số của lần A). Lệch nhiều cần điều tra trước khi dùng kết quả.
- Kết quả B và C có `eval_results.json` và log đầy đủ lưu trong repo.
- Báo cáo có bảng so sánh A/B/C và số trong paper (đối chiếu lại với bảng trong `docs/paper/fashionmv.pdf` khi viết, không lấy từ trí nhớ).

## 7. Nền tảng chạy GPU

Ba lần chạy Kaggle trước đều mất 8–12 giờ trên T4, vì `causal-conv1d` không build được và `flash-linear-attention` chưa cài, nên các lớp linear-attention chạy bằng bản PyTorch chậm. Cần thử trước khi chạy dài:

1. Cài `flash-linear-attention` và `causal-conv1d` (thử wheel dựng sẵn) trong một kernel thử ngắn, rồi đo tốc độ trên vài trăm triplet. Chưa có số liệu nên chưa biết nhanh hơn bao nhiêu.
2. Chọn Kaggle hay Colab sau khi có số đo. Mặc định ở lại Kaggle vì pipeline đã chạy được.
3. Dù chọn nền tảng nào, chỉ upload zip ảnh đã lọc (vài trăm MB đến vài GB), không upload 76 GB dữ liệu gốc.

## 8. Rủi ro và câu hỏi mở

| Vấn đề | Tác động | Hướng xử lý |
|---|---|---|
| Phiên Kaggle tối đa 12 giờ | Lần chạy dài có thể bị cắt | Sửa dependency trước (mục 7), đo thời gian bằng chạy thử nhỏ |
| Thư mục thiếu bị bỏ qua im lặng | Số query và gallery thấp hơn thực tế mà không có lỗi | Dùng tiêu chí mục 6 để phát hiện |
| Ảnh high-res lớn hơn nhiều | Chậm hơn, tốn bộ nhớ GPU hơn khi encode | Giảm `--batch_size`, theo dõi bộ nhớ |
| Fashion200K chưa có ảnh gốc | Chỉ có số từ mẫu con và gallery nhỏ, không so trực tiếp với paper | Tải `women.tar.gz` từ link của `xthan/fashion-200k` nếu muốn chạy gallery đầy đủ. **Cần bạn quyết định** |
| Chưa kiểm tra cách phân giải ảnh trong paper | Không biết lệch số do ảnh hay do nguyên nhân khác | Đọc phần dữ liệu trong paper trước khi kết luận |
| Cách lấy mẫu F200K trước đây | Gallery được dựng từ các ID trong triplet được chọn (`ProductValDataset` đọc `val_triplets`, không phải `val_captions`), nên giảm triplet cũng làm gallery nhỏ lại và R@K tăng | Nếu chạy lại F200K có lấy mẫu, nên giữ gallery đầy đủ |

## 9. Điều tôi chưa kiểm chứng

- Chưa thử mật khẩu trên `img_highres-005.zip`.
- Chưa chạy script lọc nên chưa biết có bao nhiêu ID thiếu ảnh trong zip gốc.
- Chưa kiểm tra bản `img.zip` có khớp 100% với mirror HF (cùng ảnh gốc hay không).
- Chưa đo thời gian chạy với dependency đã sửa.
