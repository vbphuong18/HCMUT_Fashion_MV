# Tổng quan dự án — để nắm được "mình đang làm gì"

> Tài liệu này viết cho chính bạn (Phượng), không phải để gửi cô. Mọi con số lấy từ `README.md`, `results/`, và đề cương `docs/proposal/decuong_MultiViewCIR_v3.tex`.

---

## 1. Dự án trong 5 câu

1. **Bài toán (CIR):** đưa vào *một ảnh + một câu chỉnh sửa* ("giống cái này nhưng hở lưng") → hệ thống tìm ra sản phẩm phù hợp.
2. **Vấn đề của các nghiên cứu cũ:** chỉ dùng **1 ảnh** cho truy vấn. Thực tế một sản phẩm có nhiều ảnh (trước/sau/nghiêng). Chi tiết "hở lưng" chỉ nhìn thấy ở ảnh sau.
3. **Bài báo nền tảng (Yuan et al., 2026):** đặt ra bài toán *Multi-View CIR* (truy xuất ở mức **sản phẩm**, dùng **nhiều ảnh**), tạo bộ dữ liệu **FashionMV** và mô hình **ProCIR**.
4. **Thực tập 1 (TT1)** = hiểu, viết lại và **tái lập** bài báo đó + chạy vài baseline.
5. **Thực tập 2 (TT2)** = đề xuất cải tiến của riêng bạn (**MV-SHN**) trên nền ProCIR.

> Cô đã chốt: **Internship 1 chính là bài báo FashionMV**. Phần "đề xuất của em" (MV-SHN) thuộc **Internship 2**, tên đề tài: *"Truy xuất ảnh sản phẩm thời trang đa phương thức có điều kiện ở mức sản phẩm với dữ liệu đa góc nhìn"*.

---

## 2. Bản đồ tổng thể

```
                     ┌────────────────────────────────────────────┐
  BÀI BÁO FashionMV  │  Dữ liệu FashionMV  +  Mô hình ProCIR       │
  (người khác làm)   └───────────────┬────────────────────────────┘
                                     │
        ┌────────────────────────────┴────────────────────────────┐
        ▼                                                         ▼
 ┌───────────── THỰC TẬP 1 ─────────────┐      ┌───────────── THỰC TẬP 2 ─────────────┐
 │ Mục tiêu: HIỂU + TÁI LẬP bài báo     │      │ Mục tiêu: CẢI TIẾN bài báo (MV-SHN)  │
 │                                      │      │                                      │
 │ A. Viết lại bài báo thành báo cáo    │      │ 0. Chẩn đoán: ProCIR có tự chọn      │
 │ B. Chạy lại ProCIR (checkpoint tác   │      │    đúng góc nhìn không? (đo VHR)     │
 │    giả) → so với số trong bài báo    │      │ 1. View-Aware Selection (G1)         │
 │ C. Chạy 1–3 baseline (CLIP,          │      │ 2. Hard Negative Mining (G5)         │
 │    FashionCLIP) → so sánh            │      │ 3. Độ bền khi thiếu góc nhìn (G2)    │
 │                                      │      │ 4. Chuyển giao FashionIQ (G3)        │
 └──────────────────────────────────────┘      └──────────────────────────────────────┘
        (phần lớn ĐÃ LÀM)                              (CHƯA bắt đầu)
```

Nói gọn: **TT1 = "chứng minh em hiểu và chạy lại được cái người ta làm". TT2 = "em làm thêm cái mới".**

---

## 3. Đã làm được gì (TT1)

| Việc cô yêu cầu | Trạng thái | Bằng chứng trong repo |
|---|---|---|
| Viết lại bài báo thành báo cáo TT1 | ✅ Xong (7 chương, ~66 trang) | `report/TT1_report.pdf`, `report/chapter/ch1-7.tex` |
| Tái lập đề xuất của bài báo (ProCIR) | ⚠️ Xong một phần: chạy **đánh giá** bằng checkpoint tác giả, **không huấn luyện lại** (tác giả chưa công bố code train) | `results/procir/` |
| Chạy 1–3 mô hình cơ bản | ✅ Xong: CLIP và FashionCLIP | `results/baselines/`, `src/baselines/` |
| Tổng quan tài liệu | ✅ Có | `docs/VoThiBichPhuong2470570_literature_overview.pdf` |
| Đề cương TT1+2 | ✅ Đã gửi, cô đã chốt đề tài | `docs/proposal/` |

### Kết quả thu được (Recall@5 / Recall@10, %)

| Mô hình | DeepFashion | Fashion200K |
|---|---|---|
| CLIP zero-shot | 22.11 / 33.85 | 22.87 / 33.60 |
| FashionCLIP | 55.17 / 67.27 | 51.00 / 62.80 |
| **ProCIR (em chạy lại)** | **74.42 / 85.25** | **87.93 / 94.53** |
| ProCIR (bài báo công bố) | 89.2 / 94.9 | 77.6 / 86.6 |

**Cách đọc bảng này (đây là phần quan trọng nhất để trình bày):**

- **Kết luận chắc chắn:** CLIP < FashionCLIP < ProCIR ở cả hai bộ dữ liệu → ProCIR thật sự mạnh hơn, đúng như bài báo nói. Đây là kết quả *tái lập được*.
- **Số tuyệt đối không khớp bài báo, và lệch hai chiều ngược nhau:**
  - DeepFashion **thấp hơn** (74 vs 89): ảnh thay thế chỉ 224×224, mô hình nhận ít token thị giác hơn ~5 lần so với ảnh gốc.
  - Fashion200K **cao hơn** (88 vs 78): em chỉ chạy 1.500 triplet nên gallery chỉ 2.367 sản phẩm (bài báo: 10.720). Gallery nhỏ → dễ tìm đúng hơn → Recall cao.
- **Vì sao dùng ảnh thay thế / mẫu con:** ảnh gốc không tải tự động được; GPU miễn phí Kaggle giới hạn 12h/phiên; ProCIR chạy cực chậm trên T4 (DeepFashion mất ~11h).
- **FashionGen: không đánh giá được** vì trang tải chính thức đã đóng. Đây là hạn chế khách quan, đã ghi rõ.
- **Phát hiện phụ:** code đánh giá gốc giữ sản phẩm nguồn trong gallery → với baseline, sản phẩm nguồn xếp hạng 1 gần như luôn → R@1 ≈ 0. Bài báo không nói chi tiết này.

> Lưu ý: hai nguyên nhân sai lệch ở trên hiện là **giả thuyết hợp lý**, chưa được kiểm chứng bằng thực nghiệm. Đó chính là lý do kế hoạch tiếp theo có hai thí nghiệm đối chứng.

---

## 4. Còn thiếu gì / việc cần làm

### 4.1. Để chốt TT1 (ưu tiên cao, rẻ hơn TT2 rất nhiều)

| # | Việc | Mục đích | Khó khăn |
|---|---|---|---|
| 1 | Chạy đối chứng DeepFashion **độ phân giải cao** | Chứng minh nguyên nhân lệch 74 vs 89 là do độ phân giải | Cần `img_highres.zip` (phải email tác giả xin) |
| 2 | Chạy Fashion200K với **gallery đầy đủ** (10.720 sản phẩm) | Có con số so sánh được trực tiếp với 77.6 | Rất tốn GPU: bản đầy đủ từng bị dừng ở giới hạn 12h (mới xong 76% gallery) |
| 3 | Chỉnh lại **cách trình bày** báo cáo tiến độ (xem mục 5) | Cô hiểu em làm gì | Chỉ cần viết lại |
| 4 | Nếu cô muốn: chạy thêm 1 baseline nữa (vd. CLIP4Cir) | "1–3 mô hình" | Cần huấn luyện, tốn công |

### 4.2. TT2 (chưa bắt đầu) — theo đề cương

| Bước | Nội dung | Giải quyết khoảng trống |
|---|---|---|
| 0 | **Chẩn đoán:** đo xem ProCIR có "chú ý" đúng góc nhìn mà câu chỉnh sửa nhắc tới không (chỉ số VHR). Nếu VHR ≥ 0.8 → ProCIR đã tự chọn tốt, đổi hướng nhấn mạnh | G1 |
| 1 | **View-Aware Selection:** module nhỏ học trọng số cho từng góc nhìn theo câu chỉnh sửa | G1 |
| 2 | **Hard Negative Mining:** lấy các ứng viên bị loại khi dựng dữ liệu làm mẫu âm khó, có lọc mẫu âm giả | G5 |
| 3 | **Độ bền:** đánh giá khi chỉ còn 1/2/3 góc nhìn; thử view-dropout khi huấn luyện | G2 |
| 4 | **Chuyển giao:** đánh giá trên FashionIQ (dữ liệu do người gán nhãn) | G3 |

### 4.3. Rủi ro lớn cần biết ngay

1. **Code huấn luyện ProCIR chưa công bố** ("Coming Soon" theo ghi chú trước đó; nên kiểm tra lại repo chính thức). Đề cương TT2 giả định *fine-tune từ checkpoint Qwen3.5-0.8B gốc* và tự cài lại quy trình huấn luyện. Nếu không có code train, TT2 phải tự viết toàn bộ pipeline → đây là rủi ro số 1.
2. **Hạ tầng:** TT1 mới chỉ chạy đánh giá trên T4 miễn phí và đã rất chậm. TT2 cần huấn luyện (~490 GPU-giờ theo ngân sách đề cương trên 1 GPU 24GB). Kaggle miễn phí **không đủ**; cần có máy/GPU thực sự.
3. **Đề cương có vài mốc thời gian đã lệch** so với thực tế, vì cô đã định nghĩa lại TT1 = bài báo (đề cương cũ xếp "chẩn đoán", "go/no-go G5", "FashionIQ" vào TT1). Cần hỏi cô để cập nhật.

---

## 5. Vì sao báo cáo tiến độ lần 1 bị phản hồi tệ (suy luận của tôi)

Tôi chưa thấy phản hồi gốc của cô nên đây là phỏng đoán từ chính văn bản báo cáo:

- **Không có câu chuyện tổng thể.** Báo cáo đi thẳng vào "R@5/R@10 = 74,42/85,25" mà chưa nói *mình đang làm gì, để làm gì, nằm ở đâu trong TT1+TT2*.
- **Không có "đã xong / chưa xong / bước kế tiếp" theo yêu cầu của cô.** Cô giao 3 việc (viết lại bài báo, tái lập, chạy 1–3 baseline) nhưng báo cáo không bám vào 3 việc đó.
- **Quá nhiều chi tiết kỹ thuật** (token, gallery size, giao thức R@1) ở đầu, trong khi kết luận chính (ProCIR > FashionCLIP > CLIP, tái lập được về thứ hạng) bị chìm.
- **Kế hoạch tiếp theo chỉ có 2 gạch đầu dòng**, không nói gì về TT2.

### Mẫu cấu trúc nên dùng cho lần báo cáo sau (1 trang)

1. **Mục tiêu TT1** — 1 câu, nhắc lại 3 việc cô giao.
2. **Kết quả theo từng việc** — mỗi việc: ✅/⚠️/❌ + 1 câu.
3. **Bảng kết quả chính** (như bảng ở mục 3) + 2–3 câu kết luận.
4. **Hạn chế + nguyên nhân** — FashionGen, ảnh 224px, gallery nhỏ.
5. **Việc tiếp theo** — chia *chốt TT1* và *chuẩn bị TT2*.
6. **Câu hỏi cần cô quyết** (mục 6).

---

## 6. Câu hỏi nên hỏi cô (để khỏi mò)

1. Với TT1, cô cần **kết quả khớp bài báo**, hay chấp nhận "thứ hạng khớp + giải thích được vì sao lệch"?
2. Có cần hoàn tất 2 thí nghiệm đối chứng (ảnh độ phân giải cao, gallery đầy đủ) **trước khi chốt TT1** không?
3. Cô có thể hỗ trợ **GPU** (máy trường/lab) cho TT2 không? Kaggle miễn phí không đủ để huấn luyện.
4. Nếu code train ProCIR vẫn chưa có, cô đồng ý để em **tự cài đặt pipeline huấn luyện**, hay thu hẹp TT2?
5. Mốc nộp báo cáo TT1 cuối cùng là khi nào?

---

## 7. Thuật ngữ nhanh

| Từ | Nghĩa dễ hiểu |
|---|---|
| **CIR** | Tìm ảnh bằng "ảnh mẫu + câu chỉnh sửa" |
| **Multi-View** | Mỗi sản phẩm có nhiều ảnh (trước, sau, nghiêng…) |
| **Triplet** | 1 mẫu dữ liệu = (sản phẩm nguồn, câu chỉnh sửa, sản phẩm đích đúng) |
| **Gallery** | Tập tất cả sản phẩm để tìm kiếm; càng lớn càng khó |
| **Recall@K** | % truy vấn mà đáp án đúng nằm trong K kết quả đầu |
| **Baseline** | Mô hình đơn giản làm mốc so sánh |
| **Zero-shot** | Dùng mô hình có sẵn, không huấn luyện thêm |
| **Checkpoint** | File trọng số mô hình đã huấn luyện |
| **Hard negative** | Mẫu sai nhưng rất giống mẫu đúng, giúp mô hình học phân biệt tinh |
| **VHR** | Tỉ lệ mô hình chú ý đúng góc nhìn mà câu chỉnh sửa nhắc tới |
| **G1/G2/G3/G5** | Các "khoảng trống nghiên cứu" trong đề cương (chọn góc nhìn / độ bền / FashionIQ / mẫu âm khó) |

---

## 8. Bản đồ thư mục

| Thư mục | Chứa gì |
|---|---|
| `docs/proposal/` | Đề cương đã gửi cô |
| `report/` | Báo cáo TT1 (LaTeX → PDF) |
| `src/baselines/` | Code CLIP / FashionCLIP |
| `src/kaggle/` | Script chạy ProCIR trên Kaggle |
| `src/data/` | Script trích ảnh thay thế |
| `results/procir/` | Kết quả + log ProCIR |
| `results/baselines/` | Kết quả + log baseline |
