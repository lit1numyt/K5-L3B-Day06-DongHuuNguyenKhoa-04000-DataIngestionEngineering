# Báo cáo cá nhân — Day 6

## Dữ liệu và pipeline

- Chủ đề, nguồn/quyền, số trang và quy tắc tính trang: Corpus là *A Survey on Automated Software Vulnerability Detection Using Machine Learning and Deep Learning*, arXiv snapshot 2306.11673v1, từ `paper/2306.11673v1.pdf`. PDF có 37 trang vật lý. Tôi tính 30 trang nội dung hợp lệ, trang 1–30; loại các trang 31–37 vì chỉ chứa tài liệu tham khảo. Nguồn và snapshot được ghi trong [data/manifest.json](../../data/manifest.json). Quyền tái phân phối của bản PDF cần được xác minh trước khi công khai repo.
- Chuẩn hóa, lỗi và giới hạn: Dùng `pypdf` để trích xuất từng trang, chuẩn hóa Unicode/khoảng trắng, loại running header/footer lặp lại và giữ `document_id`, `source`, `page`, `section`. Không có trang rỗng trong 30 trang được chọn. Giới hạn chính là PDF có bảng, biểu đồ và ngắt dòng nên một số câu bị chia giữa các đoạn; OCR không được dùng.
- Embedding model/provider/version/số chiều, vector store/version, metric, lệnh chạy và chi phí: Model local `google/embeddinggemma-2` qua Sentence Transformers 6.3.0 và Transformers 5.23.0, 768 chiều. Document dùng prompt `Document`, query dùng prompt `QuestionAnswering`. Vector store là PostgreSQL 17.12 + pgvector 0.8.7, cosine similarity, lưu bền vững trong database `day6_lab`; truy vấn exact vector search. Chạy từ `starter/` bằng `conda run -n ml-arm python -m src.run`. Chi phí API là 0; thời gian toàn pipeline canonical run là 30.683 giây.
- Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn: Mỗi chunk có `document_id`, `source`, `page`; thêm `section` được kiểm chứng từ heading PDF và cấu trúc TeX. `section` giúp truy vết và lọc các câu hỏi giới hạn vào methodology hoặc một RQ cụ thể. Cấu hình C chỉ lọc khi câu hỏi cung cấp section filter.

## Cấu hình và benchmark

- A: fixed-size chunks, 700 ký tự, overlap 100.
- B: paragraph-aware chunks, tối đa 1000 ký tự, không filter.
- C: giống B về corpus, model, chunking và metadata; chỉ thêm filter `section` khi câu hỏi có filter. Vì vậy B–C thay đúng yếu tố filter.
- Benchmark có 20 câu hỏi được chốt trước khi chạy, gồm 15 dev và 5 test. Có 7 câu filter: 3 dev và 4 test. Mỗi câu có expected answer, gold page và evidence quote; evidence được kiểm tra tồn tại trong text đã chuẩn hóa. Bộ câu hỏi nằm tại [benchmark/questions.json](../../benchmark/questions.json).
- Tiêu chí chọn chiến lược được chốt trước khi xem test: ưu tiên Evidence Hit@3 trên dev, sau đó xem filter compliance và chi phí/thời gian. Theo tiêu chí này, B được chọn làm cấu hình mặc định vì đạt 12/15 Evidence Hit@3 trên dev. Test chỉ được dùng để báo cáo sau khi lựa chọn đã cố định.

## Kết quả

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian ingest | Chi phí |
| -------- | ----- | ----------: | -------------------: | ----------: | ----------------: | -------: |
| A        | dev   |        6/15 |                 8/15 |         179 |          12.377 s |        0 |
| A        | test  |         0/5 |                  2/5 |         179 |          12.377 s |        0 |
| B        | dev   |        5/15 |                12/15 |         118 |          14.080 s |        0 |
| B        | test  |         1/5 |                  2/5 |         118 |          14.080 s |        0 |
| C        | dev   |        3/15 |                10/15 |         118 |          0.068 s* |        0 |
| C        | test  |         2/5 |                  2/5 |         118 |          0.068 s* |        0 |

`*` C dùng lại embeddings của B; thời gian độc lập chỉ là ghi index/truy xuất.

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng | Filter tuân thủ | Grounding 0/1/2 và trung bình /2 |
| -------- | ---------------------------: | -----------------------: | ----------------: | ---------------------------------- |
| A        |                     1.0 (10) |                    15/15 |   1/4 test filter | 0/3/2; 1.4                         |
| B        |                     1.1 (10) |                    15/15 |   2/4 test filter | 0/4/1; 1.2                         |
| C        |                     1.1 (10) |                    15/15 |   4/4 test filter | 0/3/2; 1.4                         |

Coherence dùng cùng 10 vị trí nội dung đã chọn trước cho mỗi strategy. Metadata được kiểm tra trên 5 câu test × top-3 = 15 hit mỗi strategy. Grounding được chấm trên 5 câu test mỗi strategy theo thang 0–2. Người chấm là tác giả, có hỗ trợ bằng script để tạo các hàng audit từ trace đã đóng băng; đây không phải đánh giá độc lập và có nguy cơ thiên lệch. Chi tiết nằm tại [starter/results/manual_audit.csv](../../starter/results/manual_audit.csv).

Trace top-3: [traces_A.jsonl](../../starter/results/traces_A.jsonl), [traces_B.jsonl](../../starter/results/traces_B.jsonl), [traces_C.jsonl](../../starter/results/traces_C.jsonl). Tổng hợp tự động: [metrics.json](../../starter/results/metrics.json). Metadata chạy: [run_metadata.json](../../starter/results/run_metadata.json).

## Phân tích và kết luận

- A–B: B giảm số chunk từ 179 xuống 118 và tăng Evidence Hit@3 trên dev từ 8/15 lên 12/15. Coherence cũng tăng nhẹ từ 1.0 lên 1.1. Đổi lại, thời gian ingest B cao hơn một chút do tạo embedding cho các chunk paragraph. Trên test nhỏ, cả hai đạt 2/5 Evidence Hit@3.
- B–C: C giữ nguyên chunk và embeddings nhưng thêm filter section. Filter compliance tăng từ 2/4 lên 4/4 trên test và từ 1/3 lên 3/3 trên dev. Tuy nhiên Hit@1 dev giảm từ 5/15 xuống 3/15 và Evidence Hit@3 dev giảm từ 12/15 xuống 10/15, cho thấy filter hữu ích cho phạm vi truy xuất nhưng có thể loại bỏ chunk liên quan khi nhãn section hoặc câu hỏi không khớp hoàn toàn.
- Hai lỗi test tiêu biểu là q17 (so sánh deep learning và classic machine learning), q18 (kiến trúc recurrent và graph phổ biến nhất), và q19 (họ vulnerability/CWE subtype). Các câu này cần tổng hợp thông tin từ bảng hoặc nhiều phần của RQ3/RQ4; top-3 thường trả về đoạn lân cận nhưng không chứa đủ evidence gold. Đây là giới hạn của semantic top-3, không nên diễn giải là lỗi của toàn bộ nội dung nguồn.
- Chiến lược chọn: B là mặc định khi ưu tiên evidence retrieval trên câu hỏi tổng quát. C phù hợp khi người dùng cung cấp section filter đáng tin và mục tiêu chính là tuân thủ phạm vi. A đơn giản hơn và có overlap, nhưng chunk dễ cắt ý hơn.
- Hạn chế: test chỉ có 5 câu, corpus chỉ gồm một bài survey, manual audit do tác giả thực hiện, và thời gian đo phụ thuộc máy/model cache. Các kết quả không chứng minh chiến lược tốt nhất cho corpus khác. Bộ PDF/TeX cũng cần kiểm tra quyền trước khi chia sẻ công khai.
- Nếu làm lại: giữ B làm baseline, bổ sung hybrid dense + lexical reranking trên dev, xử lý bảng/tiêu đề theo section rõ hơn, và tạo test holdout mới trước khi tinh chỉnh tiếp. Có thể đánh giá thêm retrieval đa chunk cho các câu yêu cầu kết hợp nhiều bảng hoặc đoạn.
