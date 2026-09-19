# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Phạm Thái Vũ  
**Mã sinh viên:** 2A202602695  
**Nhóm:** RauMa  
**Ngày:** 2026-09-19  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao biểu thị hai vector chỉ về cùng một hướng trong không gian embedding nhiều chiều, tức là hai đoạn văn bản có sự tương đồng rất lớn về mặt ý nghĩa ngữ nghĩa (semantic meaning), không phụ thuộc vào độ dài hay số lượng từ của câu.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên nộp khóa luận tốt nghiệp và hoàn thành thủ tục xét ra trường."
- Câu B: "Người học nộp báo cáo nghiên cứu cuối khóa và làm hồ sơ công nhận tốt nghiệp."
- Tại sao tương đồng: Hai câu sử dụng từ vựng hoàn toàn khác biệt nhau nhưng biểu thị cùng một khái niệm hành động học vụ và bối cảnh thực hiện trong trường đại học.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Hạn mức mượn sách tham khảo về nhà cho sinh viên là 14 ngày."
- Câu B: "Nhiệt độ sôi của nước nguyên chất ở áp suất khí quyển tiêu chuẩn là 100 độ C."
- Tại sao khác: Hai câu thuộc hai lĩnh vực tri thức hoàn toàn xa lạ nhau (quy chế quản lý thư viện trường học đối lập với định luật vật lý nhiệt học).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid bị phụ thuộc nặng nề vào độ dài vector (magnitude), do đó một đoạn văn dài và một câu ngắn dù cùng nói về một nội dung vẫn sẽ bị tính khoảng cách Euclid rất lớn. Ngược lại, cosine similarity chuẩn hóa độ dài và chỉ đo góc giữa hai vector, giúp tập trung thuần túy vào mối tương quan ngữ nghĩa.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> - Bước nhảy giữa các chunk (step): $\text{step} = \text{chunk\_size} - \text{overlap} = 500 - 50 = 450$ ký tự.
> - Số lượng chunk: $\lceil (\text{độ\_dài} - \text{overlap}) / \text{step} \rceil = \lceil (10.000 - 50) / 450 \rceil = \lceil 9.950 / 450 \rceil = \lceil 22,11 \rceil = 23$.
> *Đáp án:* **23 chunks** (kiểm chứng chính xác bằng `FixedSizeChunker(chunk_size=500, overlap=50).chunk("a"*10000)`).

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, bước nhảy giảm còn $500 - 100 = 400$ ký tự, số lượng chunk tăng lên $\lceil (10.000 - 100) / 400 \rceil = 25$ chunks (tăng 2 chunk). Ta muốn độ chồng chéo nhiều hơn để bảo toàn ngữ cảnh liền mạch tại các ranh giới cắt đoạn, ngăn chặn nguy cơ một câu văn, con số hoặc điều khoản quy định quan trọng bị xẻ làm đôi giữa hai chunk cạnh nhau.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng biểu thức chính quy Lookbehind `r'(?<=[.!?])(?:\s+|\n+)'` để tách câu chính xác tại các dấu chấm, hỏi, than hoặc xuống dòng mà không làm mất dấu câu cuối. Sau đó, gom các câu thành từng nhóm tối đa `max_sentences_per_chunk` câu và gọi `strip()`. Xử lý chặt chẽ trường hợp text rỗng hoặc chỉ có khoảng trắng để luôn trả về `[]`.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Triển khai thuật toán 2 chiều: (1) Chiều đệ quy xuống theo danh sách ưu tiên `["\n\n", "\n", ". ", " ", ""]`, nếu mảnh văn bản lớn hơn `chunk_size` thì tiếp tục hạ cấp separator để chia nhỏ; (2) Chiều gom lên (merge): nối các mảnh nhỏ liền kề lại bằng separator hiện tại cho đến khi đạt sát ngưỡng `chunk_size` để chống vỡ vụn ngữ cảnh. Base case xử lý khi text rỗng, text nhỏ hơn `chunk_size` hoặc khi danh sách separator rỗng (`separators=[]`).

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ dưới dạng danh sách các bản ghi chuẩn hóa trong bộ nhớ RAM (`self._store`). Mỗi record chứa `id`, `content`, `metadata` (được bảo đảm luôn gán khóa `doc_id` của tài liệu gốc) và vector `embedding`. Khi `search`, hàm tính điểm tương đồng cosine (bằng hàm `_dot`) giữa embedding câu truy vấn và từng record, sau đó sắp xếp giảm dần theo điểm `score` và trả về `top_k` kết quả (lược bỏ trường vector để output gọn gàng).

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Luôn thực hiện **lọc trước (pre-filtering)**: lọc tập ứng viên trong `self._store` thỏa mãn toàn bộ cặp khóa-giá trị trong `metadata_filter` trước khi tính điểm tương đồng. Cách tiếp cận này đảm bảo không bao giờ bị mất kết quả khi `top_k` bị chiếm chỗ bởi các tài liệu không khớp. Với `delete_document`, lọc loại bỏ toàn bộ các bản ghi có `metadata['doc_id'] == doc_id` hoặc `id == doc_id` và trả về `True` nếu có ít nhất 1 bản ghi bị xóa.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Kiểm tra an toàn trạng thái store rỗng để thông báo thân thiện mà không gọi LLM vô ích. Dựng prompt có ngữ cảnh trích dẫn được đánh số thứ tự `[1]`, `[2]`, `[3]` kèm thông tin nguồn (Source Traceability) và bổ sung ràng buộc chống bịa đặt (Hallucination Guardrail): chỉ trả lời dựa trên tài liệu cung cấp và nói rõ nếu không tìm thấy thông tin. Cuối cùng, gọi `llm_fn` với prompt hoàn chỉnh.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- D:\Lab\K4-L3A-Data-Foundations\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Lab\K4-L3A-Data-Foundations
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.17s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Sinh viên nộp khóa luận tốt nghiệp để xét ra trường." | "Người học nộp báo cáo tốt nghiệp để làm thủ tục hoàn thành khóa học." | Cao | 0.1534 | Đúng về xu hướng (dương) |
| 2 | "Sinh viên được phép gia hạn sách mượn về nhà." | "Sinh viên không được phép gia hạn sách mượn về nhà." | Cao (do trùng từ) | 0.0049 | Bất ngờ (gần 0) |
| 3 | "Hạn mức mượn sách tham khảo của sinh viên là 03 tài liệu trong 30 ngày." | "Giờ mở cửa phòng đọc thư viện từ 8:00 đến 20:30 từ thứ Hai đến thứ Sáu." | Trung bình | 0.1533 | Đúng |
| 4 | "Bạn đọc phải quẹt thẻ tại cổng kiểm soát an ninh tự động." | "Công thức hóa học của nước tinh khiết là hai nguyên tử hydro kết hợp với một oxy." | Rất thấp | 0.0882 | Đúng |
| 5 | "Thư viện dừng phục vụ 15 phút trước giờ đóng cửa." | "The library stops serving readers 15 minutes before closing time." | Cao | 0.0743 | Thấp hơn kỳ vọng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là Cặp 2: hai câu trùng lặp tới 90% số lượng từ vựng và chỉ khác nhau chữ "không", nhưng điểm tương đồng rơi về sát 0.0049; đồng thời Cặp 5 (câu dịch tiếng Anh) có điểm khá thấp (0.0743) khi dùng mock embedding. Điều này phản ánh rõ bản chất: `_mock_embed` chỉ băm chuỗi MD5 thành vector giả ngẫu nhiên nên không thể nắm bắt được ngữ nghĩa đa ngôn ngữ hay quan hệ đồng nghĩa thực thụ. Để RAG hoạt động tối ưu, hệ thống cần các mô hình embedding chuyên sâu được huấn luyện trên không gian đa ngữ để bảo toàn mối quan hệ ngữ nghĩa thay vì phụ thuộc vào từ vựng bề mặt.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src` (Chiến lược cá nhân của R1: `FixedSizeChunker(chunk_size=500, overlap=100)`). **5 câu hỏi này trùng khớp 100% với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Ở Thư viện Đại học Ngoại thương được mượn về nhà tối đa bao nhiêu tài liệu và trong bao lâu? *(filter: `audience: student`)* | `ftu-library-borrowing-student` (Thủ tục và hạn mức mượn cho người học) | 0.2462 *(0.6480 semantic)* | Có *(Top-2 chứa con số "03 tài liệu/30 ngày")* | Sinh viên được mượn tối đa 03 tài liệu/30 ngày đối với sách tham khảo (gia hạn 1 lần 15 ngày), sách giáo trình mượn hết môn học. |
| 2 | Chỉ số trùng lặp tối đa cho khóa luận tốt nghiệp khi tra soát trên Turnitin là bao nhiêu? | `library-digital-resources` (Khai thác CSDL và học liệu số) | 0.2424 *(0.5820 semantic)* | Không *(Do Fixed-size cắt rời dòng bảng số liệu xuống hạng 5; Recursive đạt top-2)* | Ngưỡng chỉ số trùng lặp tối đa cho Khóa luận tốt nghiệp là 25% (khi truy xuất trúng bảng quy định Turnitin). |
| 3 | Sau khi người học nộp biên bản tra soát, thư viện trả kết quả trong bao lâu và ở đâu? | `dav-library-rules` (Nội quy bảo quản tài sản phòng đọc) | 0.2455 *(0.6510 semantic)* | Không *(Bị nhiễu từ khóa "thời hạn" với chunk mượn sách; không lọt top-3)* | Thư viện trả kết quả tra soát tại K214 (Nhà K – ĐH Ngoại thương), chậm nhất trong vòng 02 ngày làm việc kể từ ngày nộp biên bản. |
| 4 | Thư viện Học viện Ngoại giao mở cửa vào Thứ Bảy từ mấy giờ đến mấy giờ? | `dav-library-rules` (Mục 2: Thời gian mở cửa áp dụng từ 01/3/2022) | 0.2348 *(0.7240 semantic)* | Có *(Top-1 trúng tuyệt đối, chứa trọn vẹn giờ mở cửa Thứ Bảy)* | Thứ Bảy mở cửa từ 9:00 đến 18:00 (thông tầm không nghỉ giữa giờ); thư viện dừng phục vụ bạn đọc trước 15 phút. |
| 5 | Mượn tài liệu về nhà trả trễ hạn bao nhiêu lần trong một năm học thì bị coi là hành vi nghiêm cấm? | `ftu-library-borrowing-faculty` (Gia hạn và xử lý trễ hạn cán bộ) | 0.3943 *(0.6150 semantic)* | Có *(Top-3 trong run có overlap; Top-1 bị lệch do từ khóa "trả trễ hạn" khác "quá hạn")* | Hành vi bị nghiêm cấm tại Thư viện: Mượn tài liệu về nhà quá hạn trên 03 lần/năm học. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 2 / 5 *(Câu 1 và Câu 4 có chunk chứa đáp án trong top-3; trong đó Câu 4 đứng Top-1 tuyệt đối)*

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Tôi học được từ R2 rằng việc thiết lập kích thước chunking (`chunk_size`) và đơn vị cắt đoạn có ảnh hưởng mang tính sống còn đến khả năng truy xuất: `FixedSizeChunker` cơ học dễ xé lẻ các cặp thông tin điều kiện – con số nằm ở ranh giới cắt đoạn, trong khi `RecursiveChunker` tôn trọng ngắt dòng/đề mục giúp bảo toàn ngữ cảnh của các bảng biểu và quy trình tốt hơn rất nhiều. Ngoài ra, việc chấm điểm 2 mức (Mức 1 theo `doc_id` và Mức 2 theo nội dung `needle`) đã lột trần sự "thổi phồng" điểm số nếu chỉ đánh giá hời hợt theo tên file.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
