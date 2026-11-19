# **ĐỀ XUẤT PHÁT TRIỂN AGENT**

Gedik No-NN  •  Quoridor / Wrongway  •  Tài liệu thảo luận kỹ thuật và định hướng report

Phạm vi: phân tích phiên bản gedik\_no\_nn.py hiện có và đề xuất năm hướng nghiên cứu D1–D5 và lộ trình phiên bản G0–G5. Các hướng, tham số và số trận bên dưới là kế hoạch; chưa phải kết quả benchmark.

**KHUYẾN NGHỊ TRỌNG TÂM**

Ưu tiên G1 (Threat-Aware Evaluation) và G2 (Tactical Wall Priors), thử riêng từng nhánh trước khi kết hợp. Dùng Gedik No-NN làm baseline tham chiếu; cần ghi rõ mã nguồn kế thừa và xác nhận quy định môn học trước khi sử dụng một bản phái sinh làm agent nộp.

# **1\. Hiện trạng Gedik No-NN**

Phiên bản hiện tại là agent PUCT/MCTS đơn luồng, dùng heuristic thủ công để đánh giá trạng thái; không còn mạng QNN1 và pretrained weights. Tuy nhiên, phần lớn logic tìm kiếm và tạo hành động vẫn kế thừa từ bản port Gedik.

| Thành phần | Cơ chế hiện tại | Điểm cần cải thiện |
| :---- | :---- | :---- |
| Trạng thái / luật chơi | Bitboard; wall\_context(), pawns(), child() | Kiểm tra độ đúng khi tích hợp, tối ưu các phép tính lặp |
| Khoảng cách | waves() flood-fill theo lớp; distance(), route(); có LRU cache | Profiling hiệu quả cache, xử lý các trạng thái tường mới |
| Chọn hành động | priors(): ưu tiên giảm đường đi; tường dựa trên Δdistance | Thiếu nhìn xa với tường chuẩn bị / phòng thủ |
| Đánh giá leaf | value(): sigmoid của chênh lệch khoảng cách và tường | Thiếu đánh giá độ an toàn đường đi và phase |
| Tree search | Node.select(): PUCT \+ FPU; search(): visit backup, proof flags | Cần kiểm chứng độ hiệu quả của c, expansion policy, budget |
| Ngân sách | TIME\_BUDGET \= 0.80 giây | Cần theo dõi timeout trên đúng Arena và Kaggle |

**Hàm value hiện tại:**  V(s) \= sigmoid\[0.80 × (d\_opp − d\_self) \+ 0.15 × (w\_self − w\_opp)\]. Đây là điểm số heuristic trong \[0,1\], chưa được hiệu chỉnh thành xác suất thắng thực.

**Wall prior hiện tại:**  Khi tường làm tăng đường đi đối thủ, logit ≈ 0.6 × (Δopp − 1.4 × Δself) − 0.4; sau đó dùng softmax để tạo prior.

# **2\. Tiêu chí lựa chọn hướng phát triển**

•  Đóng góp rõ ràng: thay đổi giải quyết một hạn chế có thể chỉ ra trong code và giải thích được cơ chế.

•  Chi phí hợp lý: phải hoạt động trong giới hạn thời gian Arena; tăng độ phức tạp chỉ khi lợi ích lớn hơn chi phí.

•  Đo lường độc lập: mỗi ý tưởng có phiên bản ablation riêng, có cùng đối thủ và cùng điều kiện thử nghiệm.

•  Minh bạch nguồn gốc: thuật toán PUCT là kiến thức chung; mã Gedik đã kế thừa cần được ghi nguồn, không trình bày như nhóm tự viết.

# **3\. Các hướng phát triển đề xuất**

## **D1 — Threat-Aware Evaluation**

**Mục tiêu:**  Đánh giá một tuyến đường không chỉ theo độ ngắn mà còn theo nguy cơ bị đặt tường kéo dài và khả năng tấn công ngược lại.

**Ý tưởng:**  Bổ sung R\_self (mức tăng shortest-path lớn nhất nếu đối thủ đặt một tường hợp lệ) và R\_opp (khả năng ta kéo dài đường đi đối thủ). Chỉ xét một tập top-K tường chiến thuật để giữ ngân sách.

**Cách triển khai:**  Tạo hàm riêng đánh giá threat; tính trên các candidate ở root hoặc một số leaf quan trọng; cache theo trạng thái tường. Có thể thử một bước phản công (worst-case response) đối với top-K ứng viên.

**Giả thuyết:**  Agent sẽ tránh các nước có shortest-path tốt nhưng dễ bị đối thủ chặn nghiêm trọng.

**Chỉ số kiểm chứng:**  Win rate G1 vs G0; số ván thua do bottleneck; số node/giây; số lần kiểm tra tường và thời gian mỗi nước.

**Rủi ro:**  Threat scan làm giảm số mô phỏng PUCT; feature một bước có thể đánh giá sai chiến thuật nhiều lượt. Cần tránh quét tất cả tường trên mọi leaf.

## **D2 — Tactical Wall Priors**

**Mục tiêu:**  Cải thiện thứ tự và xác suất ưu tiên các bức tường chiến thuật trước khi search.

**Ý tưởng:**  Thêm tín hiệu defense (ngăn đòn đặt tường của đối thủ) và setup (tường có lợi về sau nhưng immediate gain thấp) bên cạnh Δopp và Δself.

**Cách triển khai:**  Bước 1: lọc tường hợp lệ gần tuyến đường hoặc vị trí quân. Bước 2: xếp hạng nhanh. Bước 3: kiểm tra worst-case response cho top-K ứng viên. Chuẩn hóa điểm thành prior bằng softmax.

**Giả thuyết:**  Prior mới giúp PUCT phân bổ lượt thăm nhiều hơn cho các nước phòng thủ và tường chuẩn bị quan trọng.

**Chỉ số kiểm chứng:**  G2 vs G0; số nước tường chiến thuật được đưa vào top-K; tỷ lệ chọn được nước phòng thủ đúng; simulations/turn.

**Rủi ro:**  Tường setup có thể tiêu tốn lượt và tài nguyên; scoring quá đắt làm search chậm. Cần một quota dành riêng cho nước phòng thủ bắt buộc.

## **D3 — Search Improvement: Adaptive PUCT / Progressive Widening**

**Mục tiêu:**  Khám phá nhánh hiệu quả hơn khi số hành động, đặc biệt số lựa chọn đặt tường, rất lớn.

**Ý tưởng A:**  Thử hệ số khám phá c(s) phụ thuộc game phase, số tường còn lại hoặc độ bất định tại node; đối chứng với c \= 1.6 hiện tại.

**Ý tưởng B:**  Progressive widening: giới hạn số children ban đầu và mở dần theo số lượt thăm, nhưng không bỏ các nước phòng thủ bắt buộc.

**Giả thuyết:**  PUCT có thể dành nhiều mô phỏng hơn cho các nhánh hứa hẹn mà không đánh mất những nước chiến thuật quan trọng.

**Chỉ số kiểm chứng:**  G3 vs G0; simulations/turn, expanded nodes, branch width, win rate; kiểm tra trường hợp buộc phải phòng thủ.

**Rủi ro:**  Mở ít nhánh có thể bỏ sót nước thắng/thủ; thay c có thể gây phụ thuộc đối thủ. Nên thử sau G1/G2.

## **D4 — Search Efficiency & Engineering**

**Mục tiêu:**  Tăng lượng tìm kiếm trong cùng ngân sách mà không đổi hoặc làm suy giảm chiến thuật.

**Cách triển khai:**  Profile wall\_context(), waves(), route(), priors(); đo cache hit/miss; thử candidate filtering hai giai đoạn, cache thích hợp và state deduplication. Tạo kiểm thử đối chiếu legal moves và distance với Arena.

**Giả thuyết:**  Giảm thời gian tạo ứng viên và tính khoảng cách sẽ tăng số PUCT iterations hữu ích.

**Chỉ số kiểm chứng:**  Thời gian trung vị/p95 của choose\_action; simulations/turn; số lần gọi flood-fill; tỷ lệ cache hit; timeout/invalid actions.

**Rủi ro:**  Gedik đã có LRU cache; tăng cache hoặc dùng biểu diễn bitboard khác không mặc nhiên nhanh hơn. Đo profiler trước khi tối ưu.

## **D5 — Self-Trained Value Network**

**Mục tiêu:**  Thay heuristic 2-feature hiện tại bằng Value Network do nhóm tự huấn luyện, không sử dụng weights QNN1 từ Gedik.

**Pipeline:**  Sinh dữ liệu (state, result) từ self-play và/hoặc teacher nội bộ; xây feature vector hoặc board encoder; train/validate trên trận và seed tách biệt; tích hợp Value Network vào PUCT; đo tốc độ inference.

**Bước khởi đầu:**  Huấn luyện MLP gọn nhẹ với shortest-path, wall stock, route threats, game phase; thử trước khi dùng CNN lớn. So sánh G0 heuristic, G5 MLP và G1 handcrafted threat.

**Giả thuyết:**  Learned value có thể học quan hệ phi tuyến khó biểu diễn bằng một công thức cố định.

**Chỉ số kiểm chứng:**  Kết quả đấu thực tế; accuracy/calibration trên tập holdout; inference cost; số simulations/turn.

**Rủi ro:**  Dữ liệu self-play thiên lệch, overfitting hoặc model quá chậm. Nếu chỉ bắt chước teacher thì gọi là supervised/imitation; chỉ mô tả RL khi có quá trình học từ tương tác/reward.

# **4\. Ma trận ưu tiên**

| Hướng | Độ khó | Giá trị cho report | Ưu tiên |
| :---- | :---- | :---- | :---- |
| D1 — Threat-aware | Trung bình | Giải quyết hạn chế cụ thể của evaluation; dễ minh họa replay | 1 |
| D2 — Wall priors | Trung bình | Đóng góp trực tiếp vào action ordering; ablation rõ | 2 |
| D4 — Efficiency | Trung bình | Cải thiện tài nguyên và độ ổn định; chỉ số đo dễ | 3 |
| D3 — Adaptive search | Trung bình – cao | Thử nghiệm thuật toán search sâu hơn; rủi ro cao | 4 |
| D5 — Learned value | Cao | Đóng góp ML đáng kể nếu có dữ liệu và training thật | 5 |

Thứ tự trên dựa trên tính khả thi và khả năng kiểm chứng, không phải dự đoán chắc chắn hướng nào sẽ thắng. Nếu profiler cho thấy bottleneck lớn, có thể đưa hướng D4 lên ưu tiên trước D1/D2.

# **5\. Lộ trình phiên bản và ablation**

| Phiên bản | Thay đổi độc lập | Mục tiêu so sánh |
| :---- | :---- | :---- |
| G0 | gedik\_no\_nn.py (baseline ngoài) | Đo mức hiện tại; không coi là đóng góp mới |
| G1 | G0 \+ threat-aware value | G1 vs G0 |
| G2 | G0 \+ tactical wall priors | G2 vs G0 |
| G3 | G0 \+ adaptive PUCT / widening | G3 vs G0 |
| G4 | Kết hợp những thành phần vượt qua ablation (có thể gồm D4) | G4 vs G0, G1, G2, G3 |
| G5 | Custom value network tự huấn luyện | G5 vs G0, G1 và G4 |

Quy tắc phát triển: G1/G2/G3 tạo từ cùng G0 để tách tác động từng feature; G4 chỉ kết hợp các thành phần đã chứng minh giá trị. Không mặc định phiên bản số lớn hơn sẽ mạnh hơn.

# **6\. Đề xuất phương pháp đánh giá**

**Đối thủ:**  G0 (Gedik No-NN), Gedik QNN1 (benchmark ngoài), Strong Search V1.6 và các agent Minimax/Strong Search hiện có của nhóm. Luôn tách đối thủ dùng để chọn phiên bản khỏi vòng đánh giá cuối.

**Số trận:**  Giai đoạn smoke test: 4 ván/cặp để phát hiện lỗi. Giai đoạn sàng lọc: khoảng 40 ván/cặp. Giai đoạn kiểm định quan trọng: khoảng 100–200 ván/cặp nếu tài nguyên cho phép. Đây là kế hoạch, không phải số liệu đã chạy.

**Thiết kế trận:**  Đổi bên P0/P1 cân bằng; giữ nguyên time budget, Arena version và cấu hình máy. Quoridor có trạng thái khởi đầu cố định: seed khác không đồng nghĩa với khai cuộc độc lập; cần đo trùng lặp chuỗi nước đi và chỉ dùng opening đa dạng nếu luật cho phép.

**Báo cáo số liệu:**  Games, wins/draws/losses, score rate \= (W \+ 0.5D)/N, Wilson 95% CI (hoặc phương pháp thích hợp), win rate theo P0/P1, timeout/invalid, thời gian nước đi, iterations/nodes và số replay đặc trưng.

**Chống selection bias:**  Giữ một tập trận/đối thủ đánh giá cuối chưa dùng trong tuning; lưu code hash, config, seed và replay cho mọi phiên bản.

# **7\. Bằng chứng cần lưu cho report**

| Mục thầy yêu cầu | Dữ liệu cần có |
| :---- | :---- |
| (1) Key idea | Sơ đồ pipeline; công thức đánh giá/prior/search; nêu rõ phần tham khảo và phần đóng góp |
| (2) Development process | Changelog G0→G5; commit/hash; giả thuyết; kết quả ablation; các variant thất bại; số biến thể AI hỗ trợ tạo và tiêu chí chọn |
| (3) Evaluation | Danh sách opponents; số ván thực tế; cân bằng P0/P1; điều kiện Arena; bảng kết quả; replay đại diện |

# **8\. Ghi nguồn và tính độc lập của bài nộp**

G0 là bản port từ dự án Gedik của Samet Köse. Các bản G1–G5 viết tiếp trên G0 vẫn là các bản phái sinh về mặt mã nguồn, kể cả khi nhóm thay đổi evaluation hoặc train một mô hình mới. Cần giữ thông báo bản quyền/giấy phép tương ứng, ghi nguồn rõ trong report và xin xác nhận của giảng viên về việc được dùng agent ngoài làm nền.

Nếu quy định yêu cầu agent do nhóm tự xây dựng, phương án an toàn hơn là tự triển khai PUCT, board engine và action generation dựa trên mô tả thuật toán công khai; dùng Gedik làm related work và đối thủ benchmark, không dùng code/weights để nộp.

# **9\. Đề xuất bước triển khai ngay**

•  Đóng băng G0: lưu file, checksum và kết quả smoke test; đảm bảo legal action / không timeout.

•  Xây G1 độc lập: threat score top-K tại root, đo overhead và ví dụ bottleneck.

•  Xây G2 độc lập: ưu tiên tường phòng thủ/setup nhưng giữ top-K và quota hạn chế.

•  Chạy benchmark sàng lọc G1 vs G0, G2 vs G0, G1 vs G2; lựa chọn dựa trên win rate và chi phí search.

•  Sau đó mới quyết định thử G3 hoặc G5; đồng thời lưu đầy đủ nhật ký AI-assisted development nếu có.

# **Nguồn và phạm vi xác minh**

Nguồn chính: file gedik\_no\_nn.py do nhóm cung cấp (đối chiếu trực tiếp các hàm wall\_context, waves, route, value, priors, Node.select, search và choose\_action).

Nguồn gốc tham khảo: https\://github.com/msametkose-ctrl/gedik (Gedik, Samet Köse).

Tài liệu này là kế hoạch nghiên cứu dựa trên kiến trúc hiện tại. Các công thức D1/D2 và chỉ số mục tiêu là đề xuất, chưa được triển khai hoặc xác nhận hiệu quả bằng tournament.