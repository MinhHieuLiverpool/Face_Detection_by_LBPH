Radius: Nó đại diện cho bán kính xung quanh pixel trung tâm. Nó thường được đặt thành 1. Nó được sử dụng để xây dựng mẫu nhị phân cục bộ tròn.
Neighbors: Số lượng điểm mẫu để xây dựng mẫu nhị phân hình tròn.
Grid X: Số ô theo hướng ngang. Càng biểu thị nhiều ô và lưới mịn hơn, thì kích thước của vectơ đặc trưng thu được càng cao.
Grid Y: Số ô theo hướng dọc. Càng biểu thị nhiều ô và lưới mịn hơn, thì kích thước của vectơ đặc trưng thu được càng cao.

Trong cấu trúc YAML cung cấp, phần data chứa một dãy số lớn, dường như là một phần của ma trận histogram. Đây có thể là dữ liệu số đã được mã hóa để biểu diễn ma trận histogram của các khuôn mặt đã được huấn luyện.

Đối với mô hình nhận dạng khuôn mặt sử dụng thuật toán LBPH (Local Binary Patterns Histograms), histogram là một phần quan trọng để mô tả đặc trưng của mỗi khuôn mặt. Các giá trị trong histogram biểu thị tần suất xuất hiện của các đặc trưng cụ thể được sử dụng để nhận dạng khuôn mặt.

Dãy số lớn trong phần data có thể là một phần của ma trận histogram, trong đó mỗi số có thể biểu thị giá trị tần suất xuất hiện của một đặc trưng cụ thể trong histogram.

nguồn : https://websitehcm.com/face-recognition-va-face-detection-trong-opencv/