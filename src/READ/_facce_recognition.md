import cv2
import numpy as np
import os 
-----------------------------------------------------------------------------------------------------------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
-----------------------------------------------------------------------------------------------------------------------------
-Tạo một đối tượng nhận dạng khuôn mặt sử dụng thuật toán LBPH (Local Binary Patterns Histograms) thông qua phương thức cv2.face.LBPHFaceRecognizer_create(). Sau đó, nó đọc mô hình đã được huấn luyện từ tệp YAML 'trainer/trainer.yml' bằng phương thức read().
Khi mô hình được đọc vào bộ nhớ, bạn có thể sử dụng nó để nhận dạng khuôn mặt trong các hình ảnh hoặc video mới. Mô hình đã được huấn luyện sẽ chứa thông tin cần thiết để nhận dạng các khuôn mặt dựa trên các đặc trưng đã học từ dữ liệu huấn luyện trước đó.
Sau đó, bạn có thể sử dụng đối tượng recognizer để thực hiện nhận dạng khuôn mặt trên các ảnh mới hoặc trong thời gian thực, tùy thuộc vào nhu cầu của ứng dụng của bạn.

- Tạo một đối tượng bộ phân loại cascade để phát hiện khuôn mặt trong hình ảnh. Đối tượng được tạo thông qua phương thức cv2.CascadeClassifier(), và nó được cấu hình để sử dụng tệp XML "haarcascade_frontalface_default.xml" như một phần của bộ phân loại.

cascadePath: Đường dẫn đến tệp XML chứa thông tin về bộ phân loại cascade được sử dụng để phát hiện khuôn mặt. Trong trường hợp này, đó là "haarcascade_frontalface_default.xml".
faceCascade: Đối tượng bộ phân loại cascade được tạo ra và sẽ được sử dụng để phát hiện khuôn mặt trong hình ảnh.
Sau khi đối tượng faceCascade được tạo, bạn có thể sử dụng nó để phát hiện khuôn mặt trong các hình ảnh hoặc video bằng cách gọi phương thức detectMultiScale().
-----------------------------------------------------------------------------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None','HieuLiverpool','DoanTuanTai']
cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480) 
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
-----------------------------------------------------------------------------------------------------------------------------
font: Loại phông chữ được sử dụng để hiển thị tên người được nhận dạng lên hình ảnh.
id: ID của người được nhận dạng. Trong ví dụ này, được khởi tạo là 0, nhưng có thể thay đổi tùy thuộc vào việc sử dụng.
names: Danh sách tên của các người được nhận dạng.
cam: Đối tượng VideoCapture để truy cập dữ liệu từ máy ảnh.
cam.set(3, 640) và cam.set(4, 480): Thiết lập kích thước khung hình cho máy ảnh. Ở đây, kích thước được đặt là 640x480 pixels.
minW và minH: Kích thước tối thiểu của khuôn mặt được chấp nhận. Các kích thước này được tính toán dựa trên kích thước của khung hình từ máy ảnh

-----------------------------------------------------------------------------------------------------------------------------
while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
-----------------------------------------------------------------------------------------------------------------------------
 - ret, img = cam.read(): Đọc một khung hình từ máy ảnh và gán nó cho biến img. Biến ret chứa giá trị True nếu việc đọc khung hình thành công, ngược lại là False.

 - gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY): Chuyển đổi ảnh màu sang ảnh xám. Việc này giúp giảm chi phí tính toán khi xử lý ảnh, đồng thời cũng là bước tiền xử lý quan trọng trước khi nhận dạng khuôn mặt.
 img: Là ảnh màu ban đầu được đọc từ máy ảnh.
cv2.COLOR_BGR2GRAY: Đây là một hằng số của OpenCV, chỉ định rằng chúng ta muốn chuyển đổi ảnh từ không gian màu BGR (màu mặc định cho ảnh màu trong OpenCV) sang ảnh xám.
Kết quả là biến gray sẽ chứa ảnh xám tương ứng với ảnh màu img

-scaleFactor: Tham số này xác định tỷ lệ thu phóng của ảnh trong quá trình phát hiện khuôn mặt. Giá trị lớn hơn của scaleFactor có nghĩa là việc phát hiện khuôn mặt sẽ diễn ra nhanh hơn, nhưng có thể dẫn đến việc bỏ sót các khuôn mặt nhỏ hoặc đặt sai vị trí của khuôn mặt. Trong trường hợp này, giá trị 1.2 được sử dụng.
 {
    Khi scaleFactor được đặt thành một giá trị lớn hơn 1, nó chỉ định tỷ lệ tăng kích thước của cửa sổ phát hiện mỗi lần nó được chuyển đổi qua các tầng của hình ảnh. Một giá trị lớn hơn 1 sẽ làm tăng kích thước cửa sổ phát hiện, dẫn đến việc kiểm tra khu vực lớn hơn của hình ảnh mỗi lần. Điều này có thể giúp tăng tốc độ của thuật toán nhưng cũng có thể dẫn đến việc bỏ sót các khuôn mặt nhỏ hoặc bị che khuất.

Khi scaleFactor được đặt thành một giá trị trong khoảng từ 0 đến 1, nó chỉ định tỷ lệ giảm kích thước của cửa sổ phát hiện mỗi lần nó được chuyển đổi. Một giá trị trong khoảng này sẽ làm giảm kích thước cửa sổ phát hiện, dẫn đến việc kiểm tra khu vực nhỏ hơn của hình ảnh mỗi lần. Điều này có thể làm tăng độ chính xác của việc phát hiện nhưng cũng có thể làm chậm quá trình phát hiện.

Ví dụ, nếu scaleFactor được đặt thành 1.2, cửa sổ phát hiện sẽ được tăng kích thước 20% mỗi lần nó được chuyển đổi qua các tầng của hình ảnh. Điều này có thể làm tăng tốc độ của thuật toán nhưng cũng có thể dẫn đến việc bỏ sót các khuôn mặt nhỏ hoặc bị che khuất.
 }
-minNeighbors: Tham số này xác định số lượng neighbors cần phát hiện trong cửa sổ phát hiện trước khi kết luận rằng một khuôn mặt đã được tìm thấy. Giá trị cao hơn của minNeighbors có thể giúp giảm thiểu các phát hiện giả mạo, nhưng cũng có thể làm giảm độ chính xác của kết quả phát hiện. Trong trường hợp này, giá trị 5 được sử dụng.

-minSize: Kích thước tối thiểu của khuôn mặt được phát hiện. Chỉ có các khuôn mặt có kích thước lớn hơn giá trị minSize mới được coi là hợp lệ. Trong trường hợp này, kích thước tối thiểu được tính dựa trên kích thước của khung hình từ máy ảnh, với giá trị 10% của chiều rộng và chiều cao tương ứng của khung hình.
-----------------------------------------------------------------------------------------------------------------------------
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))      
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    cv2.imshow('Camera Nhận diện',img) 
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

-----------------------------------------------------------------------------------------------------------------------------
- cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2): Hàm cv2.rectangle() được sử dụng để vẽ một hộp giới hạn trên hình ảnh img. Các tham số của hàm như sau:
img: Hình ảnh đích mà chúng ta muốn vẽ hộp giới hạn lên.
(x,y): Tọa độ của góc trái trên của hộp giới hạn. Đây là tọa độ của điểm góc trái trên của hộp.
(x+w,y+h): Tọa độ của góc phải dưới của hộp giới hạn. Đây là tọa độ của điểm góc phải dưới của hộp.
(0,255,0): Màu sắc của hộp giới hạn. Trong trường hợp này, là màu xanh lá cây. Định dạng màu là (B, G, R), với B là mức độ màu xanh, G là mức độ màu lục, và R là mức độ màu đỏ.
2: Độ dày của đường viền của hộp giới hạn. Trong trường hợp này, là 2 pixels.
Kết quả là một hộp giới hạn màu xanh lá cây sẽ được vẽ xung quanh khuôn mặt được phát hiện, với độ dày của đường viền là 2 pixels.
# vẽ hình chữ nhật màu xanh 

-gray[y:y+h,x:x+w]: Phần của ảnh xám (gray) chứa khuôn mặt được xác định bởi các tọa độ (x, y, w, h) của hộp giới hạn. Đây là phần ảnh mà chúng ta muốn mô hình nhận dạng và dự đoán ID của khuôn mặt đó.
recognizer.predict(): Phương thức này của mô hình nhận dạng sẽ dự đoán ID của khuôn mặt dựa trên phần của ảnh đã chọn. Kết quả của dự đoán sẽ trả về hai giá trị: id là ID được dự đoán và confidence là mức độ tự tin của dự đoán. Đối với thuật toán LBPH, confidence là sai số giữa khuôn mặt đã được nhận dạng và mẫu khuôn mặt trong tập dữ liệu huấn luyện. Điều này có nghĩa là mức độ tự tin càng cao thì kết quả dự đoán càng chính xác.

Phương thức predict() này được sử dụng để dự đoán ID và tính toán mức độ tự tin của dự đoán, sau đó có thể sử dụng thông tin này để hiển thị kết quả hoặc thực hiện các hành động phù hợp khác trong ứng dụng của bạn.

Giá trị của confidence thường là một số dương, và càng gần 0 thì mô hình càng tự tin vào dự đoán của mình. Nếu confidence gần 0, điều này có nghĩa là mô hình rất tự tin rằng dự đoán của nó là chính xác.

Tuy nhiên, nếu confidence lớn, tức là xa 0, điều này có thể ngụ ý rằng mô hình không chắc chắn về dự đoán của mình. Trong trường hợp này, dự đoán có thể không chính xác hoặc có thể cần được kiểm tra lại.

cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2): Vẽ văn bản str(id) (ID của người được nhận dạng) lên hình ảnh img. x+5 và y-5 là tọa độ bắt đầu của văn bản, được dịch chuyển một chút để nằm bên phải và trên của hộp giới hạn. font là font chữ được sử dụng (có thể là một trong các hằng số được xác định trước trong OpenCV). 1 là kích thước của font, và (255,255,255) là màu của văn bản (trong trường hợp này, là màu trắng). 2 là độ dày của văn bản.


cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1): Tương tự như trên, nhưng với văn bản str(confidence) (mức độ tự tin của việc nhận dạng khuôn mặt) được hiển thị. (x+5, y+h-5) là tọa độ bắt đầu của văn bản, được dịch chuyển một chút để nằm bên phải và dưới của hộp giới hạn. (255,255,0) là màu của văn bản (trong trường hợp này, là màu vàng). 1 là kích thước của font, và 1 là độ dày của văn bản.

cv2.waitKey(10): Hàm này chờ đợi một sự kiện phím nhấn trong một khoảng thời gian nhất định (trong trường hợp này, 10 miligiây). Nó trả về mã ASCII của phím được nhấn.

k & 0xff: Đảm bảo rằng chỉ có 8 bit thấp nhất của k được giữ lại, vì trong một số trường hợp, waitKey() có thể trả về giá trị lớn hơn 255.

if k == 27: Kiểm tra xem liệu phím được nhấn có phải là phím Esc hay không (với mã ASCII là 27). Nếu phải, chương trình sẽ thoát khỏi vòng lặp while và kết thúc.

-----------------------------------------------------------------------------------------------------------------------------
print("\n [THÔNG BÁO] Thoát khỏi chương trình")
cam.release()
cv2.destroyAllWindows()

-----------------------------------------------------------------------------------------------------------------------------
am.release(): Giải phóng tài nguyên của thiết bị camera, đảm bảo rằng không có tiến trình nào đang sử dụng camera sau khi chương trình kết thúc.

cv2.destroyAllWindows(): Đóng tất cả các cửa sổ hình ảnh được tạo ra bởi OpenCV. Điều này đảm bảo rằng tất cả các cửa sổ sẽ được đóng khi chương trình kết thúc, ngăn cản bất kỳ cửa sổ nào vẫn còn mở sau khi chương trình đã kết thúc.