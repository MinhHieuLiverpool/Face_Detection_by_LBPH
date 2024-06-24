import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480) -----------------------------------------------------------------------------------------------------------------------------

{ 
- truyền 0 làm đối số cho cv2.VideoCapture(). Sau đó, nó thiết lập độ phân giải của video được chụp thành 640x480 pixel bằng cách sử dụng phương thức set(). Đối số đầu tiên 3 là cho chiều rộng và đối số thứ hai 4 là cho chiều cao. 
-Trong OpenCV, hàm set() được sử dụng để thiết lập các thuộc tính của đối tượng video capture, trong đó 3 đại diện cho thuộc tính CV_CAP_PROP_FRAME_WIDTH (chiều rộng khung hình) và 4 đại diện cho thuộc tính CV_CAP_PROP_FRAME_HEIGHT (chiều cao khung hình). Đây là mã hiển thị các giá trị của thuộc tính CV_CAP_PROP_*:

       + CV_CAP_PROP_FRAME_WIDTH: ID của thuộc tính cho chiều rộng khung hình của video.
       + CV_CAP_PROP_FRAME_HEIGHT: ID của thuộc tính cho chiều cao khung hình của video.
}
-----------------------------------------------------------------------------------------------------------------------------
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
-----------------------------------------------------------------------------------------------------------------------------
{
    Dòng mã này tạo một đối tượng CascadeClassifier trong OpenCV để phát hiện khuôn mặt trong hình ảnh. Đối số 'haarcascade_frontalface_default.xml' là đường dẫn đến tệp XML chứa các thông số cần thiết cho việc nhận dạng khuôn mặt. Tệp XML này chứa thông tin về cấu trúc của một bộ lọc Haar được sử dụng để phát hiện khuôn mặt.

    Bộ lọc Haar là một trong những kỹ thuật phổ biến được sử dụng trong thị giác máy tính để phát hiện các đối tượng trong hình ảnh. Bằng cách sử dụng các bộ lọc Haar và các thông số từ tệp XML, CascadeClassifier có thể phát hiện các đối tượng được đào tạo, như khuôn mặt, trong hình ảnh hoặc video.

    Nguồn : https://hanam88.com/kho-tai-lieu/63/233/phat-hien-khuon-mat-nguoi-trong-anh-su-dung-thu-vien-haar-cascade-voi-opencv-python.html
}

-----------------------------------------------------------------------------------------------------------------------------

face_id = input('\n nhập id người dùng cuối < 1 - n >:  ')

print("\n Đang khởi tạo tính năng chụp khuôn mặt. Nhìn vào camera và chờ đợi ...")
-----------------------------------------------------------------------------------------------------------------------------
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('camera dataser', img)

    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    elif count >= 30:
         break

-----------------------------------------------------------------------------------------------------------------------------

 - cam.read(): Đọc một khung hình từ camera được mở trước đó và lưu trữ nó trong biến img. Hàm này trả về một giá trị boolean (ret) cho biết liệu việc đọc khung hình đã thành công hay không.
 - cv2.flip(img, 1): Lật hình ảnh img theo chiều ngang (trục y). Điều này thường được sử dụng để tránh hình ảnh bị ngược.
 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY): Chuyển đổi hình ảnh img từ không gian màu BGR (Blue-Green-Red) sang không gian màu xám. Hình ảnh xám thường được sử dụng cho việc nhận dạng đối tượng vì nó chỉ chứa thông tin về cường độ sáng, không chứa thông tin màu sắc.
    - face_detector.detectMultiScale(gray, 1.3, 5): Sử dụng bộ phân loại cascade (face_detector) để phát hiện khuôn mặt trong ảnh xám. Phương thức detectMultiScale() sẽ trả về một danh sách các hình chữ nhật (x, y, w, h) mô tả vị trí và kích thước của các khuôn mặt được phát hiện trong ảnh.
    x, y, w, và h là các thông số mô tả vị trí và kích thước của đối tượng được phát hiện, thường là khuôn mặt trong trường hợp của bạn.

    x: là tọa độ x của góc trái trên của hình chữ nhật giới hạn khuôn mặt.
    y: là tọa độ y của góc trái trên của hình chữ nhật giới hạn khuôn mặt.
    w: là chiều rộng của hình chữ nhật giới hạn khuôn mặt.
    h: là chiều cao của hình chữ nhật giới hạn khuôn mặt.
    Vòng lặp for được sử dụng để lặp qua mỗi khuôn mặt được phát hiện trong danh sách faces
    Hàm cv2.rectangle() được sử dụng để vẽ một hình chữ nhật xung quanh khuôn mặt. Các đối số của hàm này là: hình ảnh gốc (img), tọa độ của góc trái trên ((x,y)), tọa độ của góc phải dưới ((x+w,y+h)), màu sắc của hình chữ nhật (trong trường hợp này là màu xanh dương (255,0,0)), và độ dày của đường viền (trong trường hợp này là 2).
    Biến count được tăng lên 1 sau mỗi lần lặp để đếm số lượng khuôn mặt đã được chụp.
    Hàm cv2.imwrite() được sử dụng để lưu hình ảnh của khuôn mặt vào thư mục dataset. Tên tệp được tạo ra dựa trên số thứ tự của khuôn mặt và face_id.
    Cuối cùng, hàm cv2.imshow() được sử dụng để hiển thị hình ảnh được chụp trong cửa sổ với tiêu đề "camera dataser".
    cv2.waitKey(100): Chờ 100ms để xem nếu có sự kiện từ bàn phím. Nếu không có, nó sẽ trả về -1. Nếu có, nó sẽ trả về mã ASCII của phím được nhấn.
    k = cv2.waitKey(100) & 0xff: Lấy các byte cuối cùng từ giá trị trả về (sử dụng & 0xff) để đảm bảo rằng giá trị là trong khoảng từ 0 đến 255.
    if k == 27: Nếu phím ESC được nhấn (mã ASCII là 27), chương trình sẽ thoát khỏi vòng lặp while.
    elif count >= 30: Nếu số lượng khuôn mặt đã chụp (count) đạt hoặc vượt quá 30, chương trình cũng sẽ thoát khỏi vòng lặp while.



-----------------------------------------------------------------------------------------------------------------------------

print("\n Thoát khỏi chương trình")
cam.release()
cv2.destroyAllWindows()

-----------------------------------------------------------------------------------------------------------------------------

cam.release(): Đóng kết nối với camera, giải phóng bộ nhớ và tài nguyên hệ thống mà camera đang sử dụng. Điều này giúp giải phóng tài nguyên và ngừng truy cập vào camera.

cv2.destroyAllWindows(): Đóng tất cả các cửa sổ hiển thị bởi OpenCV. Điều này giúp giải phóng tài nguyên hệ thống và kết thúc chương trình một cách sạch sẽ.


