import cv2
import numpy as np
from PIL import Image
import os
-----------------------------------------------------------------------------------------------------------------------------
    +cv2: Thư viện OpenCV, được sử dụng rộng rãi cho các công việc thị giác máy tính.
    numpy: Một thư viện cho tính toán số trong Python. Thường được sử dụng trong xử lý ảnh cho các phép toán mảng hiệu quả.
    +PIL: Thư viện Python Imaging, cung cấp các khả năng xử lý ảnh. Thường được sử dụng để mở, chỉnh sửa và lưu nhiều định dạng tệp ảnh khác nhau.
    +os: Module os cung cấp một cách di động để tương tác với hệ điều hành. Thường được sử dụng cho các hoạt động tệp như đọc thư mục, kiểm tra sự tồn tại của tệp, v.v.

-----------------------------------------------------------------------------------------------------------------------------
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()

-----------------------------------------------------------------------------------------------------------------------------
path = 'dataset': Đường dẫn tới thư mục chứa dữ liệu huấn luyện. Trong trường hợp này, giả sử rằng các hình ảnh của các khuôn mặt đã được chụp và được lưu trữ trong thư mục có tên 'dataset'.

recognizer = cv2.face.LBPHFaceRecognizer_create(): Khởi tạo một đối tượng nhận dạng khuôn mặt sử dụng thuật toán LBPH. Đối tượng này sẽ được sử dụng để huấn luyện trên các hình ảnh của các khuôn mặt trong thư mục 'dataset' và sau đó sẽ được sử dụng để nhận dạng khuôn mặt trong thời gian thực.

- Thuật toán LBPH (Local Binary Patterns Histograms) là một trong những thuật toán phổ biến trong lĩnh vực nhận dạng khuôn mặt. Nó hoạt động bằng cách xác định các mẫu nhị phân cục bộ trong ảnh và tạo ra các histogram đặc trưng cho mỗi khuôn mặt. Sau đó, những histogram này được sử dụng để huấn luyện một mô hình nhận dạng khuôn mặt.

- Sau khi tạo đối tượng nhận dạng khuôn mặt bằng LBPH, bạn có thể sử dụng nó để huấn luyện trên dữ liệu ảnh khuôn mặt và sau đó sử dụng mô hình đã huấn luyện để nhận dạng khuôn mặt trong thời gian thực.

Nguồn: https://websitehcm.com/face-recognition-va-face-detection-trong-opencv/

-----------------------------------------------------------------------------------------------------------------------------
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml"); 
#Tạo một đối tượng bộ phân loại Cascade để nhận diện khuôn mặt. Tệp XML "haarcascade_frontalface_default.xml" 
#chứa các thông số cần thiết để phát hiện khuôn mặt trong hình ảnh.

Bộ phân loại cascade là một phương pháp phổ biến được sử dụng để phát hiện các đối tượng trong hình ảnh. Nó hoạt động bằng cách áp dụng một loạt các bộ lọc đặc biệt lên hình ảnh và kiểm tra xem có mặt các đặc điểm đặc trưng của đối tượng cần phát hiện không.
Sau khi tạo đối tượng CascadeClassifier và cấu hình nó để phát hiện khuôn mặt, bạn có thể sử dụng nó để phát hiện các khuôn mặt trong hình ảnh hoặc video bằng cách gọi phương thức detectMultiScale() trên hình ảnh đó.

-----------------------------------------------------------------------------------------------------------------------------
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
-----------------------------------------------------------------------------------------------------------------------------
    -os.listdir(path): Trả về một danh sách chứa tất cả các tên tệp trong thư mục được chỉ định bởi biến path.
    [os.path.join(path,f) for f in os.listdir(path)]: Dùng một list comprehension để tạo một danh sách mới với đường dẫn đầy đủ của mỗi tệp ảnh trong thư mục. Hàm os.path.join() được sử dụng để kết hợp đường dẫn của thư mục với tên tệp ảnh.
    -faceSamples=[]: Một danh sách được khởi tạo để lưu trữ các mẫu khuôn mặt từ các tệp ảnh.
    -ids = []: Một danh sách khác được khởi tạo để lưu trữ các ID tương ứng với mỗi khuôn mặt. ID có thể là số duy nhất hoặc nhãn đại diện cho mỗi người.
    -Image.open(imagePath): Mở tệp ảnh tại đường dẫn imagePath bằng thư viện PIL (Python Imaging Library).
    -.convert('L'): Chuyển đổi ảnh sang chế độ ảnh xám. Điều này làm giảm kích thước dữ liệu và đơn giản hóa quá trình xử lý.
    -PIL_img: Là biến chứa ảnh sau khi đã được mở và chuyển đổi.
    -p.array(PIL_img, 'uint8'): Chuyển đổi ảnh từ đối tượng PIL thành một mảng NumPy, sử dụng kiểu dữ liệu unsigned integer 8-bit ('uint8'). Điều này chuyển đổi ảnh từ định dạng PIL sang một mảng NumPy có thể được sử dụng cho việc xử lý và huấn luyện mô hình nhận dạng khuôn mặt.
-----------------------------------------------------------------------------------------------------------------------------
 id = int(os.path.split(imagePath)[-1].split(".")[1])
-----------------------------------------------------------------------------------------------------------------------------

- {
    os.path.split(imagePath)[-1] được sử dụng để lấy phần cuối cùng của đường dẫn tệp ảnh, nghĩa là tên tệp ảnh.

    os.path.split(imagePath): Phân tách đường dẫn của tệp ảnh thành một tuple gồm hai phần: đường dẫn thư mục và tên tệp.
    [-1]: Chọn phần tử cuối cùng của tuple, nghĩa là tên tệp ảnh.
    Ví dụ, nếu imagePath là dataset/User.1.1.jpg, thì os.path.split(imagePath)[-1] sẽ trả về 'User.1.1.jpg'. Điều này giúp lấy tên của tệp ảnh từ đường dẫn đầy đủ.
}


{
    - .split("."): Phân tách tên tệp ảnh bằng dấu chấm, tạo ra một danh sách các phần của tên tệp.
    -  [1]: Chọn phần thứ hai của danh sách, chứa ID.

    Giả sử tên tệp ảnh là 'User.1.1.jpg':

    Sau khi được phân tách bằng dấu chấm, chúng ta có một danh sách gồm ['User', '1', '1', 'jpg'].
    [1] sẽ chọn phần tử thứ hai trong danh sách, nghĩa là '1'.
}
    

-----------------------------------------------------------------------------------------------------------------------------
faces = detector.detectMultiScale(img_numpy)

-----------------------------------------------------------------------------------------------------------------------------
- etector.detectMultiScale(img_numpy): Sử dụng bộ phân loại cascade đã được tạo (detector) để phát hiện các khuôn mặt trong ảnh (img_numpy). Kết quả là một danh sách các hình chữ nhật (x, y, w, h) mô tả vị trí và kích thước của các khuôn mặt được phát hiện trong ảnh.
-Cuối cùng, biến id chứa ID của người trong ảnh, và biến faces chứa danh sách các khuôn mặt được phát hiện trong ảnh, mỗi khuôn mặt được mô tả bằng một hình chữ nhật (x, y, w, h).
-----------------------------------------------------------------------------------------------------------------------------

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids
-----------------------------------------------------------------------------------------------------------------------------
-Với mỗi hình chữ nhật mô tả khuôn mặt trong danh sách faces, đoạn mã này cắt ảnh từ img_numpy theo vị trí và kích thước của khuôn mặt, sau đó thêm ảnh đã cắt và ID tương ứng vào faceSamples và ids.
-img_numpy[y:y+h, x:x+w]: Đây là cách cắt ảnh từ mảng NumPy img_numpy. Đoạn mã này chọn phần của ảnh bắt đầu từ tọa độ (x, y) và kết thúc ở tọa độ (x+w, y+h). Điều này tạo ra một phần của ảnh chứa khuôn mặt, với chiều rộng là w và chiều cao là h.
Cuối cùng, đoạn mã trả về faceSamples và ids, chứa danh sách các mẫu khuôn mặt và các ID tương ứng của những người trong ảnh.
- faceSamples.append(...): Đoạn mã này thêm phần của ảnh chứa khuôn mặt vào danh sách faceSamples, để sau này có thể được sử dụng cho việc huấn luyện mô hình nhận dạng khuôn mặt.
Kết quả là faceSamples sẽ chứa danh sách các mẫu khuôn mặt được cắt ra từ ảnh gốc, mỗi mẫu khuôn mặt là một phần của ảnh chứa khuôn mặt đó.

-----------------------------------------------------------------------------------------------------------------------------
print ("\n Nó sẽ mất một vài giây. Chờ đợi ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml') 

print("\n  {0} khuôn mặt được đào tạo. Thoát khỏi chương trình".format(len(np.unique(ids))))

-----------------------------------------------------------------------------------------------------------------------------

getImagesAndLabels(path): Gọi hàm getImagesAndLabels() để lấy danh sách các mẫu khuôn mặt và các ID tương ứng từ thư mục path.
recognizer.train(faces, np.array(ids)): Sử dụng phương thức train() của mô hình nhận dạng khuôn mặt (recognizer) để huấn luyện mô hình với dữ liệu huấn luyện. Đối số faces là danh sách các mẫu khuôn mặt, và np.array(ids) là danh sách các ID tương ứng.
Trong đó :
 +faces: Danh sách các mẫu khuôn mặt, mỗi mẫu là một phần của ảnh chứa khuôn mặt.
 +np.array(ids): Danh sách các ID tương ứng với mỗi mẫu khuôn mặt. Biến này được chuyển đổi thành một mảng NumPy để có thể sử dụng trong quá trình huấn luyện

 Dòng mã này ghi mô hình nhận dạng khuôn mặt đã được huấn luyện vào một tệp YAML có tên là 'trainer.yml'.