import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 1024) 
cam.set(4, 768) 

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = input('\n nhập id người dùng cuối < 1 - n >:  ')
print("\n Đang khởi tạo tính năng chụp khuôn mặt. Nhìn vào camera và chờ đợi ...")
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
        cv2.imshow('Camera dataset', img)
    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    elif count >= 50:
         break

print("\n Thoát khỏi chương trình")
cam.release()
cv2.destroyAllWindows()


