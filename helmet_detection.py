import cv2
from ultralytics import YOLO

model = YOLO("helmet.pt")

img=cv2.imread("helmet.jfif")

result=model(img)
print(result)

count=0
for r in result:
    for box in r.boxes:
        cls=int(box.cls[0])
        label=model.names[cls]
        conf=float(box.conf[0])

        if conf > 0.5:

           count+=1
        x,y,w,h=map(int,box.xyxy[0])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(img,f"{label}{count}{conf:2f}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)


cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()