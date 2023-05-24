import cv2
import os

for i in ["start","stop","left","right"]:
    
# load the video using path of video ( my video length is 37 sec )
    video_path = f"./SVM-and-KNN-Classifier/data/{i}_crop.mp4"
    video = cv2.VideoCapture(video_path)

    success = True
    count = 1
    image_id = 1

    while success:
        success , frame = video.read()
        
        if success == True:
            
            if count%7 == 0:
                name = os.path.join(f"./SVM-and-KNN-Classifier/data/{i}" ,str(image_id)+ f"2-{i}" +".jpg")
                image_id += 1
                cv2.imwrite(name,frame)
            
            count += 1
        else:
            break

    print("Total Extracted Frames :",image_id)   