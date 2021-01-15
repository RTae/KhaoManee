import numpy as np
import cv2
from PIL import Image, ImageDraw
import io

def detect(img, mtcnn, landmark_state='false', data='false'):
    color = (255, 136, 75) # Orange

    # Convent btye to string
    nparr = np.fromstring(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Covert cv2 to pil
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Detect with mtcnn and return bounding boxes and landmarks
    boxes, prob, landmarks = mtcnn.detect(img, landmarks=True)
    
    if boxes is not None:
        if data.lower() == 'true':
            if landmark_state.lower() == 'true':
                return {
                    'bounding_boxes': [ [ int(b) for b in box ] for box in boxes ],
                    'landmarks': [ [ [ int(p) for p in points] for points in landmark] for landmark in landmarks ],
                }
            else:
                return {
                    'bounding_boxes': [ [ int(b) for b in box ] for box in boxes ],
                }

        draw = ImageDraw.Draw(img)
        for idx in range(len(boxes)):
            # Draw Rectangle
            draw.rectangle(boxes[idx].tolist(), outline=color, width=3)
            if landmark_state.lower() == 'true':
                # Draw Landmarks
                for landmark in landmarks[idx]:
                    r = 3
                    x, y = landmark
                    leftUpPoint = (x-r, y-r)
                    rightDownPoint = (x+r, y+r)
                    twoPointList = [leftUpPoint, rightDownPoint]
                    draw.ellipse(twoPointList, fill=color)
        
        # Convert int to byte
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return img_byte_arr
    
    else:
        if data.lower() == 'true':
            if landmark_state.lower() == 'true':
                return {
                    'bounding_boxes': [],
                    'landmarks': []
                }
            else:
                return {
                    'bounding_boxes': []
                }
        else :
            return None