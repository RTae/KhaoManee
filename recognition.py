from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import cv2
import json
import io

def recognition(img, mtcnn, resnet, device, data='false'):
    color = (255, 136, 75) # Orange
    fnt = ImageFont.truetype("./data/font/THSarabunNew.ttf", 30)

    # Convent btye to string
    nparr = np.fromstring(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Covert cv2 to pil
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    names = []
    embeddings = []

    # Load embed vector that prepare
    with open('./data/vector512.txt') as json_file:
        weight_vector_dicts = json.load(json_file)
        for w in weight_vector_dicts:
            names.append(w['name'])
            embeddings.append( [ float(v)for v in w['vector']] ) 
    embeddings = torch.tensor(embeddings)

    # Detect bb from mtcc model
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        faces = mtcnn.extract(img, boxes, save_path=None)

        # Extract Feture
        aligned_face = torch.stack([faces]).to(device)
        embeddings_face = resnet(aligned_face[0]).detach().cpu()

        # Find Euclidean distance 
        dists = [[(e1 - e2).norm().item() for e2 in embeddings_face] for e1 in embeddings]

        faces_rec_list = []
        for idx, bb in enumerate(boxes):
            temp_index = -1
            temp_lowest_dist = 1
            for index, dist in enumerate(dists):
                if dist[idx] < temp_lowest_dist:
                    temp_lowest_dist = dist[idx]
                    temp_index = index
            
            if temp_index == -1:
                continue
            else :
                temp = {
                    'name': names[temp_index],
                    'bounding_box': [int(b) for b in bb],
                }
                faces_rec_list.append(temp)

        if data.lower() == 'false':

            draw = ImageDraw.Draw(img)
            for idx in range(len(faces_rec_list)):
                # Draw Rectangle
                point = faces_rec_list[idx]['bounding_box']
                text = faces_rec_list[idx]['name']
                draw.rectangle(point, outline=color, width=3)

                text_size = fnt.getsize(text.capitalize())

                draw.rectangle((point[0] ,point[1] - text_size[1] + 5 ,point[0] + text_size[0] ,point[1]), fill=color)
                draw.text((point[0] ,point[1] - text_size[1]), text.capitalize(), font = fnt, fill=(255,255,255))

            # Convert int to byte
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            return img_byte_arr

        else :
            return faces_rec_list
    
    else:
        if data.lower() == 'true':
            return {
                'name': '',
                'bounding_boxes': []
            }
        else :
            return None
