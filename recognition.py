from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import cv2
import json
import io

def recognition(img, mtcnn, resnet, device, annoy, data='false'):
    color = (255, 136, 75) # Orange
    fnt = ImageFont.truetype("./data/font/THSarabunNew.ttf", 30)

    # Convent btye to string
    nparr = np.fromstring(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Covert cv2 to pil
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    with open('./data/map_dict_name.txt') as file:
        dict_name = json.load(file)


    # Detect bb from mtcc model
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        faces = mtcnn.extract(img, boxes, save_path=None)

        # Extract Feture
        aligned_face = torch.stack([faces]).to(device)
        embeddings_faces = resnet(aligned_face[0]).detach().cpu()
        embeddings_faces = embeddings_faces.numpy()
        
        # Find Euclidean distance 
        faces_rec_list = []
        for idx, vector in enumerate(embeddings_faces):
            index_face, dist_face = annoy.get_nns_by_vector(vector, 1, include_distances = True)
            if dist_face[0] < 1:
                name = dict_name[str(index_face[0])]
                temp = {
                    'name': name,
                    'bounding_box': [int(b) for b in boxes[idx]]
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
