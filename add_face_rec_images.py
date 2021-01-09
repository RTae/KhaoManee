from models.inception_resnet_v1 import InceptionResnetV1
from models.mtcnn import MTCNN
from torchvision import datasets
from tqdm import tqdm
import torch
import json

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define Model
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Extrac fetrue from images
dataset = datasets.ImageFolder('./data/images')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}

faces_aligned = []
names = []
for x, y in tqdm(dataset):
    box, _ = mtcnn.detect(x)
    faces = mtcnn.extract(x, box, save_path=None)
    if faces is not None:
        faces_aligned.append(faces)
        names.append(dataset.idx_to_class[y])

faces_aligned = torch.stack(faces_aligned).to(device)
embeddings = resnet(faces_aligned).detach().cpu()

# Save the weight
weight_vector_dict = [ {'name': name, 'vector': [ str(v) for v in v512]} for name, v512 in zip(names,embeddings.numpy())]

with open('./data/vector512.txt', 'w') as outfile:
    json.dump(weight_vector_dict, outfile)
