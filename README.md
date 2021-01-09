# KhaoManee
***KhaoManee*** is a face-detection and face-recognition API that provides face bounding box and face-landmarks for face-detection. KhaoManee's face-detection model uses ***MTCNN*** model that returns 5 points face-landmark. KhaoManee's face-recognition API can recognize the member of Parliament in Thailand, but it can detect only a few people such as ***Prayut Chan-o-cha***, ***Prawit Wongsuwan***, ***Thanathorn Juangroongruangkit***. I will add more in the future. If you have any person who is a member of Parliament in Thailand that you are interested in. You can tell me by creating an issue. I will consider it and add it. KhaoManee's face-recognition use ***Facenet*** model to find similarity with embedding vector (For this project, I use vector size is 512). Last, this project implements the model with Pytorch and  API with FastAPI. If you have any problems or any questions,  you can tell me by creating issue :pray:

### :pushpin: Requirement

1. python 3.7.x or Higher

2. torch 1.5 or higher

3. torchvision 0.8.2 or higher

4. fastapi 0.63.0 or higher

5. uvicorn 0.13.3 or higher

6. python-multipart 0.0.5 or higher

7. requests 2.25.1 or higher

### :gear: How to use

1. Clone this project.
    ```
    git clone https://github.com/RTae/KhaoManee.git
    ```

2. Run this line to install package,
    ```
    pip install -r reqguirements.txt
    ```

3. Run this line to run FastAPI in local.
    ```
    uvicorn main:app --reload
    ```

4. [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to see the document of this api. Done !!!

### :bust_in_silhouette: How to add more face to recognize
    You can
1. Create a folder with the name that you want to recognize.

    ```
    ./data/imges/<your_folder_with_name>
    ```

    ***example*** /data/imges/prayuth

2. Add images of the person that you want to  recognize at least one imges (recommed to be      straight face picture).

3. Run this line to add.
    ```
    python add_face_rec_images.py
    ```

4. Run this line to test it
    ```
    uvicorn main:app --reload
    ```

### :whale: Docker Build

***Build***
```
    docker build -t myimage .
```

***Run***
```
    docker container run -d --name mycontainer -p 80:80 myimage
```

or from my build

```
    docker container run -d --name mycontainer -p 80:80 rtae/khaomanee:1.0.5
```

### :fire: Demo


### :blue_book: Reference
1. timesler's facenet+mtcnn repo [repository](https://github.com/timesler/facenet-pytorch/tree/dd0b0e4b5b124b599f75b87e570910e5d80c8848#the-fastmtcnn-algorithm) (Big Thank you to this repo :pray: :pray:)

2. F. Schroff, D. Kalenichenko, J. Philbin. FaceNet: A Unified Embedding for Face Recognition and Clustering, arXiv:1503.03832, 2015. [PDF](https://arxiv.org/pdf/1503.03832.pdf)

3. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. VGGFace2: A dataset for recognising face across pose and age, International Conference on Automatic Face and Gesture Recognition, 2018. [PDF](https://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf)

4. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks, IEEE Signal Processing Letters, 2016. [PDF](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)

5. FastAPI [link](https://fastapi.tiangolo.com)