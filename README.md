# KhaoManee
This is KhaoManee

### :pushpin: Requirement

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

### :blue_book: Reference
1. timesler's facenet+mtcnn repo [repository](https://github.com/timesler/facenet-pytorch/tree/dd0b0e4b5b124b599f75b87e570910e5d80c8848#the-fastmtcnn-algorithm)

2. F. Schroff, D. Kalenichenko, J. Philbin. FaceNet: A Unified Embedding for Face Recognition and Clustering, arXiv:1503.03832, 2015. [PDF](https://arxiv.org/pdf/1503.03832.pdf)

3. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. VGGFace2: A dataset for recognising face across pose and age, International Conference on Automatic Face and Gesture Recognition, 2018. [PDF](https://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf)

4. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks, IEEE Signal Processing Letters, 2016. [PDF](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)

5. FastAPI [link](https://fastapi.tiangolo.com)