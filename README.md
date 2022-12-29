# CapstoneDesign_Closet-fairy

## 인공지능기반 가상 피팅 시스템

> 분반 : 세종대학교 캡스톤디자인 컴공 001반 <br>
> 팀명 : 옷장의 요정 <br>
> 팀원 : 이진수, 김희재, 박형준, 오준석
<br>

-------

<br>

### 프로젝트 설명
- 사용자의 전신사진과 옷 이미지를 가상 피팅 인공지능에게 제공.
- 학습된 인공지능을 통해 옷이 입혀진 이미지를 사용자에게 출력
- AWS의 다양한 서비스를 탐색 및 활용

<br>

### 인공지능 모델
- ### **[CP-VTON plus](https://minar09.github.io/cpvtonplus/)** <br>

    CP-VTON plus는 <br>
의상의 기하변환을 수행하는  ***GMM(Geometric Matching Module)*** 과 <br>
변환된 옷을 인체 이미지와 합성하는 ***TOM(Try-On Module)*** 으로 구성된 모델이다.<br>
최종적으로 TOM을 통해 합성된 이미지를 얻기 위해서는 <br>
    1. 사람 이미지
    2. 사람 mask
    3. 사람 parse
    4. 사람 pose
    5. 옷 이미지
    6. 옷 mask

    <br>
    6개의 데이터를 필요로 한다. 이는 VITON dataset에서 모두 제공한다.
    
    <br>
기존의 CP-VTON plus모델은 위와 같은 데이터가 모두 [VITON 데이터셋](#데이터셋)에서 제공되지만,<br>
우리가 서비스 하고자 하는 상황에서는 1번과 5번, 즉 사람과 옷 이미지 외의 데이터는 미리 가지고 있기 힘들다.<br>   따라서 1번 이미지로 부터 2. 3. 4. 번의 이미지를, 5번 이미지로 부터 6번 이미지를 생성할 수 있도록 하였다. <br>
이를 위해 사용한 모델과 기법은 아래와 같다.<br>

- ### [Graphonomy: Universal Human Parsing via Graph Transfer Learning](https://openaccess.thecvf.com/content_CVPR_2019/html/Gong_Graphonomy_Universal_Human_Parsing_via_Graph_Transfer_Learning_CVPR_2019_paper.html)
    - 사람 이미지를 팔, 다리, 상체 등으로 Parsing 하는 모델로,<br>
    이 모델을 통해 2. 3. 번 데이터를 구한다.

<br>

- ### [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1812.08008)
    - 사람 이미지의 Pose를 추정하는 모델로,<br>
    이 모델을 통해 4.번 데이터를 구한다.

<br>    

- ### [OpenCV Threshold 함수](https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html)
     - 이미지를 특정 임계치를 기준으로 이진화 하는 함수로,<br>
     이 함수를 통해 6번 데이터를 구한다.


<br>

### 데이터셋
- **VITON** 



