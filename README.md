# CapstoneDesign_Closet-fairy

## 인공지능기반 가상 피팅 시스템

> 분반 : 세종대학교 캡스톤디자인 컴공 001반 <br>
> 팀명 : 옷장의 요정 <br>
> 팀원 : 이진수, 김희재, 박형준, 오준석
>> 팀원 별 역할 :<br>
이진수 : 인공지능 모델 총괄 <br>
김희재 : 팀장, 데이터베이스 구축 <br>
박형준 : DB와의 연동, 홈페이지 프론트엔드<br>
오준석 : 프론트 엔드와 벡엔드, 딥러닝 서버간의 상호작용
-------

<br>

### 프로젝트 설명
- 쇼핑몰의 의류를 사용자의 전신사진을 통해 가상으로 입어볼 수 있는 서비스 개발.
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

||||
|:---|:----|:----|
|1. 사람 이미지<br><img src="readme img/human.jpg">|2. 사람 mask<br><img src="readme img/human_mask.png">|3. 사람 parse<br><img src="readme img/human_parse.png">|
|4. 사람 pose<br><img src="readme img/human_pose.png">|5. 옷 이미지<br><img src="readme img/dress.jpg">|6. 옷 mask<br><img src="readme img/dress_mask.jpg">|


6개의 데이터를 필요로 한다. 이는 VITON dataset에서 모두 제공한다.<br>
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
- **VITON** <br>
VITON 데이터셋은 Try On task를 위한 데이터셋이다.<br>
14,221쌍의 train set, 2032쌍의 test set으로 구성되어있다.
각 이미지 셋은 위에서 언급한 6개의 데이터로 구성되어있다.<br>

||||
|:---|:----|:----|
|1. 사람 이미지<br><img src="readme img/human.jpg">|2. 사람 mask<br><img src="readme img/human_mask.png">|3. 사람 parse<br><img src="readme img/human_parse.png">|
|4. 사람 pose<br><img src="readme img/human_pose.png">|5. 옷 이미지<br><img src="readme img/dress.jpg">|6. 옷 mask<br><img src="readme img/dress_mask.jpg">|

<br><br>

### 소개영상
[![유튜브 이미지](http://img.youtube.com/vi/VHm2lB1ET0Y/0.jpg)](https://youtu.be/VHm2lB1ET0Y)<br>
https://www.youtube.com/watch?v=VHm2lB1ET0Y

<br>
<br>

### 과제 보고서

|||
|---|---|
|요구사항 분석서|[링크](https://github.com/LeeJinSoo-BIN/CapstoneDesign_Closet-fairy/blob/master/readme%20img/%EC%9A%94%EA%B5%AC%EC%82%AC%ED%95%AD%20%EB%B6%84%EC%84%9D%EC%84%9C.pdf)|
|수정 과제 계획서|[링크](https://github.com/LeeJinSoo-BIN/CapstoneDesign_Closet-fairy/blob/master/readme%20img/%EC%88%98%EC%A0%95%20%EA%B3%BC%EC%A0%9C%20%EA%B3%84%ED%9A%8D%EC%84%9C.pdf)|
|소프트웨어 설계서|[링크](https://github.com/LeeJinSoo-BIN/CapstoneDesign_Closet-fairy/blob/master/readme%20img/%EC%86%8C%ED%94%84%ED%8A%B8%EC%9B%A8%EC%96%B4%20%EC%84%A4%EA%B3%84%EC%84%9C.pdf)|
|최종 보고서|[링크](https://github.com/LeeJinSoo-BIN/CapstoneDesign_Closet-fairy/blob/master/readme%20img/%EC%B5%9C%EC%A2%85%20%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf)|


<br><br><br><br>

|| tool |
| ------ | ------ |
| 개발언어 | ![issue badge](https://img.shields.io/badge/Python-3.9-blue.svg) ![issue badge](https://img.shields.io/badge/javascript-blue.svg) |
| 데이터베이스 | ![issue badge](https://img.shields.io/badge/AWS-grey.svg) ![issue badge](https://img.shields.io/badge/DynamoDB-grey.svg) ![issue badge](https://img.shields.io/badge/Python-3.9-lightgrey.svg)|
| 웹 페이지 UI | ![issue badge](https://img.shields.io/badge/HTML-5-green.svg) ![issue badge](https://img.shields.io/badge/CSS-gray.svg) ![issue badge](https://img.shields.io/badge/Flask-gray.svg) ![issue badge](https://img.shields.io/badge/Bootstrap-gray.svg)  |
| 모델 서버 | ![issue badge](https://img.shields.io/badge/CP%20VTON-plus-green.svg) ![issue badge](https://img.shields.io/badge/pytorch-1.10.8+cu108-green.svg) ![issue badge](https://img.shields.io/badge/Flask-gray.svg)|
| 모델 개발 환경 | AMD Pyzen Threadripper PRO 3000WX <br> CUDA Version: 10.8 <br> NVIDIA RTX A5000 24GB |
| 개발환경 | Windows10 64bit <br> Ubuntu 18.04.2 LTS |