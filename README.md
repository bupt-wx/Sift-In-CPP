# Sift-In-CPP
This is SIFT feature detection and matching implemented in C + +

Environment version information：
VS2017、Opencv3.4.3

Reference link：

1.https://blog.csdn.net/maweifei/article/details/58227605
2.https://blog.csdn.net/tfliu2/article/details/81613303

The algorithm is mainly divided into the following steps，The example picture is shown in the figure below：

![6](https://user-images.githubusercontent.com/84729271/142439566-a3606a37-d1ea-4c53-a6fe-2871faafbb78.jpg)
![7](https://user-images.githubusercontent.com/84729271/142439645-d9d67e6a-7e04-498c-a43f-1abdab7367f9.jpg)


①For feature detection, SIFT feature detection is performed on the sample picture, and the results are shown in the figure below：
<img width="225" alt="src_with_point" src="https://user-images.githubusercontent.com/84729271/142442183-3391d57d-2eb4-47cf-afda-02a63f1d9dea.png">
<img width="225" alt="src2_with_point" src="https://user-images.githubusercontent.com/84729271/142442197-c7eb6602-e6e8-4d34-b233-9d147c6907b1.png">



②Feature matching: after feature matching the two images to be matched after feature detection, the matching point pairs are as follows：
![image](https://user-images.githubusercontent.com/84729271/142440814-b44c6278-5619-4c9b-9b29-533ebd58ee22.png)


③Image affine transformation and image stitching, and the final stitching result is shown in the figure below：
![image](https://user-images.githubusercontent.com/84729271/142440923-1938e6f2-2720-49e9-996d-e663b0d601d3.png)

