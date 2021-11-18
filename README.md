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
![image](https://user-images.githubusercontent.com/84729271/142440613-ba2e6ed7-db6d-469b-a1c6-2452d136a127.png)
![image](https://user-images.githubusercontent.com/84729271/142440637-fa79e25d-510c-4965-a10d-fc332536155c.png)


②Feature matching: after feature matching the two images to be matched after feature detection, the matching point pairs are as follows：
![image](https://user-images.githubusercontent.com/84729271/142440814-b44c6278-5619-4c9b-9b29-533ebd58ee22.png)


③Image affine transformation and image stitching, and the final stitching result is shown in the figure below：
![image](https://user-images.githubusercontent.com/84729271/142440923-1938e6f2-2720-49e9-996d-e663b0d601d3.png)

