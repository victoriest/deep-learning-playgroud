# deep-learning-playgroud-tf2

## 物体检测 yolov4

yolov4模型编译, 训练参照github项目: <https://github.com/AlexeyAB/darknet>中的readme进行

该实现有预训练模型参数, 并且可以读取主干网络的参数, 通过标记少量对错半对的图片样本(100张左右), 进行迁移学习, 可以解决我手头上样本量小的问题. 

#### 安装darknet

在linux下的安装很轻松, 直接进入darknet目录, 修改Makefile文件, make就行了

在安装过程中(win10环境下), 折腾了半天; 经验是不要用readme中的build.ps1文件安装. 正确的安装步骤:

	• Cuda10.1 安装时需要安装vs的插件
	• 安装opencv, 我安装的是4.4版本
	
下载解压缩到你想要安装的目录下。用文本编辑器（如Notepad++）打开darknet.vcxproj，用搜索功能查找CUDA （这里就是CUDA的版本号了），并修改为自己CUDA的版本号，我的CUDA版本是10.1，所以修改为CUDA 10.1（一共有两处需要修改的）并保存，

之后，将VS设置调成x64，Release

接下来就是配置项目属性：

VC++目录–>包含目录–>编辑  添加OpenCV路径：（请根据自己的路径添加）

D:\Softwares\opencv4.4.0\opencv\build\include

D:\Softwares\opencv4.4.0\opencv\build\include\opencv2

VC++目录–>库目录–>编辑  添加OpenCV路径：（请根据自己的路径添加）

D:\Softwares\opencv4.4.0\opencv\build\x64\vc15\lib

链接器->输入->附加依赖项  添加

opencv_world440.lib

ok，自此配置完成，直接点生成->生成解决方案即可，生成成功以后会在darknet-master\build\darknet\x64下生成一个darknet.exe文件。


#### 图片标注工具

Yolo_mark <https://github.com/AlexeyAB/Yolo_mark> 

LabelImg <https://github.com/tzutalin/labelImg.git>

#### 使用darknet yolov4

利用darknet的原生dll, so文件, 在python上调用

有个比较好用的darknet的python封装:<https://github.com/goktug97/PyYOLO>

经过改造后, api封装在 yolov4_detector.py中


## 不定长文本识别 CRNN

参照github项目: <https://github.com/YCG09/chinese_ocr>进行实践

其中CTPN部分, 在之前的git项目中有介绍.
 
文本识别部分, 改造为兼容tensorflow2

数据准备: 生成一些列手写数字图片(1-5)个. 代码在 data_pretreatment/muti_digit_train_data_generator.py

训练: variable_length_digit_htr/train.py

使用: variable_length_digit_htr/crnn_train.py



