# deep-learning-playgroud

## 手写数字识别
* 使用简单CNN模型, mnist数据集, tensorflow框架进行训练

* 使用了更复杂的模型, 在kaggle上mnist的数据集acc为99.7%的模型
  
	<https://www.kaggle.com/c/digit-recognizer>

	<https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6>


* 使用了VGG16模型
  
	利用keras改写VGG16经典模型在手写数字识别体中的应用<https://www.cnblogs.com/LHWorldBlog/p/8677131.html>

	**发现VGG16的input需要是224*224*3, 大材小用了**

* 随后 使用了CRNN模型
	参考了文章:[Recurrent Convolutional Neural Network For Svhn](https://jimlee4530.github.io/Recurrent%20Convolutional%20Neural%20Network%20for%20SVHN)
	和github项目:<https://github.com/JimLee4530/RCNN>

	得到了不错的泛化效果 (单手写数字识别)

* 该过程中, 从0学习了keras, tensorflow的使用, 参考链接如下:

	<https://keras-zh.readthedocs.io/>

	[keras官方中文文档](https://keras.io/zh/)

	[【干货】史上最全的Tensorflow学习资源汇总](https://zhuanlan.zhihu.com/p/35515805)
	
	[python – Keras：如何保存模型并继续培训？](https://codeday.me/bug/20180921/257413.html)

	[Keras ModelCheckpoint 保存训练过程中的最佳模型权重](https://blog.csdn.net/qq_27871973/article/details/84955977)

	- [Keras保存最好的模型](https://www.jianshu.com/p/0711f9e54dd2)

	- [如何为Keras中的深度学习模型建立Checkpoint](https://cloud.tencent.com/developer/article/1049579)

    	- [How to Check-Point Deep Learning Models in Keras](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)

	[How to model Convolutional recurrent network ( CRNN ) in Keras](https://stackoverflow.com/questions/48356464/how-to-model-convolutional-recurrent-network-crnn-in-keras?rq=1)

* 了解数据归一化的重要性: 对于同一个模型的收敛数据归一化后, 的收敛速度快很多

	[机器学习——标准化/归一化的目的、作用和场景](https://blog.csdn.net/zenghaitao0128/article/details/78361038)

* 模型加载, 保存, 多gpu利用等tip, 只保存权重的好处是, 模型文件体积变小

	[python – Keras：如何保存模型并继续培训？](https://codeday.me/bug/20180921/257413.html)

	[Keras ModelCheckpoint 保存训练过程中的最佳模型权重](https://blog.csdn.net/qq_27871973/article/details/84955977)

	[Keras保存最好的模型](https://www.jianshu.com/p/0711f9e54dd2)

	[如何为Keras中的深度学习模型建立Checkpoint](https://cloud.tencent.com/developer/article/1049579)
    	- [How to Check-Point Deep Learning Models in Keras](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)

	[Tensorflow加载预训练模型和保存模型](https://yq.aliyun.com/articles/567023)

* 为了达到python离线训练模型，Java在线预测的功能
 
	[Java调用Keras、Tensorflow模型](https://www.jianshu.com/p/0016a34c82c8)

	[将keras的h5模型转换为tensorflow的pb模型](https://blog.csdn.net/u010159842/article/details/84481478)

* 框架对于多GPU资源的使用方式:

	[Keras同时用多张显卡训练网络](https://www.jianshu.com/p/db0ba022936f)
    	- [官方文档：multi_gpu_model](https://keras.io/utils/#multi_gpu_model)

	[多gpu训练，单gpu保存(多gpu下训练的model在单gpu下测试出错)](https://github.com/YCG09/chinese_ocr/issues/94)
    	- [keras 多GPU训练，单GPU预测](https://www.codeleading.com/article/231257812/)

	[Keras多GPU训练以及载入权重无效的问题](https://blog.csdn.net/DumpDoctorWang/article/details/84099022)

	[Keras官方文档 - 如何在 GPU 上运行 Keras?](https://keras.io/zh/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus)

	[Tensorflow官方文档 - 使用GPU](https://www.tensorflow.org/guide/using_gpu)

* Keras的图片预处理
[Keras的官方文档 - 图片生成器ImageDataGenerator](https://keras-cn.readthedocs.io/en/latest/preprocessing/image/)

* 通过将训练数据转化为npz格式的文件提高加载效率:
<https://github.com/victoriest/deep-learning-playgroud/blob/master/handwrite_digit_ocr/npz_util.py>

* Keras训练模型时用到的generator, 目前来说依旧没有玩转:
	[A detailed example of how to use data generators with Keras](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)

	[keras 两种训练模型方式fit和fit_generator(节省内存)](https://blog.csdn.net/u011311291/article/details/79900060)
    	- [keras数据自动生成器，继承keras.utils.Sequence，结合fit_generator实现节约内存训练](https://blog.csdn.net/u011311291/article/details/80991330)

[How to use Keras fit and fit_generator (a hands-on tutorial)](https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/)


* 2019-07-22 日更新: 手写英文字母识别
    可以通过使用RCNN的模型的少许修改, 用EMNIST数据集训练以及测试, 准确率在95%.
    
    修改内容: 将输出Y维度, 从10改为26. 即, 26个英文字母, 不分大小写.
     
    [EMNIST数据集](https://www.nist.gov/node/1298471/emnist-dataset)

    [EMNIST的数据集加载lib](https://pypi.org/project/emnist/)


* 2019-07-26 日更新: 手写英文单词短语识别

    参考github项目:
    <https://github.com/githubharald/SimpleHTR>
    
    使用其中的模型识别模型, 其中为了让识别出来的结果更加准确(纠正拼写错误, 如book 识别成了buuk), 需要加入解码器[CTCWordBeamSearch](https://github.com/githubharald/CTCWordBeamSearch).
    
    结果发现该解码器需要支持tensorflow的自定义操作, 而自定义操作不能在windows平台下使用. 所以需要找替代方案: 
    
    [partten](https://www.clips.uantwerpen.be/pages/pattern), 一个基于python的自然语言处理工具包. 
    
    使用过程中可能会遇到网络问题导致, 报错:
    ```
    zipfile.BadZipFile: File is not a zip file
    ```
    找到该文件下载的路径,删除 重新下载即可.
    
    为了识别短语, 加入分词算法:<https://github.com/githubharald/WordSegmentation>
    
    另外为了在flask里加载多个keras模型, 总是报错:
    
    ```
    ValueError: Tensor Tensor is not an element of this graph
    ```
    
    强制将Flask改为单线程模式就行了
    
    ```
    if __name__ == '__main__':
        app.run(host="0.0.0.0", port=8080, threaded=False)
    ```
    
    或者直接使用生产级的WSGI容器. 
    
    参考文章:
    * [Handwriting Recognition using Tensorflow](https://medium.com/@moshnoi2000/handwritting-recognition-using-tensorflow-aaf84fa9c587)
    
    * [Handwriting recognition using Tensorflow and Keras](https://towardsdatascience.com/handwriting-recognition-using-tensorflow-and-keras-819b36148fe5)
    
    * [FAQ: Build a Handwritten Text Recognition System using TensorFlow](https://towardsdatascience.com/faq-build-a-handwritten-text-recognition-system-using-tensorflow-27648fb18519)
    
    * [Handwriting OCR: handwriting recognition and language modeling with MXNet Gluon](https://medium.com/apache-mxnet/handwriting-ocr-handwriting-recognition-and-language-modeling-with-mxnet-gluon-4c7165788c67)
    
    * [Beam Search Decoding in CTC-trained Neural Networks](https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7)
    
    * <https://github.com/githubharald/CTCDecoder>
    
    * [Word Beam Search: A CTC Decoding Algorithm](https://towardsdatascience.com/word-beam-search-a-ctc-decoding-algorithm-b051d28f3d2e)
    

## 文档边缘检测
目标是从照片识别出文档区域, 进行了两个模型的训练以及测试
#### 路径1:
根据github项目<https://github.com/senliuy/Keras_HED_with_model>和<https://github.com/lc82111/Keras_HED>进行实践, 
	1. 下载训练数据：<http://vcl.ucsd.edu/hed/HED-BSDS.tar> 并解压到工程根目录下
	2. 下载预训练模型：<https://github.com/fchollet/deep-learning-models/releases> 中搜索文件’vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5‘，下载并拷贝到./models目录下


#### 路径2:
根据这篇文章:[深度学习实践文档检测](https://zhuanlan.zhihu.com/p/56336225), 以及github项目<https://github.com/RRanddom/tf_doc_localisation>进行的相关的文档边缘检测实践.

## 不定长文本识别
参照github项目: <https://github.com/YCG09/chinese_ocr>进行实践
#### 其中遇到的坑:
原型工程:
ctpn: <https://github.com/eragonruan/text-detection-ctpn>
chinese_ocr: <https://github.com/YCG09/chinese_ocr>

首先下载两个工程, 并按照readme安装相关依赖的python库

我们的首要使用的工程是chinese_ocr, 
在该工程里, 有一个ctpn的目录, 该目录是一个cptn的模型, 这个比较麻烦, 重点讲这里:

在windows环境中, 需要使用c编译两个(三个?)库. 
其目录在cptn/lib/utils中, 在linux环境下, 直接make.sh就可以了, 但是在windows下我们需要如下N步:
需要在命令行下, 进入该目录:
```
cython bbox.pyx
cython cython_nms.pyx
Cython nms.pyx
cython gpu_nms.pyx(GPU可选)

Python setup.py build_ext--inplace
```

不出意料的话会报错:
这时候就需要ctpn的工程下同目录的setup.py了, 
``` python
from distutils.core import setup

Import numpy as np
From Cython.Build import cythonize

numpy_include=np.get_include()
#setup(ext_modules=cythonize("bbox.pyx"),include_dirs=[numpy_include])
setup(ext_modules=cythonize("cython_nms.pyx"),include_dirs=[numpy_include])
```

把编译好的东西拷贝到ctpn的工程的utils目录下
哦对了 你会遇到这个问题:
```
"ValueError: Buffer dtype mismatch, expected 'int_t' but got 'long long'" for sample_with_gt_wrapper 
```
改成 intp_t重新编译即可
<https://github.com/CharlesShang/FastMaskRCNN/issues/163>

参考连接:
与CPTN（文字识别网络）作斗争的记录
来自 <https://www.jianshu.com/p/027e9399e699> 

win10+tensorflow CPU 部署CTPN环境
来自 <https://blog.csdn.net/u010554381/article/details/86519960> 

文本识别text-detection-ctpn环境搭建
来自 <https://blog.csdn.net/qq_35513792/article/details/89174958> 

<https://github.com/Li-Ming-Fan/OCR-DETECTION-CTPN>
<https://github.com/eragonruan/text-detection-ctpn/issues/73>



## 参考文档

#### 参考的git项目:
<https://github.com/qjadud1994/CRNN-Keras>
<https://github.com/xiaofengShi/CHINESE-OCR>
<https://github.com/sbillburg/CRNN-with-STN>
<https://github.com/eragonruan/text-detection-ctpn>

#### 常用的模型预训练数据的github项目
<https://github.com/fchollet/deep-learning-models/releases>

#### Tensorflow各种官方的预训练模型 - TensorFlow-Slim image classification model library
<https://github.com/tensorflow/models/tree/1af55e018eebce03fb61bba9959a04672536107d/research/slim>

#### 对比许多模型在多个数据集中的测试效果的表格 - What is the class of this image ?
<http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html>

#### 数据集

[深度学习开放数据集](https://deeplearning4j.org/cn/opendata#%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%BC%80%E6%94%BE%E6%95%B0%E6%8D%AE%E9%9B%86)

[数据集大全：25个深度学习的开放数据集](https://zhuanlan.zhihu.com/p/35399323)

[CASIA Online and Offline Chinese Handwriting Databases](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html)

