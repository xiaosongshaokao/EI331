## EI331 Final Project ##

Car Plate Detect

### 代码食用说明 ###

数据集文件没传，克隆到本地后加进去就行。可以配合pycharm git操作更为方便

- .gitignore文件已经设置好把那个比较大的数据集文件夹忽略了，上传时可放心上传代码

- 关于pycharm和git的联动使用：
  - 当然首先你得下载一个git
  - 设置里面有个version control，把自己的github账号密码配好，再把git的安装目录告诉pycharm
  - 这样配置基本就完成了，然后可以从VSC那里选择克隆项目到本地，把本项目的地址粘贴放进pycharm就行
  - 关于代码的上传：首先commit，再push（记住这个顺序就ok）
  - 如果有几个人同时更改导致版本变动，可能要merge选择一个版本
 
其他问题大家可以在群里讨论或者自行探索一下

### 报错说明 ###

- 文件路径可能不对

- 训练模型的名字也可能不一样

- opencv版本请保持在4.0以上

- 第27行```c = sorted(contours, key=cv2.contourArea, reverse=True)[1]```一句在跑某些图片的时候会报错

Git测试
