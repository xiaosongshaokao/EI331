import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
# 这个代码用来训练和测试字符识别模型
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphbets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z']
chinese = ['zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin',
                'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong',
                'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun',
                'zh_zang', 'zh_zhe']

# 定义了神经网络的类
class char_cnn_net:
    def __init__(self):
        self.dataset = numbers + alphbets + chinese #所有需要识别的字符集合
        self.dataset_len = len(self.dataset)
        self.img_size = 20
        self.y_size = len(self.dataset)
        self.batch_size = 100

        self.x_place = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size, self.img_size], name='x_place')
        self.y_place = tf.placeholder(dtype=tf.float32, shape=[None, self.y_size], name='y_place')
        self.keep_place = tf.placeholder(dtype=tf.float32, name='keep_place')

    def cnn_construct(self):
        x_input = tf.reshape(self.x_place, shape=[-1, 20, 20, 1])

        cw1 = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01), dtype=tf.float32)
        cb1 = tf.Variable(tf.random_normal(shape=[32]), dtype=tf.float32)
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_input,filter=cw1,strides=[1,1,1,1],padding='SAME'),cb1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_place)

        cw2 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 64], stddev=0.01), dtype=tf.float32)
        cb2 = tf.Variable(tf.random_normal(shape=[64]), dtype=tf.float32)
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1,filter=cw2,strides=[1,1,1,1],padding='SAME'),cb2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_place)

        cw3 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=0.01), dtype=tf.float32)
        cb3 = tf.Variable(tf.random_normal(shape=[128]), dtype=tf.float32)
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2,filter=cw3,strides=[1,1,1,1],padding='SAME'),cb3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_place)

        conv_out = tf.reshape(conv3, shape=[-1, 3 * 3 * 128])

        fw1 = tf.Variable(tf.random_normal(shape=[3 * 3 * 128, 1024], stddev=0.01), dtype=tf.float32)
        fb1 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully1 = tf.nn.relu(tf.add(tf.matmul(conv_out, fw1), fb1))
        fully1 = tf.nn.dropout(fully1, self.keep_place)

        fw2 = tf.Variable(tf.random_normal(shape=[1024, 1024], stddev=0.01), dtype=tf.float32)
        fb2 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully2 = tf.nn.relu(tf.add(tf.matmul(fully1, fw2), fb2))
        fully2 = tf.nn.dropout(fully2, self.keep_place)

        fw3 = tf.Variable(tf.random_normal(shape=[1024, self.dataset_len], stddev=0.01), dtype=tf.float32)
        fb3 = tf.Variable(tf.random_normal(shape=[self.dataset_len]), dtype=tf.float32)
        fully3 = tf.add(tf.matmul(fully2, fw3), fb3, name='out_put')

        return fully3
    #训练模型函数
    def train(self,data_dir,save_model_path):
        print('ready load train dataset')
        X, y = self.init_data(data_dir)
        print('success load' + str(len(y)) + 'datas')
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0) # 将原始数据按照比例分割为“测试集”和“训练集”

        out_put = self.cnn_construct()
        predicts = tf.nn.softmax(out_put)
        predicts = tf.argmax(predicts, axis=1)
        actual_y = tf.argmax(self.y_place, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, actual_y), dtype=tf.float32))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_put, labels=self.y_place))
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        train_step = opt.minimize(cost)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            saver = tf.train.Saver()
            while True:
                train_index = np.random.choice(len(train_x), self.batch_size, replace=False)
                train_randx = train_x[train_index]
                train_randy = train_y[train_index]
                _, loss = sess.run([train_step, cost],
                                   feed_dict={self.x_place:train_randx,self.y_place:train_randy,self.keep_place:0.75})
                step += 1

                if step % 10 == 0:
                    test_index = np.random.choice(len(test_x), self.batch_size, replace=False)
                    test_randx = test_x[test_index]
                    test_randy = test_y[test_index]
                    acc = sess.run(accuracy,feed_dict={self.x_place : test_randx, self.y_place : test_randy,
                                                       self.keep_place : 1.0})
                    print(step, loss)
                    if step % 50 == 0:
                        print('accuracy:' + str(acc))
                    if step % 500 == 0:
                        saver.save(sess, save_model_path, global_step=step)
                    if acc > 0.99 and step > 500:
                        saver.save(sess, save_model_path, global_step=step)
                        break

    def test(self,x_images,model_path):
        text_list = []
        out_put = self.cnn_construct()
        predicts = tf.nn.softmax(out_put)
        predicts = tf.argmax(predicts, axis=1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            module_file = tf.train.latest_checkpoint(model_path)
            saver.restore(sess, module_file)
            preds = sess.run(predicts, feed_dict={self.x_place: x_images, self.keep_place: 1.0})
            for i in range(len(preds)):
                pred = preds[i].astype(int)
                text_list.append(self.dataset[pred])
            return text_list

    def list_all_files(self,root):
        files = []
        list = os.listdir(root)
        for i in range(len(list)):
            element = os.path.join(root, list[i])
            if os.path.isdir(element):
                temp_dir = os.path.split(element)[-1]
                if temp_dir in self.dataset:
                    files.extend(self.list_all_files(element))
            elif os.path.isfile(element):
                files.append(element)
        return files
    # 初始化数据
    def init_data(self,dir):
        X = []
        y = []
        if not os.path.exists(data_dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(dir)

        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)#以灰度图的形式读取图像
            if src_img.ndim == 3:# 如果出错则跳过
                continue
            resize_img = cv2.resize(src_img, (20, 20))
            X.append(resize_img)
            # 获取图片文件全目录
            dir = os.path.dirname(file)
            # 获取图片文件上一级目录名
            dir_name = os.path.split(dir)[-1]
            vector_y = [0 for i in range(len(self.dataset))]# 这在干嘛
            index_y = self.dataset.index(dir_name)
            vector_y[index_y] = 1
            y.append(vector_y)

        X = np.array(X)
        y = np.array(y).reshape(-1, self.dataset_len)
        return X, y

    def init_testData(self,dir):
        test_X = []
        if not os.path.exists(test_dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(test_dir)
        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            if src_img.ndim == 3:
                continue
            resize_img = cv2.resize(src_img, (20, 20))
            test_X.append(resize_img)
        test_X = np.array(test_X)
        return test_X

# 主函数
if __name__ == '__main__':
    cur_dir = sys.path[0] #读取当前目录
    data_dir = os.path.join(cur_dir, 'carIdentityData\cnn_char_train') # 进行拼接得到数据目录
    # print(data_dir)
    test_dir = os.path.join(cur_dir, 'carIdentityData\cnn_char_test') # 拼接得到测试目录（这里数据和测试是一样的，单纯拷贝而已，会有点影响）
    train_model_path = os.path.join(cur_dir, 'carIdentityData\model\char_recongnize\model.ckpt')# 存放模型的目录
    model_path = os.path.join(cur_dir,'carIdentityData\model\char_recongnize')# 存放模型的目录

    train_flag = 0 # 为0时测试模型准确率，为1时进行模型训练
    net = char_cnn_net() # 初始化cnn神经网络

    if train_flag == 1:
        # 训练模型
        net.train(data_dir,train_model_path)
    else:
        # 测试部分
        test_X = net.init_testData(test_dir)
        print(test_X)
        text = net.test(test_X,model_path)
        print(text)
