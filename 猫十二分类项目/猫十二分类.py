#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""注意

该命令只需运行一次

该行命令用于解压缩，生成数据集、预训练模型、reader.py文件
"""

get_ipython().system('unzip data/data8136/cat_data_sets_models.zip')




# In[1]:


import os
import shutil
import paddle as paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import reader


# In[2]:


# 获取cats数据reader.train()
train_reader = paddle.batch(reader.train(), batch_size=16)
#test_reader = paddle.batch(reader.val(), batch_size=16)



# In[3]:


# 定义残差神经网络（ResNet）
def resnet50(input):
    def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1, act=None, name=None):
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=(filter_size - 1) // 2,
                                   groups=groups,
                                   act=None,
                                   param_attr=ParamAttr(name=name + "_weights"),
                                   bias_attr=False,
                                   name=name + '.conv2d.output.1')
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(input=conv,
                                       act=act,
                                       name=bn_name + '.output.1',
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance', )

    def shortcut(input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(input, num_filters, stride, name):
        conv0 = conv_bn_layer(input=input,
                              num_filters=num_filters,
                              filter_size=1,
                              act='relu',
                              name=name + "_branch2a")
        conv1 = conv_bn_layer(input=conv0,
                              num_filters=num_filters,
                              filter_size=3,
                              stride=stride,
                              act='relu',
                              name=name + "_branch2b")
        conv2 = conv_bn_layer(input=conv1,
                              num_filters=num_filters * 4,
                              filter_size=1,
                              act=None,
                              name=name + "_branch2c")

        short = shortcut(input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name + ".add.output.5")

    depth = [3, 4, 6, 3]
    num_filters = [64, 128, 256, 512]

    conv = conv_bn_layer(input=input, num_filters=64, filter_size=7, stride=2, act='relu', name="conv1")
    conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv_name = "res" + str(block + 2) + chr(97 + i)
            conv = bottleneck_block(input=conv,
                                    num_filters=num_filters[block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    name=conv_name)

    pool = fluid.layers.pool2d(input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    return pool


# In[5]:


image = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')


# In[6]:


# 获取分类器的上一层
pool = resnet50(image)
# 停止梯度下降
pool.stop_gradient = True
# 由这里创建一个基本的主程序
base_model_program = fluid.default_main_program().clone()

# 这里再重新加载网络的分类器，大小为本项目的分类大小
model = fluid.layers.fc(input=pool, size=12, act='softmax')


# In[7]:


# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-4)
opts = optimizer.minimize(avg_cost)

# 定义训练场所
place = fluid.CUDAPlace(0)#用GPU训练
#place = fluid.CPUPlace() #用CPU训练
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())


# In[8]:


# 官方提供的原预训练模型
src_pretrain_model_path = 'pretrained_models/ResNet50_pretrained/ResNet50_pretrained'


# 通过这个函数判断模型文件是否存在
def if_exist(var):
    path = os.path.join(src_pretrain_model_path, var.name)
    exist = os.path.exists(path)
    return exist
# 加载模型文件，只加载存在模型的模型文件
fluid.io.load_vars(executor=exe, dirname=src_pretrain_model_path, predicate=if_exist, main_program=base_model_program)


# In[9]:


# 优化内存
# optimized = fluid.transpiler.memory_optimize(input_program=fluid.default_main_program(), print_log=False)

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 训练10次
for pass_id in range(126):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        # 每60个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))


# In[10]:


# 保存参数模型
save_pretrain_model_path = 'models/step-1_model/'

# 删除旧的模型文件
shutil.rmtree(save_pretrain_model_path, ignore_errors=True)
# 创建保持模型文件目录
os.makedirs(save_pretrain_model_path)
# 保存参数模型
fluid.io.save_params(executor=exe, dirname=save_pretrain_model_path)


# In[1]:


import os
import shutil
import paddle as paddle
import paddle.dataset.flowers as flowers
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import reader


# In[2]:


# 获取cats数据
train_reader = paddle.batch(reader.train(), batch_size=16)
#test_reader = paddle.batch(reader.val(), batch_size=16)


# In[5]:


# 定义残差神经网络（ResNet）
def resnet50(input, class_dim):
    def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1, act=None, name=None):
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=(filter_size - 1) // 2,
                                   groups=groups,
                                   act=None,
                                   param_attr=ParamAttr(name=name + "_weights"),
                                   bias_attr=False,
                                   name=name + '.conv2d.output.1')
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(input=conv,
                                       act=act,
                                       name=bn_name + '.output.1',
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance', )

    def shortcut(input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(input, num_filters, stride, name):
        conv0 = conv_bn_layer(input=input,
                              num_filters=num_filters,
                              filter_size=1,
                              act='relu',
                              name=name + "_branch2a")
        conv1 = conv_bn_layer(input=conv0,
                              num_filters=num_filters,
                              filter_size=3,
                              stride=stride,
                              act='relu',
                              name=name + "_branch2b")
        conv2 = conv_bn_layer(input=conv1,
                              num_filters=num_filters * 4,
                              filter_size=1,
                              act=None,
                              name=name + "_branch2c")

        short = shortcut(input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name + ".add.output.5")

    depth = [3, 4, 6, 3]
    num_filters = [64, 128, 256, 512]

    conv = conv_bn_layer(input=input, num_filters=64, filter_size=7, stride=2, act='relu', name="conv1")
    conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv_name = "res" + str(block + 2) + chr(97 + i)
            conv = bottleneck_block(input=conv,
                                    num_filters=num_filters[block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    name=conv_name)

    pool = fluid.layers.pool2d(input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    output = fluid.layers.fc(input=pool, size=class_dim, act='softmax')
    return output


# In[6]:


# 定义输入层
image = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取分类器
model = resnet50(image,12)

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=9e-5)
opts = optimizer.minimize(avg_cost)


# 定义一个使用GPU的执行器
place = fluid.CUDAPlace(0)
#place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())


# In[7]:


# 经过step-1处理后的的预训练模型
pretrained_model_path = 'models/step-1_model/'

# 加载经过处理的模型
fluid.io.load_params(executor=exe, dirname=pretrained_model_path)


# In[8]:


# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 训练10次
for pass_id in range(100):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        # 每60个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    ## 进行测试
    #test_accs = []
    #test_costs = []
    #for batch_id, data in enumerate(test_reader()):
     #   test_cost, test_acc = exe.run(program=test_program,
      #                                feed=feeder.feed(data),
       #                               fetch_list=[avg_cost, acc])
        #test_accs.append(test_acc[0])
        #test_costs.append(test_cost[0])
    # #求测试结果的平均值
    #test_cost = (sum(test_costs) / len(test_costs))
    #test_acc = (sum(test_accs) / len(test_accs))
    #print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))


# In[9]:


# 保存预测模型
save_path = 'models/step_2_model/'

# 删除旧的模型文件
shutil.rmtree(save_path, ignore_errors=True)
# 创建保持模型文件目录
os.makedirs(save_path)
# 保存预测模型
fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], target_vars=[model], executor=exe)




# In[10]:


# 在生成模型文件（model）后运行

SAVE_DIRNAME = './models/step_2_model'  # 保存好的 inference model 的路径




"""
运行generate_CSV_file_with_infer.py
脚本文件generate_CSV_file_with_infer.py实现生成CSV文件的功能
脚本文件将读取训练好的模型（model）和测试集数据（test）
并将模型对测试集数据的预测结果保存为CSV文件
"""

# coding:utf-8
# from __future__ import print_function
import os
import json

import paddle
import paddle.fluid as fluid
import numpy as np
from PIL import Image
import sys



TOP_K = 1


DATA_DIM = 224

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

# 下面行代码保存时的写法匹配
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model( SAVE_DIRNAME, exe,
                # model_filename='model',
                # params_filename='params'
                # model_filename = 'fc_0.w_0',
                # params_filename = 'params'
                )


def real_infer_one_img(im):
    infer_result = exe.run(
        inference_program,
        feed={feed_target_names[0]: im},
        fetch_list=fetch_targets)

    # print(infer_result)
    # 打印预测结果
    mini_batch_result = np.argsort(infer_result)  # 找出可能性最大的列标，升序排列
    # print(mini_batch_result.shape)
    mini_batch_result = mini_batch_result[0][:, -TOP_K:]  # 把这些列标拿出来
    # print('预测结果：%s' % mini_batch_result)
    # 打印真实结果
    # label = np.array(test_y)  # 转化为 label
    # print('真实结果：%s' % label)
    mini_batch_result = mini_batch_result.flatten() #拉平
    mini_batch_result = mini_batch_result[::-1] #逆序
    return mini_batch_result


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

def process_image(img_path):
    img = Image.open(img_path)
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=DATA_DIM, center=True)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype(np.float32).transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    img = np.expand_dims(img, axis=0)
    return img


def convert_list(my_list):
    my_list = list(my_list)
    my_list = map(lambda x:str(x), my_list)
    # print('_'.join(my_list))
    return '_'.join(my_list)


def infer(file_path):
    im = process_image(file_path)
    result = real_infer_one_img(im)
    result = convert_list(result)
    return result




def createCSVFile(cat_12_test_path):
    lines = []

    # 获取所有的文件名
    img_paths = os.listdir(cat_12_test_path)
    for file_name in img_paths:
        file_name = file_name
        file_abs_path = os.path.join(cat_12_test_path, file_name)
        result_classes = infer(file_abs_path)

        file_predict_classes = result_classes

        line = '%s,%s\n'%(file_name, file_predict_classes)
        lines.append(line)

    with open('csv_file_path.csv', 'w') as f:
        f.writelines(lines)


abs_path = r'./data_sets/cat_12/cat_12_test' # cat_12_test 文件夹的真实路径
createCSVFile(abs_path)


# In[11]:


get_ipython().system('rm -rf submit.sh')
get_ipython().system('wget -O submit.sh http://ai-studio-static.bj.bcebos.com/script/submit.sh')
get_ipython().system('sh submit.sh csv_file_path.csv 86e7908d13da43c5afe26ace13a095e3')


# In[ ]:




