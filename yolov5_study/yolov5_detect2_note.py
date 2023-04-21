# 参考博客
# https://blog.csdn.net/weixin_47872288/article/details/127705212
# https://blog.csdn.net/weixin_69398563/article/details/126378699

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

# 如果要写ROS节点，需要导入rospy。
# std_msgs.msg的目的是可以使用std_msgs/String消息类型来发布  
import rospy                           # Python版本的ROS客户端库，提供了Python程序需要的接口（rospy就是一个Python模块） 有关于node、topic、service、param、time相关操作
from std_msgs.msg import String        # File: std_msgs/String.msg。 这意味着我们message的格式为String

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

#一开始的文件都是导入包名,后面需要验证导入的包路径正确，使用如下的代码进行验证

# 获取该文件的绝对路径：/home/zqc/projects/yolov5/detect.py
FILE = Path(__file__).resolve()   

#获取yolov5下的根本路径: /home/zqc/projects/yolov5
ROOT = FILE.parents[0]   # YOLOv5 root directory  

# 查询路径的列表是否在内，如果不在内则添加
if str(ROOT) not in sys.path:  
    sys.path.append(str(ROOT))  # add ROOT to PATH

# 将其绝对路径转换为相对路径
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative   



# 必须加这句代码，不然后面的from xxx import xxx中的import cv2会报错
# 具体原因见https://blog.csdn.net/chengmo123/article/details/112969309  
# 我们使用第二种方法
# 通过这行代码可以把ROS写入path中的路径给清除，进行可以import anaconda中的cv2包。
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


# 对应run函数主要内容如下：
    # 判断命令行传入的参数以及进行处理
    # 新建保存模型的文件夹
    # 下载模型权重
    # 加载带预测的图片
    # 执行模型推理过程，产生结果
    # 打印最终的信息

# ==========================================================================================================================================================
# ====================================================================1.1 传入、处理参数======================================================================
# ==========================================================================================================================================================

# 函数装饰器 这里相当于run = smart_inference_mode(run)；
# smart_inference_mode()来自/utils/torch_utils.py 
# 如果torch>=1.9.0则应用torch.inference_mode()装饰器，否则使用torch.no_grad()装饰器
@smart_inference_mode()     
def run(                    # def 函数名(参数): 函数体                             # run()参数有默认值，但是实际上没用，有用的是parse_opt()的参数默认值default
        weights=ROOT / 'yolov5n.pt',  # model path or triton URL                    # 模型权重或路径          
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)          # 图像路径
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path                       # coco数据集
        imgsz=(640, 640),  # inference size (height, width)                         # 图像尺寸
        conf_thres=0.65,  # 0.25 confidence threshold                               # 置信度                 
        iou_thres=0.65,  # 0.45NMS IOU threshold                                    # NMS IOU 大小              
        max_det=1000,  # maximum detections per image                               # 每张图像的最大检测量
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu                        # cuda 为0 或者 0 1 2 3 或者 cpu，代码有所判断  没有GPU，是不是应该默认device='cpu'
        view_img=True,  # show results                                              # 展示图片结果                
        save_txt=True,  # save results to *.txt                                     # 将结果保存到*.txt       
        save_conf=True,  # save confidences in --save-txt labels                    # 保存置信度到txt标签中     
        save_crop=True,  # save cropped prediction boxes                            # 保存裁剪的预测框     
        nosave=False,  # do not save images/videos                                  # 不保存图片或者视频        
        classes=None,  # filter by class: --class 0, or --class 0 2 3               # 按照类别筛选，0 或者0 1 2
        agnostic_nms=False,  # class-agnostic NMS                                   # class-agnostic NMS
        augment=False,  # augmented inference                                       # 增强推理
        visualize=False,  # visualize features                                      # 可视化功能
        update=False,  # update all models                                          # 更新所有模型
        project=ROOT / 'runs/detect',  # save results to project/name               # 将结果保存到project/name 
        name='exp',  # save results to project/name                                 # 将结果保存到project/name 
        exist_ok=False,  # existing project/name ok, do not increment               # 现有项目名称（不要添加）
        line_thickness=1,  # bounding box thickness (pixels)                        # 边界框像素                  
        hide_labels=True,  # hide labels                                            # hide labels             
        hide_conf=True,  # hide confidences                                         # hide confidences          
        half=False,  # use FP16 half-precision inference                            # 使用FP16半精度推理
        dnn=False,  # use OpenCV DNN for ONNX inference                             # 使用OpenCV DNN进行ONNX推理
        vid_stride=1,  # video frame-rate stride                                    # 视频流的帧步
):
    # run()参数有默认值，但是实际上没用，有用的是parse_opt()的参数默认值default
    # parse_opt()中很多都是False

    # 这里的8行代码是江国来博士添加的
    # 前6行都是参数设置，
    source = 2       # 指的是摄像头2，不是默认值source=ROOT / 'data/images'
    view_img=True    # 展示图片结果。这里其实没意义，后面1.4中if webcam中又重新判定了view_img，虽然结果还是view_img=True。
    save_conf=True   # 保存置信度到.txt，这里其实没意义，因为save_txt=False
    line_thickness=3 # 边界框像素，不重要。
    classes = 0      # 只检测人
    conf_thres=0.65  # 置信度阈值0.65

    # 后2行是ROS需要的
    # https://www.jianshu.com/p/3ff246a1edd9
    # 在python语言中，ROS发布者定义格式如下：
    # pub1 = rospy.Publisher(“/topic_name”, message_type, queue_size=size)
    # 声明发布者类的一个对象pub1，注意pub1不是发布者节点名称，而是class publisher的一个对象（objec）t
    # /topic_name表示发布者节点向这个topic发布消息。message_type表示发布到话题中消息的类型
    # 发布者节点（与订阅者节点类似）使用rosrun或roslaunch命令来运行
    # 发布者节点向主节点注册发布者节点名称publisher_noda_name、话题名称topic_name、消息类型message_type、URI地址和端口

    # 现在我们创建发布者类的一个对象（objec），发布topic，topic名称为humandetectioncamera2， 发布消息的格式是String， 发布的最大数量queue_size;    
    # humandetectioncamera2这个名字可以自定义，但是不能包含‘/’符号
    # 之前有from std_msgs.msg import String；msg文件是用于话题的消息文件，扩展名为*.msg。这种msg文件只包含一个字段类型和一个字段名称
    # 如果任何订阅者都没有足够快地接收到消息，queue_size将会限制队列消息的数量。 形象的比喻：假设Publisher依次发送了100个数据了，此时才有Subscriber开始接收，那么他只能从第90个开始接收，之前的内容都已经被丢弃了。
    pub = rospy.Publisher('humandetectioncamera2', String, queue_size=10)   
    # 初始化ROS节点  
    # 告诉rospy你的ros节点名字 “humandetectioncamera2_node”，将注册到master上; anonymous=True保证进程名的唯一性，如果出现重名，会通过添加后缀方式避免    
    rospy.init_node('humandetectioncamera2_node', anonymous=True)
    
    # 可以设置发布消息的频率
    # rate = rospy.Rate(10)  
    

    # 此行代码将其source 转换字符串
	# source 为命令行传入的图片或者视频，大致为：python detect.py --source data/images/bus.jpg
    # 既然函数体中有source = 2  ，那就应该是：python detect.py --source 2
    source = str(source)
    # print("source:", source)              # source: 2

	# 是否保存预测后的图片
	# parse_opt()中nosave为false，则not nosave为true
	# source传入的参数为jpg而不是txt，则not source.endswith('.txt')为true
	# 最后则表示需要存储最后的预测结果
    # 所以只要传入的文件地址不是以.txt结尾 就都是要保存预测后的图片的
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # print("save_img:", save_img)          # save_img: True

    # 判断该文件名后缀是否在(IMG_FORMATS + VID_FORMATS) 该列表内。
	# Path(source)：为文件地址，data/images/bus.jpg; # 既然函数体中有source = 2 ，那例子是什么？
	# suffix[1:]：截取文件后缀，即为bus.jpg，而[1:]则为jpg后，最后输出为jpg
    # 该列表的IMG_FORMATS、VID_FORMATS来源于/utils/dataloaders.py     from utils.dataloaders import IMG_FORMATS, VID_FORMATS
    # 图片后缀 IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
    # 视频后缀 VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # print("is_file:", is_file)            # is_file: False

    # 判断是否为网络流地址或者是网络的图片地址
	# 将其地址转换为小写，并且判断开头是否包括如下网络流开头的
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # print("is_url:", is_url)              # is_url: False

	# 是否是使用webcam 网页数据，一般为false 
	# 判断source字符串是否只由数字组成（0、2为摄像头路径）或者txt文件 或者 网络流并且不是文件地址
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)   # github源码中是or source.endswith('.streams') or ，不是('.txt')
    # print("webcam:", webcam)              # webcam: True

    # 是否传入的为屏幕快照文件
    screenshot = source.lower().startswith('screen')
    # print("screenshot:", screenshot)      # screenshot: False
       
    # print("source:", source)              # source: 2
    # print("save_img:", save_img)          # save_img: True
    # print("is_file:", is_file)            # is_file: False
    # print("is_url:", is_url)              # is_url: False
    # print("webcam:", webcam)              # webcam: True
    # print("screenshot:", screenshot)      # screenshot: False


    # 疑问：这里是不是解析url，返回https地址？？？https://blog.csdn.net/weixin_47872288/article/details/127705212
    # 如果是网络流地址以及文件名后缀在(IMG_FORMATS + VID_FORMATS) 列表内，则对应下载该文件，source的值被修改
    if is_url and is_file:        
        source = check_file(source)  # download  
    # 我们的source被设置为0或2;is_url和is_file都是False,不运行


# ==================================
    # 总结，生成了几个比较重要的变量（用于后续的if判断？？？）：
    # source = 2       # 指的是摄像头2
    # save_img: True   # 保存预测后的图片
    # webcam=True      # 使用webcam 网页数据
    # save_txt=False   # 将结果保存到*.txt 
    # 
# ==================================

# ==========================================================================================================================================================
# ====================================================================1.2 新建文件夹======================================================================
# ==========================================================================================================================================================
    # Directories

    # 将原先传入的名字扩展成新的save_dir 如runs/detect/exp存在 就扩展成 runs/detect/exp1。每次执行detect代码模块，生成的文件夹会进行增量表示（exp序号增量显示）
    # 参数project有默认值：project=ROOT / 'runs/detect' ;    Path(project) 即获取打开的项目的路径。/home/zqc/projects/yolov5/runs/detect
    # 参数name有默认值：   name='exp'               # Path(project) / name表示两者的拼接 ：runs/detect/exp
    # 参数exist_ok有默认值：exist_ok=False，表示需要新建; exist_ok=exist_ok 前者是increment_path()的形参，后者是run()的形参
    # 检查当前Path(project) / name是否存在，如果存在就新建新的save_dir，在原有名字的基础上进行增量
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # print("save_dir:", save_dir) # print结果应该是/home/zqc/projects/yolov5/runs/detect/expn  n可以为None、2、3、4、5...
     
    # 传入的命令参数save_txt 为false，则直接创建exp文件夹/home/zqc/projects/yolov5/runs/detect/expn
    # 传入的命令参数save_txt 为true，则拼接一个/ 'labels 创建文件夹，/home/zqc/projects/yolov5/runs/detect/expn/'labels'
    # 参数save_txt是默认值save_txt=False, 
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # 这个'labels'是哪里定义的？？？ 之前只有hide_labels=True？？？

# ==================================
    # 总结：
    # save_dir只是一个Path对象，内容是/home/zqc/projects/yolov5/runs/detect/expn
    # save_dir.mkdir(parents=True, exist_ok=True)才是新建文件夹命令
# ==================================



# ==========================================================================================================================================================
# ====================================================================1.3 模型加载======================================================================
# ==========================================================================================================================================================

    # Load model

    # 选择CPU或者GPU，主要为逻辑判断（此处代码就不post出）
    # 此处的device 为 None or 'cpu' or 0 or '0' or '0,1,2,3'
    device = select_device(device)   # 默认device='0'，但是我们没有GPU，是不是应该默认device=''或device='cpu'

    # YOLOv5 MultiBackend 类，用于各种后端的 python 推理
    # 模型后端框架，传入对应的参数
    # from models.common import DetectMultiBackend
    # 以下参数都在一开始初始化中定义了
    # run()参数虽然有默认值，但是实际上没用，有用的是parse_opt()的参数默认值default
    # weights=ROOT / 'yolov5s.pt'; device='';  dnn=False; data=ROOT / 'data/coco128.yaml'; half=False
    # 可以print试试
    # print("weights:", weights)            
    # print("device:", device)        
    # print("dnn:", dnn)            
    # print("data:", data)              
    # print("half:", half)              
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    
    # 补充1：    
    # dnn:OpenCV中的深度学习模块（DNN）只提供了推理功能，不涉及模型的训练，
    # DNN支持多种深度学习框架，比如TensorFlow，Torch和Darknet。支持的网络结构涵盖了常用的目标分类，目标检测和图像分割的类别
    # OpenCV DNN模块只支持图像和视频的深度学习推理。它不支持微调和训练。不过，OpenCV DNN模块可以作为任何初学者进入基于深度学习的计算机视觉领域的一个完美起点。

    # 补充2：
    # /models/common.py中的class DetectMultiBackend，其可以根据weights参数的具体文件后缀，判断使用的是什么框架，
    # 参考： https://blog.csdn.net/weixin_47872288/article/details/127705212
    # PyTorch:   weights = *.pt
    # ONNX Runtime:      *.onnx
    # TensorFlow GraphDef:    *.pb

    #加载完模型之后，对应读取模型的步长、类别名、pytorch模型类型
    stride, names, pt = model.stride, model.names, model.pt
    # 可以print试试
    # print("stride:", stride)            
    # print("names:", names)        
    # print("pt:", pt)            

    # 判断模型步长是否为32的倍数
    # https://blog.csdn.net/sp7414/article/details/116781653
    # yolo v5 设置的 img_size 并不会影响任意尺寸图像的检测，这个数值设置的目的是使输入图像先被 resize 成 640×640，满足检测网络结构，最后再 resize 成原始图像尺寸，进行显示。
    imgsz = check_img_size(imgsz, s=stride)  # check image size

# ==================================
    # 疑问：
    # DetectMultiBackend()可以获取weights对应的模型信息？？？？？？？
# ==================================


# ==========================================================================================================================================================
# ====================================================================1.4 加载带预测的图片====================================================================
# ==========================================================================================================================================================


    # Dataloader
    bs = 1  # batch_size
    
    # from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
    # 流加载器，类似代码 python detect.py——source 'rtsp://example.com/media.mp4'    
	# 1.1中 webcam定义处：source字符串只由数字组成，则webcam 为True, 使用webcam 网页数据 
    if webcam: 
        # 检查一下环境是否可以使用opencv.imshow显示图片。 主要有两点限制: cv2.imshow()不能在docker环境中使用;cv2.imshow()也不能在和Google Colab环境中使用
        view_img = check_imshow(warn=True)  # from utils.general import (xxx, check_imshow,xxx) 
        # 加载流 可以加载网络摄像头甚至Youtube中的视频链接
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)   # from utils.dataloaders import  LoadStreams
        bs = len(dataset)     # batch_size
    elif screenshot:  # 是否传入的为屏幕快照文件; 屏幕截图，加载带预测图
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)     # from utils.dataloaders import  LoadScreenshots
    else:     # 加载图
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) # from utils.dataloaders import  LoadImages
    # 保存路径
    # 每个batch_size的vid_path与vide_writer 二维数组 初始化为None
    vid_path, vid_writer = [None] * bs, [None] * bs


# ==================================
    # 总结：
    # LoadStreams的变量： 
        # source                  # 2
        # img_size=imgsz          # imgsz=(640, 640),在check_img_size之后变成list，[640, 640]
        # stride=stride           # stride= model.stride  模型的步长
        # auto=pt                 # pt = model.pt  pytorch模型类型
        # vid_stride=vid_stride   # 默认vid_stride=1 视频流的帧步

    # 疑问
    # dataset = LoadStreams()
    # bs = len(dataset)
    # LoadStreams()的返回值是什么呢？字符、列表、元组？？？我没从dataloaders.py中看明白
    # https://blog.csdn.net/weixin_47872288/article/details/127705212
    # 看后续代码 for path, im, im0s, vid_cap, s in dataset:
    # 说明dataset中有很多参数。
    # 看dataloaders.py中class LoadStreams:  def __next__(self):  return self.sources, im, im0, None, ''

# ==================================



# ==========================================================================================================================================================
# ===================================================================  1.5 执行推理模型====================================================================
# ==========================================================================================================================================================

    # Run inference

    # 通过运行一次推理来预热模型（内部初始化一张空白图预热模型）
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup   传入一张图片，让GPU先热身一下

    # seen是计数的功能，dt用来存储时间，
    # from utils.general import (xxx, Profile, xxx)
    # 用法：@Profile() decorator装饰器 or 'with Profile():上下文管理器 
    # with dt[0]: 没有实际作用，不用管
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())  
    
    # dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) 
    # dataloaders.py中class LoadStreams:  def __next__(self):  return self.sources, im, im0, None, ''
    # dataset数据集遍历，
    # path为图片或视频的路径？？？是url吗
    # im为resize后的图片
    # im0s为原图
    # vid_cap 空；当读取的图片为None时，读取视频为视频源
    # s 打印图片的信息
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:  # 没有实际作用，不用管
            # numpy array to tensor and device
            # torch.from_numpy()的作用是将生成的数组转换为张量
    	    # 在模型中运算，需要转换成pytorch，从numpy转成pytorch
    	    # 在将其数据放入到cpu 或者 gpu中
            im = torch.from_numpy(im).to(model.device)
            # 半精度训练 uint8 to fp16/32
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 归一化
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 图片为3维(RGB)，在前面添加一个维度，batch_size=1。本身输入网络的图片需要是4维， [batch_size, channel, w, h]
            # 【1，3，640，480】
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim  缺少batch这个尺寸，所以将它扩充一下，变成[1，3,640,480]

        # Inference 
        # 进行推理
        # visualize 一开始为false，如果为true则对应会保存一些特征
        with dt[1]:
            # 可视化文件路径 调用utils/general.py文件中的increment_path类增加文件或目录路径
            # 这里由于parse_opt():参数默认为visualize=False，实际上不执行increment_path，依然visualize = False
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # augment推理时是否进行多尺度、翻转（TTA）推理
            # 数据的推断增强，但也会降低速度。最后检测出的结果为18900个框
            # 结果为【1，18900，85】，预训练有85个预测信息，4个坐标 + 1个置信度 +80各类别
            pred = model(im, augment=augment, visualize=visualize)

        # NMS 
        # 非极大值阈值过滤
	    # conf_thres: 置信度阈值；iou_thres: iou阈值
        # classes: 是否只保留特定的类别 默认为None.    # classes = 0;只检测人 
        # agnostic_nms: 进行nms是否也去除不同类别之间的框 默认False   # agnostic_nms=False;
        # max_det: 每张图片的最大目标个数 默认1000，超过1000就会过滤   # max_det=1000 # 每张图像的最大检测量
        # pred: [1,num_obj, 6] = [1,5, 6]   这里的预测信息pred还是相对于 img_size(640) 。本身一开始18900变为了5个框，6为每个框的 x左右y左右 以及 置信度 类别值
        with dt[2]:
            # non_max_suppression的Returns: list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            # xyxy：指的是框的左上角坐标(x,y)和右下角坐标(x,y), 一共四个量
            # xywh：指的是框的中心点坐标(x,y)和框的宽度width和height, 一共四个量
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
       
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 如果节点关闭就终止当前for循环
        if rospy.is_shutdown():   # 检查rospy.is_shutdown标志位, rospy.is_shutdown()的意思是只要节点关闭就返回true
            break     # 终止当前for循环                 
        

        # Process predictions 
        # 把所有的检测框画到原图中
        # 对每张图片进行处理，将pred(相对img_size 640)映射回原图img0 size
        # 此处的det 表示5个检测框中的信息
        for i, det in enumerate(pred):  # per image     i：每个batch的信息，det:表示5个检测框的信息
            # 每处理一张图片，就会加1
            seen += 1
            # 输入源是网页，对应取出dataset中的一张照片
            if webcam:  
                # 如果输入源是webcam，则batch_size>=1 取出dataset中的一张图片
                p, im0, frame = path[i], im0s[i].copy(), dataset.count  # path为图片路径  im0s为原图，1080 * 810？？？   dataset数据集遍历
                s += f'{i}: '  # s 打印图片的信息
            else:
                # 但是大部分情况都是从LoadImages流读取本地文件中的照片或者视频 所以batch_size=1
                # p为当前图片或者视频绝对路径
    	        # im0原始图片
    	        # frame: 初始为0  可能是当前图片属于视频中的第几帧
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path 
            # 图片的保存路径
            # Path().name 目录的最后一个部分 # 返回当前路径中的文件名+文件后缀   
            # save_dir:.../runs/detect/expn
            # save_path：.../runs/detect/expn/...
            save_path = str(save_dir / p.name)  # im.jpg   
            # txt 保存路径（保存预测框的坐标），每张图片对应一个框坐标信息
            # Path().stem：目录最后一个部分，没有后缀。
            # txt_path：.../runs/detect/expn/labels/...
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt  默认不存

            # 输出信息，图片shape (w, h)
            s += '%gx%g ' % im.shape[2:]  # print string
            # gn = [w, h, w, h]  用于后面的归一化
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh 归一化增益  获得原图的宽和高的大小
            
            # imgpath没啥用，后面没用到，可以注释掉
            imgpath = str(txt_path+'.jpg')  
            # 标志位，是否发现人.我感觉没啥用
            findhuman = 0                

            # print(frame)

            # imc: for save_crop 在save_crop中使用 ，
            # 默认save_crop=false,所以 imc = im0 # 原图
            imc = im0.copy() if save_crop else im0  # for save_crop  是否要将检测的物体进行裁剪
            # 绘图工具，画图检测框的粗细，一种通过PIL，一种通过CV2处理
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 定义绘图工具

            if len(det): # det:表示5个检测框的信息
                # Rescale boxes from img_size to im0 size
                # 将预测信息（相对img_size 640）映射回原图 img0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()    #scale_coords坐标映射功能

                # Print results
                # 统计每个框的类别
                for c in det[:, 5].unique():  # np.unique()返回参数数组中所有不同的值，并按照从小到大排序
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                findhuman = 1 # ???????这里为什么要赋值为1？？？？？？？

                # Write results
                # 保存预测信息: txt、img0上画框、crop_img
                for *xyxy, conf, cls in reversed(det):  # reversed(det)反转det的顺序
                    print("find human at")
                    # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化 转为list再保存
                    # 其实，如果下面用xyxy(左上角 + 右下角)，那么这里计算xywh(中心的 + 宽高)没用，后面xywh用不到了
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 
                    # line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    line = (cls, *xyxy, conf) if save_conf else (cls, *xyxy) # label format
                    humandata = ('%g ' * len(line)).rstrip() % line + '\n'
                    print(humandata)
                    
                    # 之前创建了publisher类的一个对象pub    pub = rospy.Publisher('humandetectioncamera2', String, queue_size=10)
                    # class publisher类中，有一个函数def publish(self, *args, **kwds):发布信息到对应话题
                    pub.publish(humandata)  # 发布信息humandata到对象pub的话题'humandetectioncamera2'
                    
                    # save_txt：False
                    # save_img:True
                    # save_crop:False
                    # view_img=True
 
                    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id+score+xywh
                    if save_txt:  # Write to file  
                        with open(f'{txt_path}.txt', 'a') as f:
                            # .rstrip 去除后面的空白字符串
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # 在原图上画框 + 将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # 如果需要就将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            #if findhuman:
                #print(imgpath)
                #cv2.imwrite(imgpath, imc) #save a copy of image

            
            # save_txt：False
            # save_img:True
            # save_crop:False
            # view_img=True

            # Stream results
            #返回画好的图片
            im0 = annotator.result()  # im0为绘制好的图片
            if view_img:              # 使用webcam时, 如果可以使用opencv.imshow显示图片,则view_img为True
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # 通过imshow显示出框
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 是否需要保存图片或视频（检测后的图片/视频 里面已经被我们画好了框的） img0
            if save_img:
                # 若为图片
                if dataset.mode == 'image':     
                    # 向路径中保存图片
                    cv2.imwrite(save_path, im0)
                # 是视频或者流
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

# ==========================================================================================================================================================
# ===========================================================================1.6 打印信息====================================================================
# ==========================================================================================================================================================

    # Print results
    # seen为预测图片总数，dt为耗时时间，求出平均时间
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    # 保存预测的label信息 xywh等   save_txt
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        # strip_optimizer函数将optimizer从ckpt中删除  更新模型
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    # 最后将结果输出在控制台，detect模块到此结束。






# ==========================================================================================================================================================
# ===========================================================================解析命令行格式下的参数====================================================================
# ==========================================================================================================================================================

def parse_opt():   # 传入的参数，以上的参数为命令行赋值的参数，如果没有给定该参数值，会有一个default的默认值进行赋值
    parser = argparse.ArgumentParser()   # 创建解析器
    # argparse.ArgumentParser()用法解析 https://www.cnblogs.com/yibeimingyue/p/13800159.html 

    # argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。argparse模块的作用是用于解析命令行参数。
    # 我们很多时候，需要用到解析命令行参数的程序，目的是在终端窗口(ubuntu是终端窗口，windows是命令行窗口)输入训练的参数和选项。

    # 我们常常可以把argparse的使用简化成下面四个步骤
    # 1：import argparse                     # 首先导入该模块；
    # 2：parser = argparse.ArgumentParser()  # 然后创建一个解析对象；
    # 3：parser.add_argument()               # 然后向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项；
    # 4：parser.parse_args()                 # 最后调用parse_args()方法进行解析；解析成功之后即可使用。

    
    # 添加参数 
      # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
      # 每个参数解释：
      # name or flags:字符串的名字或者列表。默认第一个参数
      # nargs：应该读取的命令行参数个数
      # type：命令行参数应该被转换成的类型
      # default：
      # action:当参数在命令行中出现时使用的动作
      # help：参数的帮助信息。相当于注释
      
    # 相关参数解释 
    # https://blog.csdn.net/weixin_42645636/article/details/128775467

    # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，会打印这些描述信息。
    
    # “action='store_true'”说明 
    # https://blog.csdn.net/qq_45708837/article/details/128383032
    # 这个类型的参数和之前的有很大区别，大家可以把他理解成一个“开关”。根据参数是否设置默认值，可以分为两种情况。
    # 例如，parser.add_argument('--view-img', action='store_true', help='show results')，在检测的时候系统要把我检测的结果实时的显示出来，
    # python detect.py --view-img -> view-img为True
    # python detect.py -> view-img为False
    # 假如我文件夹有5张图片，如果指定了这个参数的话，那么模型每检测出一张就会显示出一张，直到所有图片检测完成。如果我不指定这个参数，那么模型就不会一张一张的显示出来。
    
    # nargs='+'说明  
    # https://blog.csdn.net/weixin_49148527/article/details/125200095
    # nargs='*'： 表示参数可设置零个或多个
    # nargs='+'：表示参数可设置一个或多个
    # nargs='?'： 表示参数可设置零个或一个
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # 可以把“0”赋值给“classes”，也可以把“0”“2”“4”“6”都赋值给“classes”


    # weights: 指定网络权重路径,可以使用自己训练的权重,也可以使用官网提供的权重。
    # source： 指定网络输入的路径，默认指定的是文件夹，也可以指定具体的文件或者扩展名等
    # data: 配置文件路径, 包括image/label/classes等数据集基本信息, 在训练时如果不自己指定数据集，系统会自己下载coco128数据集. 训练自己的文件, 需要作相应更改
    # imgsz: 网络输入图片大小, 默认的大小是640.模型在检测图片前会把图片resize成640的size，然后再喂进网络里，并不是说会把们最终得到的结果resize成640大小
    # conf-thres：confidence-threshold,置信度阈值.这里参数到底设置成多少好呢？还是根据自己的数据集情况自行调整
    # iou-thres：这个参数就是调节IoU的阈值，IoU是NMS算法中的参数。这里简单介绍一下NMS和IoU https://blog.csdn.net/weixin_42645636/article/details/128775467
    # max-det：最大检测数量，即图像中框住人物的数量上限，默认是最多检测1000个目标。
    # device：GPU数量，如果不指定的话，他会自动检测，这个参数是给GPU多的土豪准备的。
    # view-img：检测的时候是否实时的把检测结果显示出来。action='store_true'，默认
    # save-txt：是否把检测结果保存成一个.txt的格式。命令为python detect.py --save-txt
    # save-conf：是否以.txt的格式保存目标的置信度。单独指定这个命令是没有效果的；必须和--save-txt配合使用，即： python detect.py --save-txt --save-conf。会在.txt多出一列。
    # save-crop：是否把模型检测的物体裁剪下来，如果开启了这个参数会在crops文件夹下看到几个以类别命名的文件夹，里面保存的都是裁剪下来的图片。
    # nosave：是否保存预测的结果。未声明default，action='store_true'，即默认False，当命令参数包含--nosave时，才会为True，虽然为True，但是还会生成exp文件夹，只不过是一个空的exp。
    # classes：要检测的类别。看一下coco128.yaml的配置文件， 0: person  1: bicycle  2: car 3: motorcycle 比如说我这里给classes指定“0”，那么意思就是只检测人这个类别。
    # agnostic-nms：增强版的nms。这里简单介绍一下NMS https://blog.csdn.net/weixin_42645636/article/details/128775467
    # augment：一种增强的方式。
    # visualize：是否把特征图可视化出来，如果开启了这和参数可以看到exp文件夹下又多了一些文件
    # update：如果指定这个参数，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息。
    # project：保存测试日志的文件夹路径。即预测结果保存的路径。这里是/yolov5/run/detect,会生成很多exp文件夹。
    # name：保存测试日志文件夹的名字。即预测结果保存的文件夹名字。所以最终是保存在project/name中
    # exist-ok：是否重新创建日志文件, 这个参数的意思就是每次预测模型的结果是否保存在原来的文件夹。
    # line-thickness：调节预测框线条粗细，因为有的时候目标重叠太多会产生遮挡
    # hide-labels：是否隐藏标签
    # hide-conf：是否隐藏标签的置信度
    # half：是否使用 FP16 半精度推理。简单介绍一下低精度技术：https://blog.csdn.net/weixin_42645636/article/details/128775467
    # dnn：是否使用 OpenCV DNN 进行 ONNX 推理。DNN即Deep Neural Networks
    # vid-stride：视频帧率步幅
    
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')     # coco128.yaml配置文件中0: person ，即只检测人
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()    # 解析参数
    
    # https://blog.csdn.net/weixin_47872288/article/details/127705212
    # 此处对传入的imgsz参数加以判断。
    # 如果命令行不写imgsz参数，则默认imgsz为640，即长度len(opt.imgsz)为1，就需要修改为二维640 * 640。如果命令行传入的imgsz参数为640 * 640 ，则不修改
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand    # imgsz: 网络输入图片大小, 默认的大小是640

    # 将其所有的参数信息进行打印
    print_args(vars(opt))   # print_arg函数是从/utils/general.py导入的，from utils.general import print_args  # print函数的参数arguments

    # 将其opt的参数返回，后续调用main函数需要调用该参数
    return opt    

# ==========================================================================================================================================================
# ===========================================================================主函数=========================================================================
# ==========================================================================================================================================================

def main(opt):
    # 检查requirement的依赖包 有无成功安装，如果没有安装部分会在此处报错
    check_requirements(exclude=('tensorboard', 'thop'))
    # 如果成功安装，将其所有的参数代入，并执行此处的run函数
    run(**vars(opt))   
    # vars(opt)返回对象object的属性和属性值的字典对象
    # 一个星（*）：表示接收的参数作为元组来处理 
    # 两个星（**）：表示接收的参数作为字典来处理 
    # run(**vars(opt))的作用则是把字典vars(opt)变成关键字参数传递。如果 vars(opt) 等于 {'a':1,'b':2,'c':3} ，那这个代码就等价于 run(a=1,b=2,c=3) 。


# 整体的主函数为：
if __name__ == "__main__":      # 该语句用来当文件当作脚本运行时候，就执行代码；但是当文件被当做Module被import的时候，就不执行相关代码。
    # 解析命令行格式下的参数,parse_opt()包含参数.
    opt = parse_opt() 
    # 调用主函数,参数opt传递给main()中，最后run()函数调用          
    main(opt)                   
# 对应命令行格式下的参数可以为图片或者视频流：python detect.py --source data/images/bus.jpg，后面可以加很多参数