# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project : TransFuse
# @File : args_trans.py
# @Time : 2021/11/9 14:15

class Args():
	# For training
	# path_ir = ['G:/datasets/Image-fusion/KAIST/lwir/']
	path_ir = ['kaist_dataset/kaist_train/set00/V000/lwir']
	# path_ir = ['/vol/scratch/SoC/misc/2024/fxlf1861/dataset_full/new_baseline/V000/lwir']

	# path_ir = ['/data/Disk_B/KAIST-RGBIR/lwir/']
	cuda = True
	# cuda = False

	lr = 0.0002
	epochs = 4
	batch =8
	train_num = 20000
	step = 10
	# Network Parameters
	channel = 1
	Height = 256
	Width = 256
 
	crop_h = 256
	crop_w = 256

	vgg_model_dir = "./models/vgg"
	resume_model_auto_ir = "/vol/scratch/SoC/misc/2024/fxlf1861/new_models/new_auto_encoder_epoch_4_ir.model"
	resume_model_auto_vi = "/vol/scratch/SoC/misc/2024/fxlf1861/new_models/new_auto_encoder_epoch_4_vi.model"
	# resume_model_auto_ir = None
	# resume_model_auto_vi = None

	# resume_model_trans = "./models/transfuse/fusetrans_epoch_4.model"
	resume_model_trans = None
	save_fusion_model = "/vol/scratch/SoC/misc/2024/fxlf1861/new_models/"
	save_loss_dir = "./models/loss"