#!/usr/bin/env python
# -*- coding:utf-8   -*-

# import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math


def loadData(filePath1,filePath2,filePath3,filePath4,filePath5):
    fr1 = open(filePath1, 'r+')
    fr2 = open(filePath2, 'r+')
    fr3 = open(filePath3, 'r+')
    fr4 = open(filePath4, 'r+')
    fr5 = open(filePath5, 'r+')
    lines1 = fr1.readlines()
    lines2 = fr2.readlines()
    lines3 = fr3.readlines()
    lines4 = fr4.readlines()
    lines5 = fr5.readlines()

    # 这里可以记录一下挨个数值分别是啥意思
    loss_1 = []
    loss_2 = []
    loss_3 = []
    loss_4 = []
    loss_5 = []
    for line1 in lines1:
        #loss_1.append(math.log(float(line1.strip().split('\n')[0])))
        loss_1.append(float(line1.strip().split('\n')[0]))
    for line2 in lines2:
        #loss_2.append(math.log(float(line2.strip().split('\n')[0])))
        loss_2.append(float(line2.strip().split('\n')[0]))
    for line3 in lines3:
        loss_3.append(float(line3.strip().split('\n')[0]))
        #miou.append(float(line3.strip().split('\n')[0]) )
        #items = line3.strip().split(',     ')
        #items = line3.strip().split(',')
        # print(items)
        # ep_0.append(float(items[-3]))
        # ep_1.append(float(items[-2]))
        # ep_2.append(float(items[-1]))
        #miou.append(float(items[2]))
    for line4 in lines4:
        #loss_1.append(math.log(float(line1.strip().split('\n')[0])))
        loss_4.append(float(line4.strip().split('\n')[0]))
    for line5 in lines5:
        #loss_1.append(math.log(float(line1.strip().split('\n')[0])))
        loss_5.append(float(line5.strip().split('\n')[0]))
    return loss_1, loss_2, loss_3,loss_4,loss_5


if __name__ == '__main__':
    config = {
        "font.family": 'serif',
        "font.size": 18,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)

    filePath1='train_loss.txt'
    filePath2='train_loss1.txt'
    filePath3='train_loss2.txt'
    filePath4='train_loss3.txt'
    filePath5='train_loss4.txt'
    # e=[]
    # for i in range(100): e.append(i)

    loss_1, loss_2,loss_3,loss_4,loss_5 = loadData(filePath1,filePath2,filePath3,filePath4,filePath5)

    e = list(range(1, 101))

    # 调节图像大小,清晰度
    # plt.figure(figsize=(6, 5), dpi=100)
    plt.figure(figsize=(6, 5),dpi=130)

    #plt.rc('font', family='Times New Roman',size = 12,weight='bold')  #全部改字体
    #plt.rc('font', family='Times New Roman',size = 12)  #全部改字体
    # plt.xlabel('轮数/轮', fontsize=12,fontdict={'family' : 'SimSun'}) # fontdict={'family' : 'Times New Roman'}, weight='bold'
    # plt.ylabel('损失值', fontsize=12,fontdict={'family' : 'SimSun'})   # fontdict={'family' : 'Times New Roman'}, weight='bold'

    plt.rc('font',size = 12)  #全部改字体
    plt.xlabel('轮数/轮', fontsize=12) # fontdict={'family' : 'Times New Roman'}, weight='bold'
    plt.ylabel('损失值', fontsize=12)   # fontdict={'family' : 'Times New Roman'}, weight='bold'
    plt.title('')

    # plt.plot(e, loss_1, color='red')
    # plt.plot(e, loss_2, color='orange')
    # plt.plot(e, loss_3, color='green')
    # plt.plot(e, loss_4, color='purple')
    # plt.plot(e, loss_5)

    plt.plot(e, loss_1, linestyle='-',color='red',linewidth = '1.2') #'-'
    plt.plot(e, loss_2, linestyle='--', color='orange',linewidth = '1.3') #'--'
    plt.plot(e, loss_3, linestyle='-.', color='green',linewidth = '1.4')
    plt.plot(e, loss_4, linestyle=':', color='purple',linewidth = '1.6')
    plt.plot(e, loss_5, marker = 'o',ms=1.8,linewidth = '1')

    #plt.legend(('本文方法','U-Net+MSFF','U-Net+MLPA', 'U-Net+MSFF+PPA','U-Net+MSFF+PCA'), loc='upper right',prop={'family' : 'HGSS_CNKI'},fontsize=12)  #prop={'family' : 'Times New Roman', 'size'   : 16}
    plt.legend(('本文方法', '$\mathrm{U-Net+MSFF}$', '$\mathrm{U-Net+MLPA}$', '$\mathrm{U-Net+MSFF+PPA}$',
                '$\mathrm{U-Net+MSFF+PCA}$'), loc='upper right',
               fontsize=12)  # prop={'family' : 'Times New Roman', 'size'   : 16}
    plt.savefig("Test.svg", dpi=600, bbox_inches='tight')  # 无边框, bbox_inches='tight'
    plt.show()


    # plt.xlabel(u"epoch")
    # plt.ylabel(u"MIoU")  # xlabel、ylabel：设置坐标轴标签
    # plt.xticks(fontsize=9)  # xticks、yticks：设置坐标轴刻度   title：标题    fontsize：字体大小
    # plt.yticks(fontsize=9)
    # # plt.plot(x1,y,'--',label="色彩度")
    # plt.plot(e, m, color='orange', label="MIoU")
    # plt.legend(fontsize=9)  # legend：图例
    # plt.savefig(r"beta.jpg", bbox_inches='tight')
