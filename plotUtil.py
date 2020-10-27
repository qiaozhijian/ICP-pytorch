import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import h5py
from scipy.spatial.transform import Rotation
from util.util import transform_point_cloud, npmat2euler
np.set_printoptions(threshold=np.inf)

figure_id = 0
class GlobalVar():
    def __init__(self):
        self.self_att_src=None
        self.cross_self_att_src=None
        self.cross_att_src=None
        self.self_att_tgt=None
        self.cross_self_att_tgt=None
        self.cross_att_tgt=None
        self.src=None
        self.tgt=None

def format(pcl):
    if type(pcl) is torch.Tensor:  pcl=pcl.detach().cpu().numpy();
    if len(pcl.shape)==3:  pcl=pcl[0];
    if pcl.shape[0]>pcl.shape[1]: pcl=pcl.T;
    return pcl

def plot3d2(pcl1,pcl2,s1=1,s2=1,color1='blue',color2='red'):
    pcl1 = format(pcl1)
    pcl2 = format(pcl2)

    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection='3d')

    ax.scatter(pcl1[0], pcl1[1], pcl1[2], color=color1, s=s1)
    ax.scatter(pcl2[0], pcl2[1], pcl2[2], color=color2, s=s2)
    ax.view_init(elev=-90, azim=0)

    limit_max = np.max([np.max(pcl1),np.max(pcl2)])*0.65;limit_min = np.min([np.min(pcl1),np.min(pcl2)])*0.65
    ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
    ax.set_xlim([limit_min, limit_max]);ax.set_ylim([limit_min, limit_max]);ax.set_zlim([limit_min, limit_max])
    # plt.axis('off')
    plt.show()

def plot3d1(pcl1,s1=1,color1='blue'):
    pcl1 = format(pcl1)

    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection='3d')

    ax.scatter(pcl1[0], pcl1[1], pcl1[2], color=color1, s=s1)
    # 默认 azim=-60, elev=30
    ax.view_init(elev=-90, azim=0)

    limit_max = np.max([pcl1]);limit_min = np.min([pcl1])
    ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
    ax.set_xlim([limit_min, limit_max]);ax.set_ylim([limit_min, limit_max]);ax.set_zlim([limit_min, limit_max])
    plt.axis('off')
    plt.show()

def savePlt(plot_path_dir,bpcl1,bpcl2=None,bpcl3=None, s=0.7,title = '', frame = None):
    # matplotlib.use('Agg')
    global  figure_id
    if bpcl2 is None:
        bpcl1 = format(bpcl1)
        fig=plt.figure(figsize=(10,10))
        for pcl1 in bpcl1:
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=-90.0,azim=0.0)
            ax.scatter(pcl1[0], pcl1[1], pcl1[2], color='blue', s=s)
            figure_id = figure_id + 1
            plt.axis('off')
            plt.title(title, color='black')
            plt.savefig(plot_path_dir + "/" + str(figure_id) + ".png", bbox_inches='tight', pad_inches=0.1)
            plt.clf()
    elif bpcl3 is None:
        bpcl1 = format(bpcl1)
        bpcl2 = format(bpcl2)
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=-90.0,azim=0.0)
        limit_max = np.max([np.max(bpcl1), np.max(bpcl2)]) * 0.65;
        limit_min = np.min([np.min(bpcl1), np.min(bpcl2)]) * 0.65
        ax.set_xlim([limit_min, limit_max]);
        ax.set_ylim([limit_min, limit_max]);
        ax.set_zlim([limit_min, limit_max])
        ax.scatter(bpcl1[0], bpcl1[1], bpcl1[2], color='blue', s=s)
        ax.scatter(bpcl2[0], bpcl2[1], bpcl2[2], color='red', s=s)
        if frame == None:
            figure_id = figure_id + 1
        else:
            figure_id = frame
        plt.axis('off')
        plt.title(title, color='black')
        plt.savefig(plot_path_dir + "/" + str(figure_id) + ".png", bbox_inches='tight', pad_inches=0.1)
        plt.clf()
    else:
        if type(bpcl1) is torch.Tensor:
            bpcl1=bpcl1.detach().cpu().numpy()
            bpcl2=bpcl2.detach().cpu().numpy()
            bpcl3=bpcl3.detach().cpu().numpy()
        fig=plt.figure(figsize=(15,15))
        for pcl1, pcl2, pcl3 in zip(bpcl1, bpcl2, bpcl3):
            flag = [False,False,True,True]
            if flag[0]:
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=-90.0,azim=0.0)
                ax.scatter(pcl1[0], pcl1[1], pcl1[2], color='blue', s=s)
                plt.axis('off')
                plt.title(title, color='black')
                plt.savefig(plot_path_dir[0] + "/" + str(figure_id) + ".png", bbox_inches='tight', pad_inches=0.1)

            if flag[1]:
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=-90.0,azim=0.0)
                ax.scatter(pcl2[0], pcl2[1], pcl2[2], color='red', s=s)
                plt.axis('off')
                plt.title(title, color='black')
                plt.savefig(plot_path_dir[1] + "/" + str(figure_id) + ".png", bbox_inches='tight', pad_inches=0.1)

            if flag[2]:
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=-90.0,azim=0.0)
                ax.scatter(pcl1[0], pcl1[1], pcl1[2], color='blue', s=s)
                ax.scatter(pcl2[0], pcl2[1], pcl2[2], color='red', s=s)
                plt.axis('off')
                plt.title(title, color='black')
                plt.savefig(plot_path_dir[2] + "/" + str(figure_id) + ".png", bbox_inches='tight', pad_inches=0.1)

            if flag[3]:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pcl1[0], pcl1[1], pcl1[2], color='blue', s=s)
                ax.scatter(pcl3[0], pcl3[1], pcl3[2], color='red', s=s)
                ax.view_init(elev=-90.0,azim=0.0)
                plt.axis('off')
                plt.title(title, color='black')
                plt.savefig(plot_path_dir[3] + "/" + str(figure_id) + ".png", bbox_inches='tight', pad_inches=0.1)
            # plt.show()
            figure_id = figure_id + 1
            plt.clf()

def visualization_test(args,src, target,transformed_target,folderName='visualization'):
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + folderName):
        os.makedirs('checkpoints/' + args.exp_name + '/' + folderName)
        os.makedirs('checkpoints/' + args.exp_name + '/' + folderName + '/pcl1')
        os.makedirs('checkpoints/' + args.exp_name + '/' + folderName + '/pcl2')
        os.makedirs('checkpoints/' + args.exp_name + '/' + folderName + '/pcl3')
        os.makedirs('checkpoints/' + args.exp_name + '/' + folderName + '/pcl4')
    path = 'checkpoints/' + args.exp_name + '/' + folderName + '/'
    plot_path_dir = [path+'pcl1',path+'pcl2',path+'pcl3',path+'pcl4']
    savePlt(plot_path_dir,src, target,transformed_target)
    return 0

def vScores(src,tgt,scores,mode = 0,s=1.0):
    global figure_id
    src = src.detach().cpu().numpy().squeeze()
    tgt = tgt.detach().cpu().numpy().squeeze()
    scores = scores.detach().cpu().numpy().squeeze()
    tgt = tgt + np.full(shape=tgt.shape,fill_value=1)
    scoresColSum = np.sum(scores,axis=0)
    idxNone = np.argsort(scoresColSum, axis=-1, kind='quicksort', order=None)
    print(scoresColSum[idxNone])

    # pyplot.imshow(np.sqrt(np.sqrt(scores)))
    # pyplot.waitforbuttonpress();
    # return 0
    name = 'plane'
    if not os.path.exists('experiment/' + name + str(mode)):
        os.makedirs('experiment/' + name + str(mode))
    plot_path_dir = 'experiment/' + name + str(mode)
    dims,num_points = src.shape

    fig = plt.figure()

    if mode == 0:
        for num in range(0, num_points, 5):
            figure_id = figure_id + 1
            ax = fig.add_subplot(111, projection='3d')
            # 把每一点的所有tgt权重可视化
            ax.scatter(src[0], src[1], src[2], color='blue', s=s)
            ax.scatter(tgt[0], tgt[1], tgt[2], color='blue', s=s)
            ax.scatter(tgt[0,idxNone[:20]], tgt[1,idxNone[:20]], tgt[2,idxNone[:20]], color='red', s=5)

            # ax.scatter(src[0][num], src[1][num], src[2][num], s=50, marker='v',c='r',edgecolors='r', linewidths=2)
            # ax.scatter(tgt[0], tgt[1], tgt[2], c=scores[num], cmap='PuRd', s=s)
            ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
            plt.axis('off')
            plt.show()
            # plt.savefig(plot_path_dir + "/" + str(figure_id) + ".png", bbox_inches='tight', pad_inches=0.1)
            # plt.clf()
    elif mode == 1:
        for num in range(0, num_points, 5):
            ax = fig.add_subplot(111, projection='3d')
            idx = np.argsort(scores[num], axis=-1, kind='quicksort', order=None)
            idx = idx[-3:]
            figure_id = figure_id + 1

            # 把每一点的前三大tgt权重可视化
            ax.scatter(src[0], src[1], src[2], color='blue', s=s)
            ax.scatter(src[0][num], src[1][num], src[2][num], s=100, marker='v',edgecolors='r', linewidths=2)
            ax.scatter(tgt[0], tgt[1], tgt[2], color='blue', s=s)
            ax.scatter(tgt[0][idx], tgt[1][idx], tgt[2][idx], s=100, marker='v',c='r',edgecolors='r', linewidths=2)

            ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
            plt.axis('off')
            plt.title('top 3 scores sum: ' + str(np.sum(scores[num, idx])))
            plt.show()
            plt.savefig(plot_path_dir + "/" + str(figure_id) + ".png", bbox_inches='tight', pad_inches=0.1)
            plt.clf()
    elif mode == 2:
        topKsum = np.zeros((num_points))
        for num in range(0, num_points):
            idx = np.argsort(scores[num], axis=-1, kind='quicksort', order=None)
            idx = idx[-1:]
            topKsum[num] = np.sum(scores[num, idx])
        _range = np.max(topKsum) - np.min(topKsum)
        topKsum = (topKsum - np.min(topKsum)) / _range

        ax = fig.add_subplot(111, projection='3d')
        figure_id = figure_id + 1
        ax.scatter(src[0], src[1], src[2], c=topKsum, cmap='PuRd', s=s)

        ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
        plt.axis('off')
        plt.show()
        plt.savefig(plot_path_dir + "/" + str(figure_id) + ".png", bbox_inches='tight', pad_inches=0.1)
        plt.clf()

def getKittiPcl(seqN,binNum):
    path='/media/qzj/Document/grow/research/slamDataSet/kitti/data_odometry_velodyne/dataset/downSample/bin2/'+str(seqN).zfill(2)+'/velodyne/'+str(binNum).zfill(6)+'.bin'
    # path = '/media/qzj/My Book/KITTI/data_odometry_velodyne/dataset/sequences/' + str(seqN).zfill(2) + '/velodyne/' + str(binNum).zfill(6) + '.bin'
    pc = np.fromfile(path, dtype=np.float32, count=-1);
    pc = pc.reshape([-1, 4])
    pc = (pc.T)[0:3, :]

    return pc
def getPcMulti(seqN,binNum,multi):
    pc=np.zeros((3,1))
    for i in range(0,multi):
        binNumTmp=binNum+i
        pcTmp=getKittiPcl(seqN,binNumTmp)
        R_ab, translation_ab = getT21(seqN, binNumTmp, binNum)
        pcTmp = R_ab.dot(pcTmp) + translation_ab
        if i == 0:
            pc = pcTmp
        else:
            pc=np.hstack((pc,pcTmp))
    # plot3d1(pc)
    return pc.T
def getT21(seq,num1,num2,Velodyne=True):
    SeqPose = np.loadtxt('/media/qzj/Document/grow/research/slamDataSet/kitti/data_odometry_poses/dataset/poses/' + str(seq).zfill(2) + '.txt')
    # SeqPose = np.loadtxt('/media/qzj/My Book/KITTI/data_odometry_poses/' + str(seqN).zfill(2) + '.txt')
    # 得到相对位姿
    Tw1 = np.eye(4, 4);Tw2 = np.eye(4, 4)
    Tw1[0:3, 0:4] = SeqPose[num1].reshape([3, 4])
    Tw2[0:3, 0:4] = SeqPose[num2].reshape([3, 4])

    T21 = (np.dot(np.linalg.inv(Tw2), Tw1))
    if Velodyne:
        T21 = camera2velodyne(T21)
    R_ab = T21[0:3, 0:3]
    translation_ab = T21[0:3, 3].reshape(3, 1)
    return R_ab,translation_ab

def camera2velodyne(gt_c):
    t = np.asarray( [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,-7.210626507497e-03,  \
                    8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02, 9.999738645903e-01,4.859485810390e-04,  \
                            -7.206933692422e-03, -2.921968648686e-01,0,0,0,1]).reshape(4,4)
    # t[:3,3] = t[:3,3] *0.0
    gt_v = np.matmul(np.linalg.inv(t).dot(gt_c),t)
    return gt_v

def point2camera(pcl):
    T = np.asarray( [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,-7.210626507497e-03,  \
                    8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02, 9.999738645903e-01,4.859485810390e-04,  \
                            -7.206933692422e-03, -2.921968648686e-01,0,0,0,1]).reshape(4,4)

    # T = np.linalg.inv(T)
    pcl = format(pcl)
    R_ab = T[0:3, 0:3]
    translation_ab = T[0:3, 3].reshape(3, 1)
    pcl = (np.matmul(R_ab,pcl) + translation_ab.reshape([3,1]))

    return pcl


