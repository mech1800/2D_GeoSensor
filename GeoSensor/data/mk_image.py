import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

'''
データセットをイメージ化するプログラム
'''


def mk_image(pre_geometry_dataset,geometry_dataset,contact_dataset,stress_dataset,force_dataset,label,max,number):

    # データ数を6の倍数にするために最後に0行列を複数個加える
    zeros = np.zeros_like((pre_geometry_dataset[0]))
    for i in range(6-len(pre_geometry_dataset)%6):
        pre_geometry_dataset = np.vstack((pre_geometry_dataset, [zeros]))
        geometry_dataset = np.vstack((geometry_dataset, [zeros]))
        contact_dataset = np.vstack((contact_dataset, [zeros]))
        stress_dataset = np.vstack((stress_dataset, [zeros]))
        force_dataset = np.vstack((force_dataset, [zeros]))

    # 横6×縦4(pre_geometry,geometry,contact,label)の画像を作成
    for i in range(len(pre_geometry_dataset)//6):

        fig = plt.figure(figsize=(6,4))
        fig.subplots_adjust(hspace=0.2, wspace=0.6)

        # pre_geometry_dataset用
        ax1 = fig.add_subplot(4,6,1)
        ax2 = fig.add_subplot(4,6,2)
        ax3 = fig.add_subplot(4,6,3)
        ax4 = fig.add_subplot(4,6,4)
        ax5 = fig.add_subplot(4,6,5)
        ax6 = fig.add_subplot(4,6,6)

        ax1.imshow(pre_geometry_dataset[i*6],cmap=cm.jet)
        ax2.imshow(pre_geometry_dataset[i*6+1],cmap=cm.jet)
        ax3.imshow(pre_geometry_dataset[i*6+2],cmap=cm.jet)
        ax4.imshow(pre_geometry_dataset[i*6+3],cmap=cm.jet)
        ax5.imshow(pre_geometry_dataset[i*6+4],cmap=cm.jet)
        ax6.imshow(pre_geometry_dataset[i*6+5],cmap=cm.jet)

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax6.set_xticks([])
        ax6.set_yticks([])

        ax1.set_title(str(i*6+1))
        ax2.set_title(str(i*6+2))
        ax3.set_title(str(i*6+3))
        ax4.set_title(str(i*6+4))
        ax5.set_title(str(i*6+5))
        ax6.set_title(str(i*6+6))

        ax1.set_ylabel('pre_geometry', fontsize=8)

        # geometry_dataset用
        ax7 = fig.add_subplot(4,6,7)
        ax8 = fig.add_subplot(4,6,8)
        ax9 = fig.add_subplot(4,6,9)
        ax10 = fig.add_subplot(4,6,10)
        ax11 = fig.add_subplot(4,6,11)
        ax12 = fig.add_subplot(4,6,12)

        ax7.imshow(geometry_dataset[i*6],cmap=cm.jet)
        ax8.imshow(geometry_dataset[i*6+1],cmap=cm.jet)
        ax9.imshow(geometry_dataset[i*6+2],cmap=cm.jet)
        ax10.imshow(geometry_dataset[i*6+3],cmap=cm.jet)
        ax11.imshow(geometry_dataset[i*6+4],cmap=cm.jet)
        ax12.imshow(geometry_dataset[i*6+5],cmap=cm.jet)

        ax7.set_xticks([])
        ax7.set_yticks([])
        ax8.set_xticks([])
        ax8.set_yticks([])
        ax9.set_xticks([])
        ax9.set_yticks([])
        ax10.set_xticks([])
        ax10.set_yticks([])
        ax11.set_xticks([])
        ax11.set_yticks([])
        ax12.set_xticks([])
        ax12.set_yticks([])

        ax7.set_ylabel('geometry', fontsize=8)

        # contact_dataset用
        ax13 = fig.add_subplot(4,6,13)
        ax14 = fig.add_subplot(4,6,14)
        ax15 = fig.add_subplot(4,6,15)
        ax16 = fig.add_subplot(4,6,16)
        ax17 = fig.add_subplot(4,6,17)
        ax18 = fig.add_subplot(4,6,18)

        ax13.imshow(contact_dataset[i*6],cmap=cm.jet)
        ax14.imshow(contact_dataset[i*6+1],cmap=cm.jet)
        ax15.imshow(contact_dataset[i*6+2],cmap=cm.jet)
        ax16.imshow(contact_dataset[i*6+3],cmap=cm.jet)
        ax17.imshow(contact_dataset[i*6+4],cmap=cm.jet)
        ax18.imshow(contact_dataset[i*6+5],cmap=cm.jet)

        ax13.set_xticks([])
        ax13.set_yticks([])
        ax14.set_xticks([])
        ax14.set_yticks([])
        ax15.set_xticks([])
        ax15.set_yticks([])
        ax16.set_xticks([])
        ax16.set_yticks([])
        ax17.set_xticks([])
        ax17.set_yticks([])
        ax18.set_xticks([])
        ax18.set_yticks([])

        ax13.set_ylabel('contact', fontsize=8)

        # stress_dataset用
        if label == 'stress':

            ax19 = fig.add_subplot(4,6,19)
            ax20 = fig.add_subplot(4,6,20)
            ax21 = fig.add_subplot(4,6,21)
            ax22 = fig.add_subplot(4,6,22)
            ax23 = fig.add_subplot(4,6,23)
            ax24 = fig.add_subplot(4,6,24)

            im19 = ax19.imshow(stress_dataset[i*6],cmap=cm.jet)
            im20 = ax20.imshow(stress_dataset[i*6+1],cmap=cm.jet)
            im21 = ax21.imshow(stress_dataset[i*6+2],cmap=cm.jet)
            im22 = ax22.imshow(stress_dataset[i*6+3],cmap=cm.jet)
            im23 = ax23.imshow(stress_dataset[i*6+4],cmap=cm.jet)
            im24 = ax24.imshow(stress_dataset[i*6+5],cmap=cm.jet)


            divider = make_axes_locatable(ax19)
            cax19 = divider.append_axes('right', size='5%', pad=0.05)
            cbar19 = fig.colorbar(im19,cax19)
            cbar19.ax.tick_params(labelsize=5)

            divider = make_axes_locatable(ax20)
            cax20 = divider.append_axes('right', size='5%', pad=0.05)
            cbar20 = fig.colorbar(im20,cax20)
            cbar20.ax.tick_params(labelsize=5)

            divider = make_axes_locatable(ax21)
            cax21 = divider.append_axes('right', size='5%', pad=0.05)
            cbar21 = fig.colorbar(im21,cax21)
            cbar21.ax.tick_params(labelsize=5)

            divider = make_axes_locatable(ax22)
            cax22 = divider.append_axes('right', size='5%', pad=0.05)
            cbar22 = fig.colorbar(im22,cax22)
            cbar22.ax.tick_params(labelsize=5)

            divider = make_axes_locatable(ax23)
            cax23 = divider.append_axes('right', size='5%', pad=0.05)
            cbar23 = fig.colorbar(im23,cax23)
            cbar23.ax.tick_params(labelsize=5)

            divider = make_axes_locatable(ax24)
            cax24 = divider.append_axes('right', size='5%', pad=0.05)
            cbar24 = fig.colorbar(im24,cax24)
            cbar24.ax.tick_params(labelsize=5)


            ax19.set_xticks([])
            ax19.set_yticks([])
            ax20.set_xticks([])
            ax20.set_yticks([])
            ax21.set_xticks([])
            ax21.set_yticks([])
            ax22.set_xticks([])
            ax22.set_yticks([])
            ax23.set_xticks([])
            ax23.set_yticks([])
            ax24.set_xticks([])
            ax24.set_yticks([])

            ax19.set_ylabel('stress', fontsize=8)

            # 画像として保存
            fig.savefig('image/stress/'+str(number)+'/table_'+str(i+1)+'.png')

        # force_dataset用
        if label == 'force':

            ax19 = fig.add_subplot(4,6,19)
            ax20 = fig.add_subplot(4,6,20)
            ax21 = fig.add_subplot(4,6,21)
            ax22 = fig.add_subplot(4,6,22)
            ax23 = fig.add_subplot(4,6,23)
            ax24 = fig.add_subplot(4,6,24)

            im19 = ax19.imshow(force_dataset[i*6],cmap=cm.jet,vmin=0,vmax=max)
            im20 = ax20.imshow(force_dataset[i*6+1],cmap=cm.jet,vmin=0,vmax=max)
            im21 = ax21.imshow(force_dataset[i*6+2],cmap=cm.jet,vmin=0,vmax=max)
            im22 = ax22.imshow(force_dataset[i*6+3],cmap=cm.jet,vmin=0,vmax=max)
            im23 = ax23.imshow(force_dataset[i*6+4],cmap=cm.jet,vmin=0,vmax=max)
            im24 = ax24.imshow(force_dataset[i*6+5],cmap=cm.jet,vmin=0,vmax=max)


            divider = make_axes_locatable(ax19)
            cax19 = divider.append_axes('right', size='5%', pad=0.05)
            cbar19 = fig.colorbar(im19,cax19)
            cbar19.ax.tick_params(labelsize=5)
            cbar19.ax.set_ylim(0,max)

            divider = make_axes_locatable(ax20)
            cax20 = divider.append_axes('right', size='5%', pad=0.05)
            cbar20 = fig.colorbar(im20,cax20)
            cbar20.ax.tick_params(labelsize=5)
            cbar20.ax.set_ylim(0, max)

            divider = make_axes_locatable(ax21)
            cax21 = divider.append_axes('right', size='5%', pad=0.05)
            cbar21 = fig.colorbar(im21,cax21)
            cbar21.ax.tick_params(labelsize=5)
            cbar21.ax.set_ylim(0, max)

            divider = make_axes_locatable(ax22)
            cax22 = divider.append_axes('right', size='5%', pad=0.05)
            cbar22 = fig.colorbar(im22,cax22)
            cbar22.ax.tick_params(labelsize=5)
            cbar22.ax.set_ylim(0, max)

            divider = make_axes_locatable(ax23)
            cax23 = divider.append_axes('right', size='5%', pad=0.05)
            cbar23 = fig.colorbar(im23,cax23)
            cbar23.ax.tick_params(labelsize=5)
            cbar23.ax.set_ylim(0, max)

            divider = make_axes_locatable(ax24)
            cax24 = divider.append_axes('right', size='5%', pad=0.05)
            cbar24 = fig.colorbar(im24,cax24)
            cbar24.ax.tick_params(labelsize=5)
            cbar24.ax.set_ylim(0, max)


            ax19.set_xticks([])
            ax19.set_yticks([])
            ax20.set_xticks([])
            ax20.set_yticks([])
            ax21.set_xticks([])
            ax21.set_yticks([])
            ax22.set_xticks([])
            ax22.set_yticks([])
            ax23.set_xticks([])
            ax23.set_yticks([])
            ax24.set_xticks([])
            ax24.set_yticks([])

            ax19.set_ylabel('force', fontsize=8)

            # 画像として保存
            fig.savefig('image/force/'+str(number)+'/table_'+str(i+1)+'.png')


# main
label = input('label means stress or force?')

# 分布荷重の最大値(μN)
max = 40

for number in [0,1,2,3,4,5,6,7,8,9]:

    print(number)

    # datasetをloadする
    pre_geometry_dataset = np.load('dataset/'+str(number)+'/pre_geometry.npy')
    geometry_dataset = np.load('dataset/'+str(number)+'/geometry.npy')
    contact_dataset = np.load('dataset/'+str(number)+'/contact.npy')
    stress_dataset = np.load('dataset/'+str(number)+'/stress.npy')
    force_dataset = np.load('dataset/'+str(number)+'/force.npy')

    mk_image(pre_geometry_dataset,geometry_dataset,contact_dataset,stress_dataset,force_dataset,label,max,number)