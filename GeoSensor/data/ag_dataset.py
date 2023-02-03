import numpy as np

'''
データセットを拡張するプログラム
'''

# データを回転
def rotate_90_180_270(pre_geometry_dataset,geometry_dataset,contact_dataset,stress_dataset,force_dataset):

    # pre_geometry
    pre_geometry_dataset_90 = np.rot90(pre_geometry_dataset,1,axes=(1,2))
    pre_geometry_dataset_180 = np.rot90(pre_geometry_dataset,2,axes=(1,2))
    pre_geometry_dataset_270 = np.rot90(pre_geometry_dataset,3,axes=(1,2))
    pre_geometry_dataset = np.vstack((pre_geometry_dataset, pre_geometry_dataset_90))
    pre_geometry_dataset = np.vstack((pre_geometry_dataset, pre_geometry_dataset_180))
    pre_geometry_dataset = np.vstack((pre_geometry_dataset, pre_geometry_dataset_270))

    # geometry
    geometry_dataset_90 = np.rot90(geometry_dataset, 1, axes=(1, 2))
    geometry_dataset_180 = np.rot90(geometry_dataset, 2, axes=(1, 2))
    geometry_dataset_270 = np.rot90(geometry_dataset, 3, axes=(1, 2))
    geometry_dataset = np.vstack((geometry_dataset, geometry_dataset_90))
    geometry_dataset = np.vstack((geometry_dataset, geometry_dataset_180))
    geometry_dataset = np.vstack((geometry_dataset, geometry_dataset_270))

    # contact
    contact_dataset_90 = np.rot90(contact_dataset, 1, axes=(1, 2))
    contact_dataset_180 = np.rot90(contact_dataset, 2, axes=(1, 2))
    contact_dataset_270 = np.rot90(contact_dataset, 3, axes=(1, 2))
    contact_dataset = np.vstack((contact_dataset, contact_dataset_90))
    contact_dataset = np.vstack((contact_dataset, contact_dataset_180))
    contact_dataset = np.vstack((contact_dataset, contact_dataset_270))

    # stress
    stress_dataset_90 = np.rot90(stress_dataset, 1, axes=(1, 2))
    stress_dataset_180 = np.rot90(stress_dataset, 2, axes=(1, 2))
    stress_dataset_270 = np.rot90(stress_dataset, 3, axes=(1, 2))
    stress_dataset = np.vstack((stress_dataset, stress_dataset_90))
    stress_dataset = np.vstack((stress_dataset, stress_dataset_180))
    stress_dataset = np.vstack((stress_dataset, stress_dataset_270))

    # force
    force_dataset_90 = np.rot90(force_dataset, 1, axes=(1, 2))
    force_dataset_180 = np.rot90(force_dataset, 2, axes=(1, 2))
    force_dataset_270 = np.rot90(force_dataset, 3, axes=(1, 2))
    force_dataset = np.vstack((force_dataset, force_dataset_90))
    force_dataset = np.vstack((force_dataset, force_dataset_180))
    force_dataset = np.vstack((force_dataset, force_dataset_270))

    return pre_geometry_dataset,geometry_dataset,contact_dataset,stress_dataset,force_dataset


# main部分

# 0～9まで
for number in range(10):

    print(number)

    # original_datasetをloadする
    pre_geometry_dataset = np.load('original_dataset/'+str(number)+'/pre_geometry.npy')
    geometry_dataset = np.load('original_dataset/'+str(number)+'/geometry.npy')
    contact_dataset = np.load('original_dataset/'+str(number)+'/contact.npy')
    stress_dataset = np.load('original_dataset/'+str(number)+'/stress.npy')
    force_dataset = np.load('original_dataset/'+str(number)+'/force.npy')

    # 90°ずつ回転させたデータをdetasetに追加
    pre_geometry_dataset,geometry_dataset,contact_dataset,stress_dataset,force_dataset = rotate_90_180_270(pre_geometry_dataset,geometry_dataset,contact_dataset,stress_dataset,force_dataset)

    # datasetを更新する
    np.save('dataset/'+str(number)+'/pre_geometry', pre_geometry_dataset)
    np.save('dataset/'+str(number)+'/geometry', geometry_dataset)
    np.save('dataset/'+str(number)+'/contact', contact_dataset)
    np.save('dataset/'+str(number)+'/stress', stress_dataset)
    np.save('dataset/'+str(number)+'/force', force_dataset)