import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
import nibabel as nib

def load_dcm_series(dcm_folder):
    """加载 DICOM 文件序列并返回 3D 图像"""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dcm_folder)
    if not series_ids:
        raise ValueError("No DICOM series found in the folder")

    series_file_names = reader.GetGDCMSeriesFileNames(dcm_folder, series_ids[0])
    reader.SetFileNames(series_file_names)
    image = reader.Execute()
    return image

def get_boundary_for_label(volume, label):
    """根据特定标签获取 x, y, z 的最小值和最大值"""
    # 提取特定标签的区域
    labeled_indices = np.where(volume == label)
    if len(labeled_indices[0]) == 0:
        return None  # 如果没有找到该标签，返回 None

    x_min, x_max = np.min(labeled_indices[0]), np.max(labeled_indices[0])
    y_min, y_max = np.min(labeled_indices[1]), np.max(labeled_indices[1])
    z_min, z_max = np.min(labeled_indices[2]), np.max(labeled_indices[2])
    return (x_min, x_max, y_min, y_max, z_min, z_max)

def get_centered_crop(image, x_min, x_max, y_min, y_max, z_min, z_max, crop_size=64):
    """根据给定的 x, y, z 范围，提取中心位置的 64x64x64 区域"""
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    z_center = (z_min + z_max) // 2

    half_crop = crop_size // 2
    x_start = max(0, x_center - half_crop)
    y_start = max(0, y_center - half_crop)
    z_start = max(0, z_center - half_crop)

    x_end = min(image.GetSize()[0], x_start + crop_size)
    y_end = min(image.GetSize()[1], y_start + crop_size)
    z_end = min(image.GetSize()[2], z_start + crop_size)

    size = [x_end - x_start, y_end - y_start, z_end - z_start]
    index = [x_start, y_start, z_start]

    size = [int(s) for s in size]
    index = [int(i) for i in index]

    cropped_region = sitk.RegionOfInterest(image, size=size, index=index)
    return cropped_region

def save_cropped_to_npy(cropped_image, output_path):
    """将裁剪的 3D 图像直接保存为 Numpy 格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    array = sitk.GetArrayFromImage(cropped_image)
    np.save(output_path, array)
    print(f"保存为 Numpy 文件: {output_path}")

def save_slices_as_images(volume, output_folder):
    """
    将 3D 图像的切片保存为 PNG 格式，确保保持原始宽高比。
    参数:
        volume: 3D NumPy 数组，表示图像的体数据 (z, y, x)。
        output_folder: 字符串，保存图像的文件夹路径。
    """
    os.makedirs(output_folder, exist_ok=True)

    # 遍历每个 z 轴切片
    for i in range(volume.shape[0]):
        slice_image = volume[i, :, :]

        # 使用 Matplotlib 保存为 PNG 格式
        plt.imshow(slice_image, cmap='gray', aspect='auto')
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')  # 保持正方形像素

        # 保存图像时去掉边距
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(os.path.join(output_folder, f'slice_{i:03d}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()


def crop(file_fold):
    file_fold = Path(file_fold)
    patients_ids = []

    for folder in file_fold.glob('**/*'):
        if folder.is_dir():
            patient_id = folder.name
            nii_files = [f for f in os.listdir(folder) if f.endswith('.nii.gz')]
            dcm_folder = folder  # 假设 DCM 文件放在名为 'dcm_folder' 的文件夹里
            assert len(nii_files) == 1, f"患者{patient_id}的nii文件数不为1"
            patients_ids.append(patient_id)
            nii_file_path = os.path.join(folder, nii_files[0])
            print(f"患者{patient_id}开始切片-------------------------------------------------------")


            # 加载 NIfTI 和 DCM 数据
            img = nib.load(nii_file_path)
            img_data = img.get_fdata()
            print(img_data.shape)

            # 遍历所有标签
            labels = np.unique(img_data)
            labels = labels[labels > 0]  # 假设标签为正数，背景为 0

            # 加载 DICOM 文件序列
            image = load_dcm_series(dcm_folder)


            print('图像大小:', image.GetSize())
            assert image.GetSize()==img_data.shape

            # 对每个标签进行裁剪和保存
            for label in labels:
                boundary = get_boundary_for_label(img_data, label)
                if boundary is None:
                    continue

                x_min, x_max, y_min, y_max, z_min, z_max = boundary
                print(f"标签 {label} 的范围: X轴({x_min}-{x_max}), Y轴({y_min}-{y_max}), Z轴({z_min}-{z_max})")

                # 获取裁剪区域
                cropped_image = get_centered_crop(image, x_min, x_max, y_min, y_max, z_min, z_max)

                # 归一化裁剪区域
                array = sitk.GetArrayFromImage(cropped_image)

                # # 示例调用
                # output_folder = f'output_slices/{patient_id}'
                # # 假设 cropped_image 是 3D NumPy 数组
                # save_slices_as_images(array, output_folder)

                print(f'最大值{np.max(array)},最小值{np.min(array)}')
                # 保存裁剪的 3D 图像
                save_path = f"{crop_path}/{dataset_class}/{patient_id}_label{int(label)}_cropped.npy"
                save_cropped_to_npy(cropped_image, save_path)
            print(f"患者{patient_id}的{len(labels)}个肺结节完成切片")

    return len(patients_ids)

if __name__ == "__main__":
    crop_path = "cropped_data"
    dataset_class = "training"
    patients_num = crop(dataset_class)
    print(f"{dataset_class}集{patients_num}个患者已完成裁剪")
