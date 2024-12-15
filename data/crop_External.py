import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import nibabel as nib
import scipy
import matplotlib.pyplot as plt
import pydicom


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
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def get_boundary_for_label(volume, label, newSpacing, Spacing):
    """根据特定标签获取 x, y, z 的最小值和最大值"""

    # 提取特定标签的区域
    labeled_indices = np.where(volume == label)
    if len(labeled_indices[0]) == 0:
        return None  # 如果没有找到该标签，返回 None
    x_min, x_max = np.min(np.round(labeled_indices[0] * Spacing[0] / newSpacing[0])), np.max(np.round(labeled_indices[0] * Spacing[0] / newSpacing[0]))
    y_min, y_max = np.min(np.round(labeled_indices[1] * Spacing[1] / newSpacing[1])), np.max(np.round(labeled_indices[1] * Spacing[1] / newSpacing[1]))
    z_min, z_max = np.min(np.round(labeled_indices[2] * Spacing[2] / newSpacing[2])), np.max(np.round(labeled_indices[2] * Spacing[2] / newSpacing[2]))
    # x_min, x_max = np.min(np.round(labeled_indices[0])), np.max(np.round(labeled_indices[0]))
    # y_min, y_max = np.min(np.round(labeled_indices[1])), np.max(np.round(labeled_indices[1]))
    # z_min, z_max = np.min(np.round(labeled_indices[2])), np.max(np.round(labeled_indices[2]))

    return (x_min, x_max, y_min, y_max, z_min, z_max)

def get_centered_crop(image, x_min, x_max, y_min, y_max, z_min, z_max, crop_size=32):
    """根据给定的 x, y, z 范围，提取中心位置的 64x64x64 区域"""
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    z_center = (z_min + z_max) // 2

    half_crop = crop_size // 2
    x_start = int(max(0, x_center - half_crop))
    y_start = int(max(0, y_center - half_crop))
    z_start = int(max(0, z_center - half_crop))

    x_end = min(image.GetSize()[0], x_start + crop_size)
    y_end = min(image.GetSize()[1], y_start + crop_size)
    z_end = min(image.GetSize()[2], z_start + crop_size)

    # x_end = int(min(image.shape[0], x_start + crop_size))
    # y_end = int(min(image.shape[1], y_start + crop_size))
    # z_end = int(min(image.shape[2], z_start + crop_size))

    size = [x_end - x_start, y_end - y_start, z_end - z_start]
    index = [x_start, y_start, z_start]

    size = [int(s) for s in size]
    index = [int(i) for i in index]

    # cropped_region = image[x_start:x_end, y_start:y_end, z_start:z_end]

    cropped_region = sitk.RegionOfInterest(image, size=size, index=index)
    return cropped_region

def save_cropped_to_npy(cropped_image, output_path):
    """将裁剪的 3D 图像直接保存为 Numpy 格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    array = sitk.GetArrayFromImage(cropped_image)
    np.save(output_path, array)
    print(f"保存为 Numpy 文件: {output_path}")

def load_scan(path):
    slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path) if s.endswith('.DCM')]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices, (float(slices[0].PixelSpacing[1]), float(slices[0].PixelSpacing[0]), float(slices[0].SliceThickness))

def transform(image, newSpacing, resamplemethod=sitk.sitkBSpline, Spacing = [1,1,1], is_label=False):
    # 设置一个Filter
    resample = sitk.ResampleImageFilter()
    # 初始的体素块尺寸
    originSize = image.GetSize()
    Spacing = image.GetSpacing()

    print(f'原始大小：{originSize}')
    print(f'原始体素：{Spacing}')
    # 初始的体素间距
    originSpacing= Spacing
    newSize = [
        int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
        int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
        int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
    ]

    # 沿着x,y,z,的spacing（3）
    # The sampling grid of the output space is specified with the spacing along each dimension and the origin.
    resample.SetOutputSpacing(newSpacing)
    # 设置original
    resample.SetOutputOrigin(image.GetOrigin())
    # 设置方向
    resample.SetOutputDirection(image.GetDirection())
    resample.SetSize(newSize)
    # 设置插值方式
    resample.SetInterpolator(resamplemethod)
    # 设置transform
    resample.SetTransform(sitk.Euler3DTransform())
    # 默认像素值
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    return resample.Execute(image), originSpacing


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
    all_label = 0
    unique_patients = []
    for folder in file_fold.glob('**/*'):
        for folder1 in folder.glob('**/*'):
            for folder2 in folder1.glob('**/*'):
                if folder2.is_dir():
                    patient_id = folder.name
                    print(f"患者{patient_id}开始切片-------------------------------------------------------")
                    nii_files = [f for f in os.listdir(folder2) if f.endswith('.nii.gz')]

                    # 加载 DICOM 文件序列
                    image = load_dcm_series(folder2)

                    #重采样
                    resampled_image, oldSpacing = transform(image = image, newSpacing = [1, 1, 1],  resamplemethod=sitk.sitkBSpline, is_label = True)
                    print(f'现在大小：{resampled_image.GetSize()}')
                    assert len(nii_files) == 1, f"患者{patient_id}的nii文件数不为1"

                    if len(nii_files) != 1:
                        print(f"患者{patient_id}的nii文件数不为1")

                    # 重采样 NIfTI 数据
                    nii_file_path = os.path.join(folder2, nii_files[0])
                    img = nib.load(nii_file_path)
                    img_data = img.get_fdata()

                    # img_itk = sitk.ReadImage(nii_file_path)
                    # nii_data = transform(image=img_itk, newSpacing=[1, 1, 1], resamplemethod=sitk.sitkNearestNeighbor, Spacing=Spacing, is_label=True)
                    # img_data = sitk.GetArrayFromImage(nii_data)  #z,y,x
                    # img_data = img_data.transpose(-1, 1, 0)

                    labels = np.unique(img_data)
                    labels = labels[labels > 0]  # 假设标签为正数，背景为 0

                    patients_ids.append(patient_id)
                    arr_image = sitk.GetArrayFromImage(resampled_image)
                    print(f'最大值{np.max(arr_image)},最小值{np.min(arr_image)}')
                    # 对每个标签进行裁剪和保存
                    for label in labels:
                        boundary = get_boundary_for_label(img_data, label, [1,1,1], oldSpacing)

                        x_min, x_max, y_min, y_max, z_min, z_max = boundary
                        print(f"标签 {label} 的范围: X轴({x_min}-{x_max}), Y轴({y_min}-{y_max}), Z轴({z_min}-{z_max})")
                        # 获取裁剪区域
                        cropped_image = get_centered_crop(resampled_image, x_min, x_max, y_min, y_max, z_min, z_max)

                        array = sitk.GetArrayFromImage(cropped_image)
                        # array = cropped_image

                        # # 示例调用
                        # output_folder = f'output_slices/{patient_id}'
                        # # 假设 cropped_image 是 3D NumPy 数组
                        # save_slices_as_images(array, output_folder)

                        print(f'裁剪范围最大值{np.max(array)},最小值{np.min(array)}')

                        assert np.min(array) != np.max(array), '归一化最大最小值不同'

                        # 正常归一化
                        normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))


                        cropped_image = sitk.GetImageFromArray(normalized_array)

                        # 保存裁剪的 3D 图像
                        save_path = f"{crop_path}/{dataset_class}/{patient_id}/label{int(label)}.npy"
                        save_cropped_to_npy(cropped_image, save_path)
                    all_label += len(labels)
                    print(f"患者{patient_id}的{len(labels)}个肺结节完成切片")

    return len(patients_ids), all_label, unique_patients

if __name__ == "__main__":
    crop_path = "cropped_data"
    dataset_class = "External test set"
    patients_num, all_labels, unique_patients = crop(dataset_class)
    print(f"{dataset_class}集{patients_num}个患者的{all_labels}个病灶已完成裁剪")
    print(unique_patients)