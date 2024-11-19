import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import scipy.ndimage
from PIL import Image
import torchvision.transforms.functional as TF
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from pydicom.misc import is_dicom
import pydicom


class NSCLCDataset(Dataset):
    def __init__(self, csv_path, images_path, preprocess=None, minority_transform=None):
        """
        Dataset class for NSCLC data.

        Args:
            csv_path (str): Path to the clinical CSV file.
            images_path (str): Path to the directory containing patient images.
            preprocess (callable, optional): Preprocessing function to apply to the volumes.
            minority_transform (callable, optional): Transformation to apply to minority class samples.
        """
        self.images_path = images_path
        self.preprocess = preprocess
        self.minority_transform = minority_transform

        # Load and clean the clinical data
        self.clinical_data = self._load_and_clean_data(csv_path)

    def __len__(self):
        return len(self.clinical_data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (patient_volume, dead_status)
                - patient_volume (torch.Tensor): The preprocessed volume data.
                - dead_status (torch.Tensor): The label indicating the dead status.
        """
        row = self.clinical_data.iloc[idx]
        patient_id = row['PatientID']
        dead_status = row['deadstatus.event']

        # Load the patient's volume
        patient_folder = os.path.join(self.images_path, patient_id)
        image_folder = self._find_image_folder(patient_folder)
        patient_volume = self._load_dicom_volume(image_folder)

        # Verify the loaded volume
        if patient_volume.size == 0:
            print(f"Warning: Empty DICOM volume for patient {patient_id} in folder {image_folder}")
            return None  # Ignore this sample if the volume is empty

        # Load the segmentation mask
        segmentation_folder = self._find_segmentation_folder(patient_folder)
        tumor_mask = self._load_segmentation_mask(segmentation_folder)

        # Adjust dimensions if necessary
        if len(tumor_mask.shape) != 3 and tumor_mask.shape[0] == 1:
            tumor_mask = tumor_mask[0]

        # Crop the volume to the tumor region
        patient_volume = self._crop_to_tumor_box(patient_volume, tumor_mask)

        # Verify after cropping
        if patient_volume.size == 0:
            print(f"Warning: Empty cropped volume for patient {patient_id}")
            return None  # Ignore this sample if the volume is empty

        # Apply preprocessing
        if self.preprocess:
            patient_volume = self.preprocess(patient_volume)

        # Apply specific transformation if the class is minority
        if dead_status == 0 and self.minority_transform:
            patient_volume = self.minority_transform(patient_volume)

        # Convert data to PyTorch tensors
        patient_volume = torch.tensor(patient_volume, dtype=torch.float32)
        dead_status = torch.tensor(dead_status, dtype=torch.long)

        return patient_volume, dead_status

    def _load_and_clean_data(self, csv_path):
        """
        Load the CSV file and remove patients without segmentation data.

        Args:
            csv_path (str): Path to the clinical CSV file.

        Returns:
            pd.DataFrame: Filtered clinical data.
        """
        clinical_data = pd.read_csv(csv_path)

        # List of indices for patients with valid segmentation data
        valid_indices = []

        for idx, row in clinical_data.iterrows():
            patient_id = row['PatientID']
            patient_folder = os.path.join(self.images_path, patient_id)
            segmentation_folder = self._find_segmentation_folder(patient_folder)

            if segmentation_folder is not None:  # Add only patients with segmentation
                valid_indices.append(idx)

        # Return only the filtered data
        return clinical_data.loc[valid_indices].reset_index(drop=True)

    def get_cleaned_data(self):
        """
        Get the cleaned clinical data.

        Returns:
            pd.DataFrame: The cleaned clinical data.
        """
        return self.clinical_data

    def _find_image_folder(self, patient_folder):
        """
        Find the folder containing the patient's CT scan images.

        Args:
            patient_folder (str): Path to the patient's directory.

        Returns:
            str: Path to the image folder.

        Raises:
            FileNotFoundError: If no valid image folder is found.
        """
        # Find all subdirectories in the patient's folder
        subdirs = [
            d for d in sorted(os.listdir(patient_folder))
            if os.path.isdir(os.path.join(patient_folder, d))
        ]

        # Traverse the subdirectories to find the one that contains multiple files
        for subdir in subdirs:
            first_level_folder = os.path.join(patient_folder, subdir)

            # Check inner subdirectories
            inner_subdirs = [
                d for d in sorted(os.listdir(first_level_folder))
                if os.path.isdir(os.path.join(first_level_folder, d))
            ]
            for inner_subdir in inner_subdirs:
                selected_folder = os.path.join(first_level_folder, inner_subdir)
                # Check the number of files in this folder
                files_in_folder = os.listdir(selected_folder)
                if len(files_in_folder) > 1:  # Choose the folder containing more than one file
                    return selected_folder

        # If no folder was found, raise an error
        raise FileNotFoundError(
            f"Unable to find a valid CT-scan folder with multiple files in {patient_folder}"
        )

    def _find_segmentation_folder(self, patient_folder):
        """
        Find the folder containing the segmentation data.

        Args:
            patient_folder (str): Path to the patient's directory.

        Returns:
            str or None: Path to the segmentation folder, or None if not found.
        """
        # Find all subdirectories in the patient's folder
        subdirs = [
            d for d in sorted(os.listdir(patient_folder))
            if os.path.isdir(os.path.join(patient_folder, d))
        ]

        if len(subdirs) < 1:
            return None

        first_level_folder = os.path.join(patient_folder, subdirs[0])
        inner_subdirs = [
            d for d in sorted(os.listdir(first_level_folder))
            if os.path.isdir(os.path.join(first_level_folder, d))
        ]

        # Find the folder that contains 'Segmentation' in its name
        segmentation_folder = next(
            (d for d in inner_subdirs if 'Segmentation' in d), None
        )

        if segmentation_folder is None:
            return None

        return os.path.join(first_level_folder, segmentation_folder)

    def _load_dicom_volume(self, image_folder):
        """
        Load the DICOM volume from the specified folder.

        Args:
            image_folder (str): Path to the folder containing DICOM images.

        Returns:
            np.ndarray: 3D numpy array representing the volume.

        Raises:
            ValueError: If no valid DICOM images are found.
        """
        image_slices = []
        for filename in sorted(os.listdir(image_folder)):
            file_path = os.path.join(image_folder, filename)
            if os.path.isfile(file_path) and is_dicom(file_path):
                try:
                    dicom_data = pydicom.dcmread(file_path, force=True)
                    image_array = dicom_data.pixel_array.astype(np.float32)
                    image_slices.append(image_array)
                except InvalidDicomError:
                    print(f"Skipped invalid DICOM file: {file_path}")

        if not image_slices:
            raise ValueError(f"No valid DICOM images found in {image_folder}")

        volume = np.stack(image_slices, axis=0)
        return volume

    def _load_segmentation_mask(self, segmentation_folder):
        """
        Load the segmentation mask from the specified folder.

        Args:
            segmentation_folder (str): Path to the folder containing segmentation masks.

        Returns:
            np.ndarray: 3D numpy array representing the segmentation mask.
        """
        mask_slices = []
        for filename in sorted(os.listdir(segmentation_folder)):
            file_path = os.path.join(segmentation_folder, filename)
            try:
                dicom_data = pydicom.dcmread(file_path, force=True)
                mask = dicom_data.pixel_array.astype(np.float32)
                mask_slices.append(mask)
            except InvalidDicomError:
                print(f"Skipped invalid DICOM file in segmentation: {file_path}")

        mask_volume = np.stack(mask_slices, axis=0)
        return mask_volume

    def _crop_to_tumor_box(self, volume, mask, padding=10):
        """
        Crop the volume around the tumor region.

        Args:
            volume (np.ndarray): 3D array of the volume.
            mask (np.ndarray): 3D array of the segmentation mask.
            padding (int, optional): Number of pixels to pad around the tumor.

        Returns:
            np.ndarray: Cropped volume.

        Raises:
            ValueError: If the tumor mask is empty.
        """
        if mask.shape[0] == 1:
            mask = mask[0]

        tumor_coords = np.argwhere(mask)

        if tumor_coords.size == 0:
            raise ValueError("Empty tumor mask. No tumor coordinates found.")

        min_z, min_y, min_x = tumor_coords.min(axis=0)
        max_z, max_y, max_x = tumor_coords.max(axis=0)

        min_z = max(min_z - padding, 0)
        max_z = min(max_z + padding, volume.shape[0])
        min_y = max(min_y - padding, 0)
        max_y = min(max_y + padding, volume.shape[1])
        min_x = max(min_x - padding, 0)
        max_x = min(max_x + padding, volume.shape[2])

        cropped_volume = volume[min_z:max_z, min_y:max_y, min_x:max_x]
        return cropped_volume


class CTPreprocess:
    def __init__(self, window_center=0, window_width=2000, target_size=(128, 128), target_slices=100):
        """
        Initialize the preprocessing class for DICOM volumes with optional resizing.

        Args:
            window_center (int, optional): Center of the window for clipping. Defaults to 0.
            window_width (int, optional): Width of the window for clipping. Defaults to 2000.
            target_size (tuple, optional): Target size (width, height) for resizing each slice. Defaults to (128, 128).
            target_slices (int, optional): Target number of slices for the volume. Defaults to 100.
        """
        self.window_center = window_center
        self.window_width = window_width
        self.target_size = target_size
        self.target_slices = target_slices

    def __call__(self, volume):
        """
        Apply the preprocessing steps to the input volume.

        Args:
            volume (np.ndarray or torch.Tensor): The input 3D volume data.

        Returns:
            np.ndarray: The preprocessed volume.
        """
        # Apply windowing (clipping)
        volume = self.apply_windowing(volume, self.window_center, self.window_width)

        # Normalize to [0, 1]
        volume = self.normalize(volume)

        # Resize each slice
        volume = self.resize_slices(volume, self.target_size)

        # Adjust the number of slices to the target count
        volume = self.resize_volume_slices(volume, self.target_slices)

        return volume

    def apply_windowing(self, volume, window_center, window_width):
        """
        Apply windowing to the volume by clipping the intensity values.

        Args:
            volume (np.ndarray): The input volume.
            window_center (int): Center of the window.
            window_width (int): Width of the window.

        Returns:
            np.ndarray: The windowed volume.
        """
        min_value = window_center - (window_width / 2)
        max_value = window_center + (window_width / 2)
        volume = np.clip(volume, min_value, max_value)
        return volume

    def normalize(self, volume):
        """
        Normalize the volume to the range [0, 1].

        Args:
            volume (np.ndarray): The input volume.

        Returns:
            np.ndarray: The normalized volume.
        """
        min_val, max_val = volume.min(), volume.max()
        if max_val > min_val:  # Avoid division by zero
            volume = (volume - min_val) / (max_val - min_val)
        return volume

    def resize_slices(self, volume, target_size):
        """
        Resize each slice in the volume to the target size.

        Args:
            volume (np.ndarray or torch.Tensor): The input volume.
            target_size (tuple): The desired size (width, height) for each slice.

        Returns:
            np.ndarray: The volume with resized slices.
        """
        # Convert to numpy array if input is a torch Tensor
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()

        resized_slices = [
            cv2.resize(slice_data, target_size, interpolation=cv2.INTER_LINEAR)
            for slice_data in volume
        ]
        return np.stack(resized_slices, axis=0)

    def resize_volume_slices(self, volume, target_slices):
        """
        Adjust the number of slices in the volume to the target count.

        Args:
            volume (np.ndarray): The input volume.
            target_slices (int): The desired number of slices.

        Returns:
            np.ndarray: The volume with adjusted number of slices.
        """
        current_slices = volume.shape[0]

        if current_slices < target_slices:
            # Padding if the volume has fewer slices than target
            padding_slices = target_slices - current_slices
            padding_shape = ((0, padding_slices), (0, 0), (0, 0))
            volume = np.pad(volume, pad_width=padding_shape, mode='constant', constant_values=0)
        elif current_slices > target_slices:
            # Interpolation if the volume has more slices than target
            zoom_factors = (target_slices / current_slices, 1, 1)
            volume = scipy.ndimage.zoom(volume, zoom=zoom_factors, order=1)

        return volume


class SafeMinorityTransform:
    def __init__(self, rotation_range=(-10, 10), vertical_flip_prob=0.5):
        """
        Transformation class for data augmentation on minority class samples.

        Args:
            rotation_range (tuple, optional): Range of degrees for random rotations. Defaults to (-10, 10).
            vertical_flip_prob (float, optional): Probability of applying a vertical flip. Defaults to 0.5.
        """
        self.rotation_range = rotation_range
        self.vertical_flip_prob = vertical_flip_prob

    def __call__(self, volume):
        """
        Apply transformations to the input volume.

        Args:
            volume (np.ndarray): 3D numpy array representing the volume.

        Returns:
            np.ndarray: Transformed volume.
        """
        # Apply a slight random rotation to each slice
        angle = random.uniform(*self.rotation_range)
        rotated_slices = []
        for slice_data in volume:
            # Convert each slice to a PIL image, apply rotation, then convert back to array
            slice_img = Image.fromarray(slice_data)
            rotated_img = TF.rotate(slice_img, angle)
            rotated_slices.append(np.array(rotated_img))
        volume = np.stack(rotated_slices, axis=0)

        # Apply vertical flip with a certain probability
        if random.random() < self.vertical_flip_prob:
            volume = np.flip(volume, axis=1).copy()  # Copy to avoid negative strides after flip

        return volume