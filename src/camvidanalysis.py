import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from collections import Counter


def analyze_camvid_dataset(base_path='/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/data/CamVid'):
    """
    Analyze the CamVid dataset to understand its structure and class distribution.
    
    Parameters:
    base_path (str): Path to the CamVid dataset
    """
    print("="*50)
    print("CamVid Dataset Analysis")
    print("="*50)
    
    # 1. Dataset structure analysis
    train_dir = os.path.join(base_path, '/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/data/CamVid/train')
    train_labels_dir = os.path.join(base_path, '/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/data/CamVid/train_labels')
    val_dir = os.path.join(base_path, '/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/data/CamVid/val')
    val_labels_dir = os.path.join(base_path, '/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/data/CamVid/val_labels')
    test_dir = os.path.join(base_path, '/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/data/CamVid/test')
    test_labels_dir = os.path.join(base_path, '/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/data/CamVid/test_labels')
    
    train_files = os.listdir(train_dir)
    val_files = os.listdir(val_dir)
    test_files = os.listdir(test_dir)
    
    print(f"Number of training images: {len(train_files)}")
    print(f"Number of validation images: {len(val_files)}")
    print(f"Number of test images: {len(test_files)}")
    print(f"Total images: {len(train_files) + len(val_files) + len(test_files)}")
    print()
    
    # 2. Image dimensions analysis
    sample_img_path = os.path.join(train_dir, train_files[0])
    sample_img = Image.open(sample_img_path)
    width, height = sample_img.size
    print(f"Image dimensions: {width}x{height}")
    print()
    
    # 3. Read class mapping file
    class_map_path = os.path.join(base_path, 'class_dict.csv')
    class_map_df = pd.read_csv(class_map_path)
    
    # Filter for the 5 relevant classes
    target_classes = ['Car', 'Pedestrian', 'Bicyclist', 'MotorcycleScooter', 'Truck_Bus']
    filtered_class_map = class_map_df[class_map_df['name'].isin(target_classes)]
    
    print("Target classes with their RGB values:")
    for idx, row in filtered_class_map.iterrows():
        print(f"{row['name']}: RGB({row['r']}, {row['g']}, {row['b']})")
    print()
    
    # 4. Class distribution analysis for each dataset split
    print("\n" + "="*50)
    print("TRAINING SET ANALYSIS")
    print("="*50)
    train_distribution = analyze_class_distribution(train_dir, train_labels_dir, 
                                                  filtered_class_map, target_classes)
    
    print("\n" + "="*50)
    print("VALIDATION SET ANALYSIS")
    print("="*50)
    val_distribution = analyze_class_distribution(val_dir, val_labels_dir, 
                                                filtered_class_map, target_classes)
    
    print("\n" + "="*50)
    print("TEST SET ANALYSIS")
    print("="*50)
    test_distribution = analyze_class_distribution(test_dir, test_labels_dir, 
                                                 filtered_class_map, target_classes)
    
    # 5. Visualize the distributions
    print("\n" + "="*50)
    print("VISUALIZING CLASS DISTRIBUTIONS")
    print("="*50)
    
    # Compare class distributions across splits
    compare_set_distributions(train_distribution, val_distribution, test_distribution, target_classes)
    
    # 6. Visualize sample images
    visualize_samples(train_dir, train_labels_dir, filtered_class_map, 3)
    
    return {
        'train': train_distribution,
        'val': val_distribution,
        'test': test_distribution
    }

def analyze_class_distribution(img_dir, mask_dir, class_map_df, target_classes, sample_size=None):
    """
    Analyze the distribution of classes in the dataset.
    
    Parameters:
    img_dir (str): Directory containing images
    mask_dir (str): Directory containing mask labels
    class_map_df (DataFrame): Dataframe with class RGB mappings
    target_classes (list): List of target class names
    sample_size (int): Number of images to sample for analysis (None = use all)
    
    Returns:
    dict: Class distribution statistics
    """
    # Get list of mask files
    mask_files = os.listdir(mask_dir)
    
    # Sample a subset to speed up analysis
    if sample_size and sample_size < len(mask_files):
        print(f"Analyzing class distribution (sampling {sample_size} out of {len(mask_files)} images)...")
        mask_files = np.random.choice(mask_files, sample_size, replace=False)
    else:
        print(f"Analyzing class distribution (all {len(mask_files)} images)...")
    
    # Create a dict to store class RGB values
    class_rgb = {}
    for idx, row in class_map_df.iterrows():
        class_rgb[row['name']] = (row['r'], row['g'], row['b'])
    
    # Count occurrences and pixels per class
    class_occurrences = {cls: 0 for cls in target_classes}
    class_pixels = {cls: 0 for cls in target_classes}
    class_instances = {cls: 0 for cls in target_classes}  # Count separate instances
    total_pixels = 0
    
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Count class occurrences and pixels
        for cls in target_classes:
            rgb = class_rgb[cls]
            # Create a binary mask for this class
            binary_mask = np.all(mask == rgb, axis=2)
            pixel_count = np.sum(binary_mask)
            
            if pixel_count > 0:
                class_occurrences[cls] += 1
                class_pixels[cls] += pixel_count
                
                # Count separate instances (connected components)
                binary_mask_np = binary_mask.astype(np.uint8)
                num_labels, _ = cv2.connectedComponents(binary_mask_np)
                class_instances[cls] += num_labels - 1  # Subtract 1 for background
            
            total_pixels += mask.shape[0] * mask.shape[1]
    
    # Calculate percentages
    class_pct = {cls: (pixels / total_pixels) * 100 for cls, pixels in class_pixels.items()}
    
    # Calculate average instance size
    avg_instance_size = {}
    for cls in target_classes:
        if class_instances[cls] > 0:
            avg_instance_size[cls] = class_pixels[cls] / class_instances[cls]
        else:
            avg_instance_size[cls] = 0
    
    print("Class occurrences (number of images containing each class):")
    for cls, count in class_occurrences.items():
        print(f"{cls}: {count}/{len(mask_files)} images ({count/len(mask_files)*100:.2f}%)")
    
    print("\nClass pixel distribution:")
    for cls, pct in class_pct.items():
        print(f"{cls}: {pct:.2f}% of pixels")
    
    print("\nClass instance counts and average sizes:")
    for cls, count in class_instances.items():
        print(f"{cls}: {count} instances, avg size: {avg_instance_size[cls]:.1f} pixels")
    
    return {
        'occurrences': class_occurrences,
        'sample_size': len(mask_files),
        'pixel_percentages': class_pct,
        'instances': class_instances,
        'avg_instance_size': avg_instance_size
    }

def visualize_class_distribution(class_dist):
    """
    Visualize the class distribution as a bar chart.
    
    Parameters:
    class_dist (dict): Class distribution statistics
    """
    print("\nVisualizing class distribution...")
    
    # Class occurrences
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    
    classes = list(class_dist['occurrences'].keys())
    occurrences = [class_dist['occurrences'][cls] for cls in classes]
    
    # Percentage of images containing each class
    percentages = [count/class_dist['sample_size']*100 for count in occurrences]
    
    plt.bar(classes, percentages, color='skyblue')
    plt.title('Percentage of Images Containing Each Class')
    plt.ylabel('% of Images')
    plt.xticks(rotation=45)
    
    # Pixel percentages
    plt.subplot(1, 2, 2)
    pixel_pct = [class_dist['pixel_percentages'][cls] for cls in classes]
    plt.bar(classes, pixel_pct, color='lightgreen')
    plt.title('Percentage of Pixels Per Class')
    plt.ylabel('% of Pixels')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def compare_set_distributions(train_dist, val_dist, test_dist, target_classes):
    """
    Compare class distributions across train, validation, and test sets.
    
    Parameters:
    train_dist (dict): Training set distribution statistics
    val_dist (dict): Validation set distribution statistics
    test_dist (dict): Test set distribution statistics
    target_classes (list): List of target class names
    """
    # 1. Compare image occurrences
    plt.figure(figsize=(15, 10))
    
    # Image occurrence comparison
    plt.subplot(2, 1, 1)
    
    width = 0.25  # width of the bars
    x = np.arange(len(target_classes))
    
    # Calculate percentages
    train_pct = [train_dist['occurrences'][cls]/train_dist['sample_size']*100 for cls in target_classes]
    val_pct = [val_dist['occurrences'][cls]/val_dist['sample_size']*100 for cls in target_classes]
    test_pct = [test_dist['occurrences'][cls]/test_dist['sample_size']*100 for cls in target_classes]
    
    # Create bars
    plt.bar(x - width, train_pct, width, label='Train', color='blue', alpha=0.7)
    plt.bar(x, val_pct, width, label='Validation', color='green', alpha=0.7)
    plt.bar(x + width, test_pct, width, label='Test', color='red', alpha=0.7)
    
    plt.xlabel('Classes')
    plt.ylabel('% of Images')
    plt.title('Percentage of Images Containing Each Class Across Datasets')
    plt.xticks(x, target_classes, rotation=45)
    plt.legend()
    
    # 2. Compare pixel distributions
    plt.subplot(2, 1, 2)
    
    # Get pixel percentages
    train_pixels = [train_dist['pixel_percentages'][cls] for cls in target_classes]
    val_pixels = [val_dist['pixel_percentages'][cls] for cls in target_classes]
    test_pixels = [test_dist['pixel_percentages'][cls] for cls in target_classes]
    
    # Create bars
    plt.bar(x - width, train_pixels, width, label='Train', color='blue', alpha=0.7)
    plt.bar(x, val_pixels, width, label='Validation', color='green', alpha=0.7)
    plt.bar(x + width, test_pixels, width, label='Test', color='red', alpha=0.7)
    
    plt.xlabel('Classes')
    plt.ylabel('% of Pixels')
    plt.title('Percentage of Pixels Per Class Across Datasets')
    plt.xticks(x, target_classes, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 3. Print a table with complete numeric data
    print("\nDetailed Class Distribution Across Datasets (% of pixels):")
    print("="*80)
    print(f"{'Class':<20} {'Train %':>10} {'Validation %':>15} {'Test %':>10} {'Overall Balance':>20}")
    print("-"*80)
    
    for cls in target_classes:
        train_pct = train_dist['pixel_percentages'][cls]
        val_pct = val_dist['pixel_percentages'][cls]
        test_pct = test_dist['pixel_percentages'][cls]
        
        # Calculate a simple balance measure (standard deviation across splits)
        balance = np.std([train_pct, val_pct, test_pct])
        balance_indicator = "Good" if balance < 1.0 else "Moderate" if balance < 2.0 else "Poor"
        
        print(f"{cls:<20} {train_pct:>10.2f} {val_pct:>15.2f} {test_pct:>10.2f} {balance_indicator:>20}")
    
    print("="*80)

def visualize_samples(img_dir, mask_dir, class_map_df, num_samples=3):
    """
    Visualize sample images with their corresponding masks.
    
    Parameters:
    img_dir (str): Directory containing images
    mask_dir (str): Directory containing mask labels
    class_map_df (DataFrame): Dataframe with class RGB mappings
    num_samples (int): Number of samples to visualize
    """
    print("\nVisualizing sample images with masks...")
    
    # Get list of image files
    img_files = os.listdir(img_dir)
    
    # Randomly select samples
    samples = np.random.choice(img_files, min(num_samples, len(img_files)), replace=False)
    
    plt.figure(figsize=(15, 5*num_samples))
    
    for i, img_file in enumerate(samples):
        # Get image and mask paths
        img_path = os.path.join(img_dir, img_file)
        mask_file = img_file.split('.')[0] + '_L.png'
        mask_path = os.path.join(mask_dir, mask_file)
        
        # Read image and mask
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Display image and mask
        plt.subplot(num_samples, 2, i*2+1)
        plt.imshow(img)
        plt.title(f'Image: {img_file}')
        plt.axis('off')
        
        plt.subplot(num_samples, 2, i*2+2)
        plt.imshow(mask)
        plt.title(f'Mask: {mask_file}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Adjust the base path according to your environment
    base_path = '/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/data/CamVid'
    
    # Check if the directory exists
    if not os.path.exists(base_path):
        print(f"Warning: {base_path} does not exist!")
        print("You need to modify the base_path variable to point to your CamVid dataset.")
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found dataset at: {path}")
                base_path = path
                break
    
    # Run the analysis
    distribution_data = analyze_camvid_dataset(base_path)
    
    # Save the results to CSV files
    print("\nSaving class distribution data to CSV files...")
    
    # Create a directory for results if it doesn't exist
    os.makedirs("analysis_results", exist_ok=True)
    
    # Save pixel distribution data
    pixel_data = {
        'Class': list(distribution_data['train']['pixel_percentages'].keys()),
        'Train (%)': [distribution_data['train']['pixel_percentages'][cls] for cls in distribution_data['train']['pixel_percentages']],
        'Validation (%)': [distribution_data['val']['pixel_percentages'][cls] for cls in distribution_data['train']['pixel_percentages']],
        'Test (%)': [distribution_data['test']['pixel_percentages'][cls] for cls in distribution_data['train']['pixel_percentages']]
    }
    
    pixel_df = pd.DataFrame(pixel_data)
    pixel_df.to_csv("analysis_results/pixel_distribution.csv", index=False)
    
    # Save instance data
    instance_data = {
        'Class': list(distribution_data['train']['instances'].keys()),
        'Train Instances': [distribution_data['train']['instances'][cls] for cls in distribution_data['train']['instances']],
        'Validation Instances': [distribution_data['val']['instances'][cls] for cls in distribution_data['train']['instances']],
        'Test Instances': [distribution_data['test']['instances'][cls] for cls in distribution_data['train']['instances']],
        'Train Avg Size': [distribution_data['train']['avg_instance_size'][cls] for cls in distribution_data['train']['avg_instance_size']],
        'Validation Avg Size': [distribution_data['val']['avg_instance_size'][cls] for cls in distribution_data['train']['avg_instance_size']],
        'Test Avg Size': [distribution_data['test']['avg_instance_size'][cls] for cls in distribution_data['train']['avg_instance_size']]
    }
    
    instance_df = pd.DataFrame(instance_data)
    instance_df.to_csv("analysis_results/instance_statistics.csv", index=False)
    
    print("Analysis completed and results saved to 'analysis_results' directory.")