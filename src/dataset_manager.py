import os


class CamVidDatasetManager:
    """
    A class to manage paths and access to the CamVid dataset.
    Automatically handles path creation and provides easy access to file lists and data pairs.
    """
    def __init__(self, base_dir='/home/roboticsstudent/Documents/COMP0248_RamneekAhluwalia/data/CamVid'):
        # Base directory for the CamVid dataset
        self.base_dir = base_dir
        
        # Define subdirectories
        self.train_dir = os.path.join(base_dir, 'train')
        self.train_labels_dir = os.path.join(base_dir, 'train_labels')
        self.val_dir = os.path.join(base_dir, 'val')
        self.val_labels_dir = os.path.join(base_dir, 'val_labels')
        self.test_dir = os.path.join(base_dir, 'test')
        self.test_labels_dir = os.path.join(base_dir, 'test_labels')
        
        # Validate directories exist
        self._validate_directories()
        
        # Initialize file lists
        self._init_file_lists()
        
        # Initialize data pairs
        self._init_data_pairs()
    
    def _validate_directories(self):
        """Validate that all required directories exist."""
        required_dirs = [
            self.train_dir, self.train_labels_dir,
            self.val_dir, self.val_labels_dir,
            self.test_dir, self.test_labels_dir
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory not found: {directory}")
    
    def _init_file_lists(self):
        """Initialize file lists for train, val, and test sets."""
        self.train_files = os.listdir(self.train_dir)
        self.val_files = os.listdir(self.val_dir)
        self.test_files = os.listdir(self.test_dir)
    
    def _create_pairs(self, files, image_dir, mask_dir):
        """
        Create pairs of image and mask file paths.
        
        Args:
            files: List of image filenames
            image_dir: Directory path for images
            mask_dir: Directory path for masks
            
        Returns:
            List of tuples containing (image_path, mask_path)
        """
        pairs = []
        for img_file in files:
            img_path = os.path.join(image_dir, img_file)
            mask_file = f"{os.path.splitext(img_file)[0]}_L.png"
            mask_path = os.path.join(mask_dir, mask_file)
            pairs.append((img_path, mask_path))
        return pairs
    
    def _init_data_pairs(self):
        """Initialize data pairs for train, val, and test sets."""
        self.train_pairs = self._create_pairs(self.train_files, self.train_dir, self.train_labels_dir)
        self.val_pairs = self._create_pairs(self.val_files, self.val_dir, self.val_labels_dir)
        self.test_pairs = self._create_pairs(self.test_files, self.test_dir, self.test_labels_dir)
    
    def get_train_image_paths(self):
        """Get list of training image paths."""
        return [pair[0] for pair in self.train_pairs]
    
    def get_train_mask_paths(self):
        """Get list of training mask paths."""
        return [pair[1] for pair in self.train_pairs]
    
    def get_val_image_paths(self):
        """Get list of validation image paths."""
        return [pair[0] for pair in self.val_pairs]
    
    def get_val_mask_paths(self):
        """Get list of validation mask paths."""
        return [pair[1] for pair in self.val_pairs]
    
    def get_test_image_paths(self):
        """Get list of test image paths."""
        return [pair[0] for pair in self.test_pairs]
    
    def get_test_mask_paths(self):
        """Get list of test mask paths."""
        return [pair[1] for pair in self.test_pairs]
    

