import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import os
import glob
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import zipfile
import tempfile
import shutil
from plant_disease_detector import UNet  # Import from our main file

class PlantDiseaseDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, img_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            self.image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        print(f"Found {len(self.image_files)} images in dataset")
        
        # If mask directory is provided, verify corresponding masks exist
        if mask_dir and os.path.exists(mask_dir):
            self.has_masks = True
            print(f"Using masks from: {mask_dir}")
        else:
            self.has_masks = False
            print("No mask directory found. Using synthetic masks for demonstration.")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Try with PIL if OpenCV fails
            image = np.array(Image.open(img_path))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Load or create mask
        if self.has_masks:
            # Try to find corresponding mask file
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = None
            
            # Look for mask with same name in mask directory
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                potential_mask = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(potential_mask):
                    mask_path = potential_mask
                    break
            
            if mask_path:
                mask = cv2.imread(mask_path, 0)  # Load as grayscale
                mask = cv2.resize(mask, (self.img_size, self.img_size))
                mask = mask.astype(np.float32) / 255.0
            else:
                mask = self._create_synthetic_mask(image)
        else:
            mask = self._create_synthetic_mask(image)
        
        # Convert to tensor
        image_tensor = torch.tensor(image).permute(2, 0, 1)  # HWC to CHW
        mask_tensor = torch.tensor(mask).unsqueeze(0)  # Add channel dimension
        
        return image_tensor, mask_tensor
    
    def _create_synthetic_mask(self, image):
        """Create synthetic mask for demonstration when real masks aren't available"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Create some random "diseased" areas for demonstration
        num_regions = np.random.randint(1, 5)
        for _ in range(num_regions):
            center_x = np.random.randint(0, w)
            center_y = np.random.randint(0, h)
            radius = np.random.randint(10, min(h, w) // 4)
            
            # Create circle
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Add to mask with some probability
            circle_mask = dist_from_center <= radius
            mask[circle_mask] = np.random.uniform(0.3, 1.0)
        
        return mask

def extract_zip(zip_path, extract_to):
    """Extract ZIP file to temporary directory"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted ZIP to: {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Failed to extract ZIP: {e}")
        return False

def find_image_and_mask_dirs(extracted_dir):
    """Find images and masks directories in extracted content"""
    image_dir = None
    mask_dir = None
    
    # Common directory structures
    possible_structures = [
        # Structure 1: root/images and root/masks
        (['images'], ['masks', 'labels']),
        # Structure 2: root/train/images and root/train/masks
        (['train/images', 'training/images'], ['train/masks', 'train/labels', 'training/masks']),
        # Structure 3: root/Images and root/Masks
        (['Images'], ['Masks', 'Labels']),
        # Structure 4: everything in root
        ([None], [None])
    ]
    
    for img_dirs, mask_dirs in possible_structures:
        for img_dir_pattern in img_dirs:
            for mask_dir_pattern in mask_dirs:
                # Determine paths to check
                if img_dir_pattern is None:
                    img_check_dir = extracted_dir
                else:
                    img_check_dir = os.path.join(extracted_dir, img_dir_pattern)
                
                if mask_dir_pattern is None:
                    mask_check_dir = extracted_dir
                else:
                    mask_check_dir = os.path.join(extracted_dir, mask_dir_pattern)
                
                # Check if directories exist and contain images
                if os.path.exists(img_check_dir):
                    image_files = []
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        image_files.extend(glob.glob(os.path.join(img_check_dir, ext)))
                    
                    if image_files:
                        image_dir = img_check_dir
                        mask_dir = mask_check_dir if os.path.exists(mask_check_dir) else None
                        print(f"✅ Found image directory: {image_dir}")
                        if mask_dir:
                            print(f"✅ Found mask directory: {mask_dir}")
                        return image_dir, mask_dir
    
    # If no structure found, use root directory
    if os.path.exists(extracted_dir):
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(extracted_dir, ext)))
        
        if image_files:
            print(f"✅ Using root directory: {extracted_dir}")
            return extracted_dir, None
    
    return None, None

def select_zip_file():
    """Open dialog to select ZIP file"""
    root = tk.Tk()
    root.withdraw()
    zip_path = filedialog.askopenfilename(
        title="Select Dataset ZIP File",
        filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
    )
    root.destroy()
    return zip_path

def setup_dataset_from_zip():
    """Interactive dataset setup from ZIP file"""
    print("\n" + "="*50)
    print("🌿 Plant Disease Model Training - ZIP Dataset")
    print("="*50)
    
    # Select ZIP file
    print("\n1. Select ZIP file containing plant images dataset:")
    zip_path = select_zip_file()
    
    if not zip_path:
        print("❌ No ZIP file selected. Exiting.")
        return None, None, None
    
    if not os.path.exists(zip_path):
        print("❌ Selected ZIP file does not exist.")
        return None, None, None
    
    # Create temporary directory for extraction
    temp_dir = tempfile.mkdtemp(prefix="plant_dataset_")
    print(f"📁 Extracting to temporary directory: {temp_dir}")
    
    # Extract ZIP
    if not extract_zip(zip_path, temp_dir):
        shutil.rmtree(temp_dir)
        return None, None, None
    
    # Find image and mask directories
    image_dir, mask_dir = find_image_and_mask_dirs(temp_dir)
    
    if not image_dir:
        print("❌ No images found in the ZIP file.")
        shutil.rmtree(temp_dir)
        return None, None, None
    
    # Count images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    print(f"📊 Found {len(image_files)} images in dataset")
    
    if mask_dir:
        mask_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            mask_files.extend(glob.glob(os.path.join(mask_dir, ext)))
        print(f"📊 Found {len(mask_files)} mask images")
    else:
        print("💡 No mask directory found. Using synthetic masks for training.")
    
    return image_dir, mask_dir, temp_dir

def train_model():
    """Main training function with ZIP support"""
    
    # Setup dataset from ZIP
    image_dir, mask_dir, temp_dir = setup_dataset_from_zip()
    if not image_dir:
        return
    
    try:
        # Create dataset
        dataset = PlantDiseaseDataset(image_dir, mask_dir, img_size=256)
        
        if len(dataset) == 0:
            print("❌ No valid images found for training.")
            return
        
        # Split dataset (80% train, 20% validation)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"\n📊 Dataset split:")
        print(f"   - Training samples: {len(train_dataset)}")
        print(f"   - Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        batch_size = 4
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")
        
        model = UNet(n_channels=3, n_classes=1)
        model.to(device)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Training parameters
        num_epochs = 20
        print(f"\n🚀 Starting training for {num_epochs} epochs...")
        
        # Track losses
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Update learning rate
            scheduler.step()
            
            print(f'Epoch [{epoch+1:02d}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save trained model
        model_path = 'plant_disease_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f"\n✅ Model saved as: {model_path}")
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        print("📊 Training history plot saved as: training_history.png")
        
        # Show sample predictions
        show_sample_predictions(model, val_dataset, device)
        
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directory: {temp_dir}")

def show_sample_predictions(model, dataset, device, num_samples=3):
    """Show sample predictions from the trained model"""
    print("\n🔍 Generating sample predictions...")
    
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Get random sample
        idx = torch.randint(0, len(dataset), (1,)).item()
        image, true_mask = dataset[idx]
        
        # Prediction
        with torch.no_grad():
            input_tensor = image.unsqueeze(0).to(device)
            prediction = model(input_tensor)
            pred_mask = torch.sigmoid(prediction).squeeze().cpu().numpy()
        
        # Convert tensors to numpy for plotting
        image_np = image.permute(1, 2, 0).numpy()
        true_mask_np = true_mask.squeeze().numpy()
        
        # Plot
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(true_mask_np, cmap='gray')
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    print("📸 Sample predictions saved as: sample_predictions.png")

def main():
    """Main function with menu"""
    print("🌿 Plant Disease Detection Model Trainer (ZIP Support)")
    print("="*50)
    
    while True:
        print("\nOptions:")
        print("1. 🚀 Train new model from ZIP file")
        print("2. 📊 View training info")
        print("3. ❌ Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\nStarting model training from ZIP...")
            train_model()
            
        elif choice == '2':
            print("\nTraining Information:")
            print("• ZIP File Structure (any of these):")
            print("  - root/images/ and root/masks/")
            print("  - root/train/images/ and root/train/masks/")
            print("  - root/Images/ and root/Masks/")
            print("  - All images in root directory")
            print("• Supported formats: JPG, PNG, JPEG")
            print("• Mask Format: Grayscale images where white = diseased, black = healthy")
            print("• Model: U-Net architecture for semantic segmentation")
            print("• Output: Trained model saved as 'plant_disease_model.pth'")
            
        elif choice == '3':
            print("👋 Exiting...")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    main()