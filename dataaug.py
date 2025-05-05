# Define image augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

image_size = 224
augmented_folder = '/kaggle/working/augmented_data'
os.makedirs(augmented_folder, exist_ok=True)

# Augmentation and visualization
for label in labels:
    folderPath = os.path.join('/kaggle/input/herlev-dataset/Herlev Dataset/train', label)
    savePath = os.path.join(augmented_folder, label)
    os.makedirs(savePath, exist_ok=True)

    for img_name in os.listdir(folderPath):
        img_path = os.path.join(folderPath, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = np.expand_dims(img, axis=0)  # Add batch dimension for augmentation

        # Generate augmented images
        gen = datagen.flow(img, batch_size=1)
        for i in range(5):  # Generate 5 augmented images per original image
            augmented_img = next(gen)[0].astype('uint8')

            # Save augmented image
            augmented_name = f"aug_{i}_{img_name}"
            augmented_path = os.path.join(savePath, augmented_name)
            cv2.imwrite(augmented_path, augmented_img)

        # Visualize original and one augmented image
        if img_name.endswith(".BMP"):  # Adjust if other file types
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB))
            plt.title("Augmented Image")
            plt.axis("off")
            plt.show()
            break  # Display visualization for one image per label
