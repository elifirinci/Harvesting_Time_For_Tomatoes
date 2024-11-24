# Veri setini yüklemek için bir fonksiyon
def load_custom_dataset(base_dir, image_size=(64, 64), validation_split=0.2, batch_size=32):
    dataset = []
    labels = []
    class_mapping = {'old': 0, 'ripe': 1,'unripe':2,'rotten':3}

    for class_name, label in class_mapping.items():
        class_dir = os.path.join(base_dir, class_name)
        for subdir in os.listdir(class_dir):
            subdir_path = os.path.join(class_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(subdir_path, filename)
                        try:
                            # Görüntüyü yükle ve yeniden boyutlandır
                            img = tf.keras.preprocessing.image.load_img(file_path, target_size=image_size)
                            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                            dataset.append(img_array)
                            labels.append(label)
                        except Exception as e:
                            print(f"Hata: {file_path}, {e}")
    
    dataset = np.array(dataset)
    labels = np.array(labels)

# Sınıf dağılımlarını yazdırma
def print_class_distribution(train_labels, val_labels):
    train_class_counts = Counter(train_labels)
    val_class_counts = Counter(val_labels)
    
    print("Training set class distribution:")
    print(f"Old: {train_class_counts[0]}, Ripe: {train_class_counts[1],Unripe: {train_class_counts[2],Rotten: {train_class_counts[3]}")
    
    print("\nValidation set class distribution:")
    print(f"Old: {val_class_counts[0]}, Ripe: {val_class_counts[1],Unripe: {val_class_counts[2],Rotten: {val_class_counts[3]}")

# Sınıf dengesini yazdırma
print_class_distribution(train_labels, val_labels)
