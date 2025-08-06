 # df_final.info()
    # # Display column names
    # print("\n=== Column Names ===")
    # print(df_final.columns.tolist())

    # # Show the first few rows
    # print("\n=== First 5 Rows ===")
    # print(df_final.head())

    # # Describe numerical columns
    # print("\n=== Summary Statistics (Numerical) ===")
    # print(df_final.describe())

    # # Check for missing values
    # print("\n=== Missing Values ===")
    # print(df_final.isnull().sum())

    # # Value counts for each categorical column (change if your class column is named differently)
    # print("\n=== Class Distribution ===")
    # for col in df_final.columns:
    #     if df_final[col].dtype == 'object' or df_final[col].nunique() < 20:
    #         print(f"\nValue counts for '{col}':")
    #         print(df_final[col].value_counts())

    # Normalize and plot the class distribution
    # class_counts = df_final['label'].value_counts(normalize=True)

    # plt.figure(figsize=(8, 5))
    # class_counts.plot(kind='bar', color='skyblue')
    # plt.title('Class Distribution')
    # plt.xlabel('Class Labels')
    # plt.ylabel('Proportion')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # # Save to file
    # plt.tight_layout()
    # plt.savefig('class_distribution.png', dpi=300)
    # plt.close()

    # print("Plot saved as 'class_distribution.png'")

 # # Replace this with your actual path
    # rename_image_files_with_underscores("/data2/users/koushani/chbmit/Eye_ML/RV_images_final")

    # patient_folder = "/data2/users/koushani/chbmit/Eye_ML/RV_images_final/CME/ACB_OU"
    # save_folder = "/data2/users/koushani/chbmit/Root/plots"
    # show_eye_pair(patient_folder, save_dir=save_folder)
    
    # df_final = build_classification_dataset("/data2/users/koushani/chbmit/data/allpatients_resized")
    # # resize_images(df_resized, output_root="/data2/users/koushani/chbmit/Eye_ML/RV_images_resized", target_size=(320, 320))
    # # print(df_final.tail(10))
    # df_final.to_csv("faa_image_classification_dataset_final.csv", index=False)
    # total_images = len(df_final)
    # print(f"Total number of data points (images): {total_images}")

    # # Create dataset
    # image_dataset = ImageDataset(df_final)
    # loader = DataLoader(image_dataset, batch_size=64, shuffle=False)

    # n_pixels = 0
    # sum_ = 0.0
    # sum_squared = 0.0
    # min_val = float('inf')
    # max_val = float('-inf')

    # for images, _ in loader:
    #     sum_ += images.sum()
    #     sum_squared += (images ** 2).sum()
    #     n_pixels += images.numel()
    #     min_val = min(min_val, images.min().item())
    #     max_val = max(max_val, images.max().item())

    # mean = sum_ / n_pixels
    # std = (sum_squared / n_pixels - mean**2).sqrt()

    # print(f"Dataset Statistics:")
    # print(f" - Total pixels      : {n_pixels}")
    # print(f" - Mean              : {mean.item():.4f}")
    # print(f" - Std Dev           : {std.item():.4f}")
    # print(f" - Min Pixel Value   : {min_val:.4f}")
    # print(f" - Max Pixel Value   : {max_val:.4f}")


# Convert tensor to numpy array and move channel to last dimension
   # image_np = image.permute(1, 2, 0).numpy()  # Now shape: [224, 224, 3]

    # # Plot using matplotlib
    # plt.imshow(image.squeeze(0), cmap='gray')  # squeeze channel for display
    # plt.title(f"Sample idx: {sample_idx_within_batch} | Label: {label}")
    # plt.axis('off')
    # plt.savefig(f'Sample idx_{sample_idx_within_batch}_image_grayscale.png', dpi=300)
    # plt.close()


 # idx = 10
    # row = df_final.iloc[idx]

    # print(row)

    # # Load and display the image
    # img_path = row['image_path']
    # label = row['label']

    # image = Image.open(img_path)

    # plt.imshow(image)
    # plt.title(f"Label: {label} | Eye: {row['eye']}")
    # plt.axis('off')
    # plt.savefig(f'Sample idx_{idx}_imagefromcsv.png', dpi=300)
    # plt.close()

    # # Convert to NumPy array
    # img_array = np.array(image)

    # # Print shape
    # print(f"Image shape: {img_array.shape}")  # (H, W, C)
    # # df_final.to_csv("faa_image_classification_dataset_final.csv", index=False)
   
    # # img_size = 224  # or 320 if that's your original image shape

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    # ])
    

    # df_train, df_test = train_test_split(df_final, test_size=0.1, stratify=df_final["label"], random_state=42)

    # df_train.to_csv("train.csv", index=False)
    # df_test.to_csv("test.csv", index=False)

    # train_dataset = EyeFFEDataset(csv_file="train.csv", transform=transform)
    # test_dataset = EyeFFEDataset(csv_file="test.csv", transform=transform)

    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # images_train, labels_train = next(iter(train_loader))
    # print(f"Image batch shape: {images_train.shape}")
    # print(f"Label batch shape: {labels_train.shape}")


    # images_test, labels_test = next(iter(test_loader))
    # print(f"Image batch shape: {images_test.shape}")
    # print(f"Label batch shape: {labels_test.shape}")

    # sample_idx = 12
    # images, labels = next(iter(train_loader))

    # # Pick sample at index 12
    # image = images[sample_idx]         # Tensor: [3, 224, 224]
    # label = labels[sample_idx]

    # print(f"Sample image shape: {image.shape}")  # Should be [3, 224, 224]

    # # Convert tensor to numpy array and move channel to last dimension
    # image_np = image.permute(1, 2, 0).numpy()  # Now shape: [224, 224, 3]

    # # Plot using matplotlib
    # plt.imshow(image_np)
    # plt.title(f"Sample idx: {sample_idx} | Label: {label}")
    # plt.axis('off')
    # plt.savefig(f'Sample idx_{sample_idx}_RGB_image.png', dpi=300)
    # plt.close()
