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