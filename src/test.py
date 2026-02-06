def show_test_predictions(test_loader, model, device, optim, data, num_images=5):
    torch.cuda.empty_cache()  # Clear GPU cache before starting
    model.eval()  # Set model to evaluation mode
    fig, axs = plt.subplots(3, num_images, figsize=(10, 8))
    
    # Get a batch of images that's large enough
    images, masks = next(iter(test_loader))
    batch_size = images.size(0)
    
    # Generate unique random indices
    selected_indices = random.sample(range(batch_size), num_images)
    
    with torch.no_grad():  # No need for gradients in evaluation
        for i, idx in enumerate(selected_indices):
            # Use the randomly selected index
            image = images[idx].to(device)
            mask = masks[idx].to(device)
            
            # Get the predicted mask from the model
            output,_ = model(image.unsqueeze(0))  # Add batch dimension (1, C, H, W)
            predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()  # Apply sigmoid and remove batch dim
            
            # Convert image and mask tensors to numpy arrays for visualization
            image_np = image.cpu().permute(1, 2, 0).numpy()  # Convert CHW to HWC
            mask_np = mask.cpu().squeeze().numpy()  # Convert mask to numpy
            
            # Plot the original image
            axs[0, i].imshow(image_np)
            axs[0, i].set_title(f"Image {i+1}")
            axs[0, i].axis('off')
            
            # Plot the ground truth mask
            axs[1, i].imshow(mask_np, cmap='gray')
            axs[1, i].set_title(f"Ground Truth {i+1}")
            axs[1, i].axis('off')
            
            # Plot the predicted mask
            axs[2, i].imshow(predicted_mask, cmap='gray')
            axs[2, i].set_title(f"Predicted Mask {i+1}")
            axs[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    fig.savefig(f'{data}_predictions_{optim}.png', dpi=600)

def visualize_feature_maps(feature_maps, ground_truths, optim, data):
    num_maps = len(feature_maps)
    fig, axs = plt.subplots(1, num_maps + 1, figsize=(15, 6))
    
    # Randomly select an image index from the batch
    batch_size = feature_maps[list(feature_maps.keys())[0]].shape[0]
    random_idx = torch.randint(0, batch_size, (1,)).item()
    
    for i, (name, feature_map) in enumerate(feature_maps.items()):
        # Feature map visualization (use first channel and normalize)
        feature_map_np = feature_map[random_idx, 0].detach().cpu().numpy()
        feature_map_np = (feature_map_np - feature_map_np.min()) / (feature_map_np.max() - feature_map_np.min())
        
        # Feature map subplot  
        axs[i].imshow(feature_map_np, cmap='viridis')
        if i == 4:
            axs[i].set_title('Output')
        else:
            axs[i].set_title(f'Decoder Stage: {i+1}')
        axs[i].axis('off')
    
    # Using the same random_idx for consistency
    ground_truth_np = ground_truths[random_idx].detach().cpu().numpy()
    ground_truth_np = ground_truth_np.squeeze(0)  # Squeeze the first dimension (channel)
    
    axs[-1].imshow(ground_truth_np, cmap='viridis')
    axs[-1].set_title('Ground Truth')
    axs[-1].axis('off')
    
    plt.tight_layout()
    
    # Add the random index to the save path to track different visualizations
    # plt.savefig(f'{data}_decoder_maps_{optim}.png', dpi=600)
    plt.show()

if __name__ == '__main__':
  show_test_predictions(is_test_loader, model, device, optim='RMSProp', data='ISIC')
  test_batch_iter = iter(kv_test_loader)
  test_images, test_masks = next(test_batch_iter)
  
  with torch.no_grad():
          test_images = test_images.to(device)
          outputs, feature_maps = tunet(test_images)
          visualize_feature_maps(feature_maps, test_masks, 'Adam', 'Kvasir')
      
          # kv_test_images = kv_test_images.to(device)
          # kv_outputs, kv_feature_maps = model(kv_test_images)
          # visualize_feature_maps(kv_feature_maps, kv_test_masks, 'Adam', 'Kvasir')
  
