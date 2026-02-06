import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_conv_curves(training_metrics, validation_metrics, optim, data):
    # Set style for better visualization
    plt.style.use('default')

    mpl.rcParams.update({
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'grid.color': 'black',
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
        'axes.edgecolor': 'black',
    })

    # Create figure and subplots
    fig, axs = plt.subplots(1, 5, figsize=(20, 6), sharex=True)
    # fig.suptitle('CVC Performance Metrics', fontsize=18, y=0.95)
    
    # Flatten axs for easier iteration
    axs = axs.flatten()
    
    # Metrics to plot (excluding dice since it's similar to accuracy)
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'fsc']
    titles = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 score']
    
    # Plot each metric
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        # Plot training metric
        axs[idx].plot(training_metrics[metric], label=f'Training', color='green', linewidth=2)
        
        # Plot validation metric
        axs[idx].plot(validation_metrics[metric], label=f'Validation', color='orange', linewidth=2)
        
        # Customize subplot
        axs[idx].set_title(f'{title}', pad=10)
        axs[idx].set_xlabel('Epoch')
        axs[idx].set_facecolor('white')
        axs[idx].grid(True, linestyle='--', alpha=0.7)
        axs[idx].legend(loc='best')
        
        # Add horizontal and vertical grid
        axs[idx].grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Set y-axis limits for metrics other than loss
        if metric != 'loss':
            axs[idx].set_ylim([0, 1.1])
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save plot if path is provided
    plt.savefig(f'{data}_conv_curve_{optim}.png', dpi=600, bbox_inches='tight')
        
    plt.show()

# Create a separate figure for Dice Score
def plot_dice_score(training_metrics, validation_metrics, optim, data):
    plt.figure(figsize=(10, 6))
    plt.plot(training_metrics['dice'], label='Training Dice', color='violet', linewidth=2)
    plt.plot(validation_metrics['dice'], label='Validation Dice', color='orange', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.ylim([0, 1.1])
    
    plt.savefig(f'{data}_dice_curve_{optim}.png', dpi=600, bbox_inches='tight')
    
    plt.show()

import pandas as pd
def save_metrics(training_metrics, validation_metrics, optim, dataset):
    # Create a DataFrame with all metrics
    data = {
        'epoch': list(range(1, len(training_metrics['loss']) + 1)),
        'train_loss': training_metrics['loss'],
        'train_accuracy': training_metrics['accuracy'],
        'train_precision': training_metrics['precision'],
        'train_recall': training_metrics['recall'],
        'train_fsc': training_metrics['fsc'],
        'train_dice': training_metrics['dice'],
        
        'val_loss': validation_metrics['loss'],
        'val_accuracy': validation_metrics['accuracy'],
        'val_precision': validation_metrics['precision'],
        'val_recall': validation_metrics['recall'],
        'val_fsc': validation_metrics['fsc'],
        'val_dice': validation_metrics['dice'],
    }
    
    df = pd.DataFrame(data)
    df.to_csv(f'{dataset}_training_history_{optim}.csv', index=False)

def calculate_roc_curve(model, data_loader, device, threshold=0.5):
    # model = model.to(device)
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs,_= model(images)
            probs = F.sigmoid(outputs).cpu().numpy()
            
            # Flatten and binarize
            all_probs.extend(probs.flatten())
            # Convert masks to binary values
            binary_masks = (masks.cpu().numpy() > threshold).astype(np.int32)
            all_labels.extend(binary_masks.flatten())
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Ensure labels are binary
    all_labels = (all_labels > threshold).astype(np.int32)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc, optim, data):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(f'{data}_roc_curve_{optim}.png', dpi=600, bbox_inches='tight')
    
    plt.show()
