from utils.metrics import plot_conv_curves, plot_dice_score, save_metrics, calculate_roc_curve, plot_roc_curve
from .train import kv_metrics, kv_train_loader, kv_val_loader, kv_test_loader

training_metrics = kv_metrics['train']
validation_metrics = kv_metrics['validation']

# Plot metrics
plot_conv_curves(training_metrics, validation_metrics, 'ADAM', 'Kvasir')

# Plot Dice score
plot_dice_score(training_metrics, validation_metrics, 'ADAM', 'Kvasir')

save_metrics(training_metrics, validation_metrics, 'ADAM', 'Kvasir')
fpr, tpr, roc_auc = calculate_roc_curve(model, kv_val_loader, device)  
plot_roc_curve(fpr, tpr, roc_auc, 'ADAM', 'Kvasir')

'''all the above processes should be repated for each dataset and RMSProp optimizer as well'''
