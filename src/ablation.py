import pandas as pd
from models.mlat import MLAT

def save_results(exp_name, metrics, save_path="ablation_table.csv"):

     # ---- Best validation Dice ----
    best_dice = max(metrics['validation']['dice'])
    best_idx = metrics['validation']['dice'].index(best_dice)
    
    # ---- Extract final validation metrics ----
    best_dice = round(best_dice, 3)
    best_precision = round(metrics['validation']['precision'][best_idx], 3)
    best_recall = round(metrics['validation']['recall'][best_idx], 3)
    best_fsc = round(metrics['validation']['fsc'][best_idx], 3)
    total_time = round(metrics['total_time_min'], 1)

    # ---- Create one-row DataFrame ----
    row = pd.DataFrame([{
        'Experiment': exp_name,
        'Dice': best_dice,
        'Precision': best_precision,
        'Recall': best_recall,
        'Fsc': best_fsc,
        'Total_Time_min': total_time
    }])

    # ---- Append or create table ----
    if os.path.exists(save_path):
        row.to_csv(save_path, mode='a', header=False, index=False)
    else:
        row.to_csv(save_path, index=False)

    print(f"Saved results to {save_path}")

def run_experiment(exp_name, **model_flags):
    print(f"\n===== Running: {exp_name} =====")

    model = MLAT(**model_flags)
    
    # Wrap the model for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
        
    model.to(device)

    if torch.__version__ >= "2.0":
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=False
        )

    torch.cuda.empty_cache()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)

    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    criterion = nn.BCEWithLogitsLoss()
    
    # Ensure criterion and optimizer are also on the correct device
    criterion = criterion.to(device)

    metrics = train_model(model=model, optimizer=optimizer, scheduler=scheduler, 
                          criterion=criterion, epochs=60, step_size=20 )
    
    save_results(exp_name, metrics)
    
    return metrics

experiments = {
    "B0_Baseline": dict(use_asib=False, use_taac=False),

    "B1_ASIB_only": dict(use_asib=True,  use_taac=False),

    "B2_TAAC_only": dict(use_asib=False, use_taac=True),

    "B3_Full_MLAT": dict(use_asib=True,  use_taac=True),
}

results = {}

for name, flags in experiments.items():
    results[name] = run_experiment(name, **flags)
