from models.baselines import UNet, AttentionUNet, TransUNet
from models.mlat import MLAT
import time
from ptflops import get_model_complexity_info

def load_model(model_class, model_path, device, **model_kwargs):
    # Recreate the model architecture
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
''' path1, path2, path3 and path4 will be the path of the saved models'''

#  Load models
# unet = load_model(model_class=UNet, model_path=path1, device=device, in_channels=3, num_classes=1)
# aunet = load_model(model_class=AttentionUNet, model_path=path2, device=device, 
#                    in_channels=3, num_classes=1)
tunet = load_model(model_class=TransUNet, model_path=path3, device=device, 
                   in_channels=3,num_classes=1)
mlat = load_model(model_class=MLAT, model_path=path4, device=device, 
                   n_channels=3,n_classes=1)

EXPERIMENTS = {
    "UNet": {
        "model_cls": UNet,
        "model_kwargs": {},
        "weight_path": path1
    },
    "Attention_UNet": {
        "model_cls": AttentionUNet,
        "model_kwargs": {},
        "weight_path": path2
    },
    "TransUNet": {
        "model_cls": TransUNet,
        "model_kwargs": {},
        "weight_path": path3
    },
    "MLAT": {
        "model_cls": MLAT,
        "model_kwargs": { "use_asib": True,"use_taac": True,
        },
        "weight_path": path4
    }
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def compute_flops(model, input_res=(3, 256, 256)):
    macs, _ = get_model_complexity_info(
        model,
        input_res,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    return (macs * 2) / 1e9  # GFLOPs


def measure_inference_time(model, device, runs=50):
    model.eval()
    dummy = torch.randn(1, 3, 256, 256).to(device)

    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.time() - start) / runs * 1000  # ms

trained_models = {}
all_results = []

for exp_name, cfg in EXPERIMENTS.items():

    print(f"\n===== Evaluating {exp_name} =====")

    # Load model
    model = cfg["model_cls"](**cfg["model_kwargs"]).to(device)

    checkpoint = torch.load(cfg["weight_path"], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    trained_models[exp_name] = model

    # Complexity params
    params = count_parameters(model)
    flops = compute_flops(model)
    inf_time = measure_inference_time(model, device)

    all_results.append({
        "Experiment": exp_name,
        "Params(M)": round(params, 2),
        "GFLOPs": round(flops, 2),
        "Inference(ms)": round(inf_time, 2)
    })

df = pd.DataFrame(all_results)
print(df)

def collect_per_image_dice(model, loader, device, threshold=0.5):
    model.eval()
    dices = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)

            # -------- FIX HERE --------
            if isinstance(outputs, tuple):
                preds = outputs[0]
            else:
                preds = outputs
            # --------------------------

            preds = torch.sigmoid(preds)
            preds = (preds > threshold).float()

            intersection = (preds * masks).sum(dim=(1,2,3))
            union = preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))

            dice = (2 * intersection + 1e-7) / (union + 1e-7)

            dices.extend(dice.cpu().numpy())

    return np.array(dices)

dice_trans = collect_per_image_dice(tunet, is_val_loader, device)
dice_mlat = collect_per_image_dice(mlat, is_val_loader, device)
from scipy.stats import wilcoxon

stat, p_value = wilcoxon(dice_trans, dice_mlat)

print(f"Wilcoxon statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
