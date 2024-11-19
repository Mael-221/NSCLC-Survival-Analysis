import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from datasets import NSCLCDataset, CTPreprocess, SafeMinorityTransform
from model import initialize_model
from sklearn.metrics import f1_score, recall_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc




# Chemins relatifs aux données
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')

# Chemins spécifiques aux fichiers et dossiers de données
CSV_PATH = os.path.join(DATA_DIR, 'NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv')
IMAGES_PATH = os.path.join(DATA_DIR, 'manifest-1603198545583/NSCLC-Radiomics')
MODEL_PATH = os.path.join(BASE_DIR, '../results/model.pth')


# Configuration du device
if torch.cuda.is_available():
    device = torch.device('cuda')  # Utilise CUDA si disponible
elif torch.backends.mps.is_available():
    device = torch.device('mps')  # Utilise MPS pour Mac si disponible
else:
    device = torch.device('cpu')  # Sinon, utilise le CPU

print(f"Using device: {device}")

def load_model(model, model_path, device):
    """
    Charge un modèle sauvegardé, en supprimant les préfixes 'module.' si nécessaire.

    Args:
        model (nn.Module): L'architecture du modèle.
        model_path (str): Chemin du fichier modèle sauvegardé.
        device (torch.device): Le périphérique (CPU/GPU) à utiliser.

    Returns:
        nn.Module: Le modèle chargé avec les poids.
    """
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint

    # Supprimer les préfixes 'module.' si le modèle a été sauvegardé avec DataParallel
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model

def split_dataset(ct_dataset):
    """
    Splits the dataset into training, validation, and test sets,
    following the exact code from the notebook.
    """
    # Define the minimum desired percentage of the minority class in test and validation sets
    min_percentage_minority_in_test = 0.1  # 10%
    min_percentage_minority_in_val = 0.1  # 10%

    # Get the cleaned clinical data from the dataset
    clinical_data = ct_dataset.get_cleaned_data()

    # Retrieve unique patient IDs and corresponding labels
    unique_patient_ids = clinical_data['PatientID'].unique()
    labels = clinical_data.set_index('PatientID').loc[unique_patient_ids]['deadstatus.event']

    # Separate IDs based on class labels
    minority_class_ids = unique_patient_ids[labels == 0]  # Minority class (label 0)
    majority_class_ids = unique_patient_ids[labels == 1]  # Majority class (label 1)

    # Define helper functions
    def calculate_split_sizes(total_size, test_ratio, val_ratio):
        """
        Calculate the sizes of test and validation sets.
        """
        test_size = int(test_ratio * total_size)
        remaining_size = total_size - test_size
        val_size = int(val_ratio * remaining_size)
        train_size = remaining_size - val_size
        return train_size, val_size, test_size

    def split_class_ids(class_ids, test_size, val_size, random_state=42):
        """
        Split class IDs into train, validation, and test sets.
        """
        train_ids, test_ids = train_test_split(class_ids, test_size=test_size, random_state=random_state)
        train_ids, val_ids = train_test_split(train_ids, test_size=val_size, random_state=random_state)
        return train_ids, val_ids, test_ids

    # Calculate total sizes
    total_patients = len(unique_patient_ids)
    train_size, val_size, test_size = calculate_split_sizes(
        total_size=total_patients,
        test_ratio=0.2,  # 20% for test
        val_ratio=0.25  # 25% of remaining for validation
    )

    # Calculate the number of minority samples needed in test and validation sets
    minority_test_size = int(max(min_percentage_minority_in_test * test_size, 1))
    minority_val_size = int(max(min_percentage_minority_in_val * val_size, 1))

    # Split minority class IDs
    minority_train_ids, minority_val_ids, minority_test_ids = split_class_ids(
        minority_class_ids,
        test_size=minority_test_size,
        val_size=minority_val_size,
        random_state=42
    )

    # Remaining sizes for majority class
    majority_test_size = test_size - minority_test_size
    majority_val_size = val_size - minority_val_size

    # Split majority class IDs
    majority_train_ids, majority_val_ids, majority_test_ids = split_class_ids(
        majority_class_ids,
        test_size=majority_test_size,
        val_size=majority_val_size,
        random_state=42
    )

    # Combine IDs to get final splits
    train_patient_ids = np.concatenate([minority_train_ids, majority_train_ids])
    val_patient_ids = np.concatenate([minority_val_ids, majority_val_ids])
    test_patient_ids = np.concatenate([minority_test_ids, majority_test_ids])

    # Map Patient IDs to dataset indices
    patient_id_to_index = {patient_id: idx for idx, patient_id in enumerate(clinical_data['PatientID'])}

    # Get indices for each split
    train_idx = [patient_id_to_index[pid] for pid in train_patient_ids]
    val_idx = [patient_id_to_index[pid] for pid in val_patient_ids]
    test_idx = [patient_id_to_index[pid] for pid in test_patient_ids]

    # Create subsets of the dataset
    train_dataset = Subset(ct_dataset, train_idx)
    val_dataset = Subset(ct_dataset, val_idx)
    test_dataset = Subset(ct_dataset, test_idx)

    return train_dataset, val_dataset, test_dataset

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("Meilleur modèle sauvegardé.")


def evaluate_model(model, test_loader, output_dir="../results"):
    """
    Evaluate the model on the test dataset and save results to a directory.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        output_dir (str): Directory to save evaluation results.

    Returns:
        None
    """
    # Créez le dossier pour enregistrer les résultats
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if images is None:
                continue  # Skip empty batches
            images, labels = images.to(device), labels.to(device)

            # Adjust dimensions for evaluation
            images = images.unsqueeze(1)  # Add channel dimension

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for positive class

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculer les métriques
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.2f}%")

    # F1 Score, Sensitivity, Specificity
    if all_labels and all_predictions:
        report = classification_report(all_labels, all_predictions, target_names=["Class 0", "Class 1"])
        print("\nClassification Report:\n", report)

        cm = confusion_matrix(all_labels, all_predictions)
        print("\nConfusion Matrix:\n", cm)

        # Sauvegarde de la matrice de confusion
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix saved to {cm_path}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.2f}")

        # Sauvegarde de la courbe ROC
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig(roc_path)
        plt.close()
        print(f"ROC curve saved to {roc_path}")

    print("\nEvaluation completed. Results saved to:", output_dir)


def main(args):
    # Initialisation des transformations
    ct_preprocess = CTPreprocess(window_center=0, window_width=2000, target_size=(128, 128))
    minority_transform = SafeMinorityTransform(rotation_range=(-10, 10), vertical_flip_prob=0.5)
    dataset = NSCLCDataset(CSV_PATH, IMAGES_PATH, preprocess=ct_preprocess, minority_transform=minority_transform)

    # Split
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Modèle
    model = initialize_model(num_classes=2)
    model.to(device)

    if args.mode == "train":
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    elif args.mode == "evaluate":
        model = load_model(model, MODEL_PATH, device)
        evaluate_model(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lung Cancer Classification")
    parser.add_argument("--mode", required=True, choices=["train", "evaluate"], help="Mode: train or evaluate")
    args = parser.parse_args()
    main(args)