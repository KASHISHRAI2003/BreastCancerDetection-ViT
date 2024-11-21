# helper_functions.py
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

def plot_loss_curves(results):
    """Plots training and test loss and accuracy curves."""
    train_loss = results['train_loss']
    test_loss = results['test_loss']
    train_acc = results['train_acc']
    test_acc = results['test_acc']

    epochs = range(len(train_loss))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, test_acc, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()

def load_image(image_path, transform=None, image_size=(224, 224)):
    """Loads an image and applies optional transformations."""
    img = Image.open(image_path).convert("RGB")
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
    return transform(img).unsqueeze(0)

def pred_and_plot_image(model, image_path, class_names, transform=None, device=torch.device("cpu")):
    """Makes a prediction on an image and plots the result."""
    # Load image
    model.eval()
    img = load_image(image_path, transform=transform).to(device)
    
    # Make prediction
    with torch.no_grad():
        pred_logit = model(img)
    
    # Convert logit to probabilities
    pred_prob = torch.softmax(pred_logit, dim=1)
    pred_label = torch.argmax(pred_prob, dim=1)

    # Display image
    plt.imshow(Image.open(image_path))
    plt.title(f"Prediction: {class_names[pred_label]} | Confidence: {pred_prob[0][pred_label]:.3f}")
    plt.axis(False)
    plt.show()
