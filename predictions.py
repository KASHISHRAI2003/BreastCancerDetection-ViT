# predictions.py
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def pred_and_plot_image(model, image_path, class_names, transform=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads an image, makes a prediction with a trained model, and plots the image with its predicted class.
    
    Args:
        model: A trained PyTorch model.
        image_path (str): File path to the target image.
        class_names (list): A list of class names the model can predict.
        transform (torchvision.transforms, optional): The transform pipeline to apply to the image. If None, uses default transforms.
        device (str, optional): Device to run the model on. Defaults to "cuda" if available.
    
    Returns:
        None
    """
    # Load and transform the image
    image = Image.open(image_path)
    
    # Apply default transform if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    image_transformed = transform(image).unsqueeze(0)  # Add batch dimension

    # Move model and image to the device
    model = model.to(device)
    image_transformed = image_transformed.to(device)

    # Set model to evaluation mode and make a prediction
    model.eval()
    with torch.no_grad():
        pred_logits = model(image_transformed)
    
    # Convert logits to probabilities
    pred_probs = torch.softmax(pred_logits, dim=1)
    
    # Get the predicted class (the class with the highest probability)
    pred_label = torch.argmax(pred_probs, dim=1).item()
    
    # Plot the image and the predicted label
    plt.imshow(image)
    plt.title(f"Prediction: {class_names[pred_label]} \nProbability: {pred_probs.max():.3f}")
    plt.axis(False)  # Turn off axes
    plt.show()
