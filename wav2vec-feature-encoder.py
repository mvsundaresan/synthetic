import torch
import torch.nn as nn
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load a .wav file and visualize the waveform
def load_audio(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=8000)  # Load waveform
    
    # Normalize waveform
    waveform = waveform / np.max(np.abs(waveform))
    
    print(f"Loaded Audio: {file_path}, Sample Rate: {sample_rate}, Shape: {waveform.shape}")
    
    # Plot the waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(waveform, sr=sample_rate)
    plt.title("Original Audio Waveform")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    
    return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0), sample_rate

# Define the Wave2Vec Feature Encoder (CNN)
class Wave2VecFeatureEncoder(nn.Module):
    def __init__(self):
        super(Wave2VecFeatureEncoder, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=0),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=0),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=0),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=0),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=0),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=0),
            nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=0),
        ])
        self.activation = nn.ReLU()

    def forward(self, x):
        feature_maps = []
        for layer in self.conv_layers:
            x = layer(x)
            x = self.activation(x)
            feature_maps.append(x)
        return x, feature_maps  # Return final output and intermediate feature maps

# Visualize feature maps at each CNN layer
def visualize_feature_maps(feature_maps):
    num_layers = len(feature_maps)
    
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 2 * num_layers))
    
    for i, feature_map in enumerate(feature_maps):
        feature_map = feature_map.detach().numpy()
        
        # Take the mean across all channels for visualization
        avg_feature_map = feature_map.mean(axis=1).squeeze()
        
        axes[i].plot(avg_feature_map, label=f"Layer {i+1}")
        axes[i].set_title(f"Feature Map at Layer {i+1}")
        axes[i].set_xlabel("Time Steps")
        axes[i].set_ylabel("Activation")
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

# Load Audio, Pass Through Model, and Visualize Outputs
def process_audio(file_path):
    # Load audio file
    waveform, sample_rate = load_audio(file_path)
    
    # Reshape for CNN input: (batch_size=1, channels=1, sequence_length)
    waveform = waveform.unsqueeze(0)  # Add batch dimension

    # Initialize the model
    model = Wave2VecFeatureEncoder()
    
    # Pass the waveform through the CNN
    output, feature_maps = model(waveform)
    
    # Print output shape
    print(f"Final Output Shape: {output.shape}")  # Expected (1, 512, T')
    
    # Visualize feature maps at each layer
    visualize_feature_maps(feature_maps)

# Run the model on a sample .wav file
if __name__ == "__main__":
    file_path = "../sample_audio/voices/sundar/F_1_Sundar.wav"  # Replace with your actual .wav file path
    process_audio(file_path)
