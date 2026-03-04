import matplotlib.pyplot as plt

# Problem 1: Varying first layer neuron count
layer_sizes = [16, 32, 64, 128, 256, 512]
test_accuracies_prob1 = [0.9395, 0.9574, 0.9664, 0.9707, 0.9783, 0.9778]

plt.figure(figsize=(10, 6))
plt.plot(layer_sizes, test_accuracies_prob1, marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Neurons in First Layer', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Problem 1: Editing Layer 1 Neurons Count', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

# Problem 2: Adding additional layers
num_layers = [2, 3, 4, 5]
test_accuracies_prob2 = [0.9778, 0.9785, 0.9767, 0.9797]

plt.figure(figsize=(10, 6))
plt.plot(num_layers, test_accuracies_prob2, marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Layers', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Problem 2: Adding Additional Layers', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks([2, 3, 4, 5])
plt.show()
```

**Add this to a new cell in your notebook, run it, and you'll get both figures.**

Then right-click on each plot to save as images for your submission.

**Also add a text/markdown cell documenting your layer architectures for Problem 2:**
```
Problem 2 Layer Architectures:
- 2 layers: 512 → 10
- 3 layers: 512 → 256 → 10
- 4 layers: 512 → 256 → 128 → 10
- 5 layers: 512 → 256 → 128 → 64 → 10