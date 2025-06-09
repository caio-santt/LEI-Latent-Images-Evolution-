A lightweight proof-of-concept for Latent Evolutionary Images (LEI): a genetic algorithm that evolves 32 × 32 VAE latent tensors—without labels or gradients—so that a frozen ViT-CIFAR-10 classifier assigns high confidence to a chosen class.
The single script (lei_genetic_algorithm.py) initializes a random latent population, applies mutation + crossover, evaluates fitness via the classifier, and saves intermediate reconstructions, final latents, and an evolution GIF.

For a complete technical description and experimental results, please refer to the accompanying paper (link to be added upon publication).
