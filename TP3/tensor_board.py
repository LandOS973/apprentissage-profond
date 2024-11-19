import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Initialisation de TensorBoard
writer = SummaryWriter("logs")

# Modèle simple
model = nn.Linear(5, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Exemple de données
inputs = torch.rand(100, 5)
targets = torch.rand(100, 1)

# Entraînement
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Enregistrer les métriques dans TensorBoard
    writer.add_scalar("Loss/train", loss.item(), epoch)

    # Enregistrer les poids et les gradients
    for name, param in model.named_parameters():
        writer.add_histogram(f"Weights/{name}", param, epoch)
        writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

    # Ajouter le graphique du modèle au premier epoch
    if epoch == 0:
        writer.add_graph(model, inputs)

# Fermer le SummaryWriter
writer.close()

# Lancer TensorBoard
# tensorboard --logdir TP3/logs