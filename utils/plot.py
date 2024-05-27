import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def plot_latent_space(latent_2d,labels,title):
    df = pd.DataFrame(latent_2d,columns["x","y"])
    df["label"] = labels
    plt.figure(figsize=(10,8))
    sns.scatterplot(data=df,x = "x",y="y",hue="label",palette="tab10",legend="full",alpha=0.8)
    plt.title(title)
    plt.show()

def plot_loss(epochs,train_losses,train_recon_losses,train_kl_losses):
    plt.plot(range(1, epochs+1), [l.detach().cpu().numpy() for l in train_losses], label='Train Loss')  
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()
    # Plot the reconstruction and KL losses
    plt.plot(range(1, epochs+1), [l.detach().cpu().numpy() for l in train_recon_losses], label='Train Reconstruction Loss')
    plt.plot(range(1, epochs+1), [t.detach().cpu().numpy() for t in train_kl_losses], label='Train KL Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Reconstruction and KL Loss Curves')
    plt.legend()
    plt.show()