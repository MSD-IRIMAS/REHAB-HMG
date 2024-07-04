import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
def plot_latent_space(latent_2d,labels,title,output_directory):
    df = pd.DataFrame(latent_2d,columns=["x","y"])
    df["label"] = labels
    plt.figure(figsize=(10,8))
    sns.scatterplot(data=df,x = "x",y="y",hue="label",palette="tab10",legend="full",alpha=0.8)
    plt.title(title)
    plt.savefig(output_directory + title+'.pdf')
    # plt.show()

def plot_loss(epochs, train_losses, train_recon_losses, train_kl_losses, output_directory):
    plt.plot(range(1, epochs + 1), train_losses, label='Total Loss')  
    plt.plot(range(1, epochs + 1), train_recon_losses, label='Reconstruction Loss')
    plt.plot(range(1, epochs + 1), train_kl_losses, label='KL Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(output_directory, 'loss.pdf'))
    plt.close()
    # plt.show()

def plot_regressor_loss(epochs,train_loss,test_loss,output_directory):
    plt.plot(range(1, epochs+1), train_loss, label='Train Loss')
    plt.plot(range(1, epochs+1), test_loss, label='test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.savefig(os.path.join(output_directory, 'loss.pdf'))
    plt.legend()
    plt.close()
    # plt.show()
def plot_true_pred_scores(predicted_scores,true_scores, output_directory ,title ):
    plt.plot(predicted_scores, label='Predicted Scores')
    plt.plot(true_scores,  label='True Scores')
    plt.xlabel('Sample Index')
    plt.ylabel('Score')
    plt.title(title)
    plt.savefig(os.path.join(output_directory, title+'.pdf'))
    plt.legend()
    plt.close()
