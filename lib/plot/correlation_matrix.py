import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_correlation_matrix(cm,
                          target_names,
                          title='Correlation matrix',
                          cmap=None,
                          save_to=None,
                          annotate=False):

    if cmap is None:
        cmap = plt.get_cmap('coolwarm')

    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90, ha='center')#, rotation=45
        plt.yticks(tick_marks, target_names, va='center')
    if annotate:
        ax = plt.gca()
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                text = ax.text(j, i, cm.round(2).values[i][j], ha="center", va="center", color="black")

    plt.tight_layout()

    if save_to:
        plt.savefig(save_to)


    #plt.ylabel('True label')
    #plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()