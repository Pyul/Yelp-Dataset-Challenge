import matplotlib.pyplot as plt
import numpy as np

def heatplot(data,labelx,labely):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # # colorbar
    # clbar=ax.colorbar()
    # clbar.set_label('Similarity')

    ax.set_xticklabels(labelx, minor=False)
    ax.set_yticklabels(labely, minor=False)
    # ax.invert_yaxis()
    # ax.xaxis.tick_top()
    plt.xticks(rotation=90)
    plt.show()


# def heatplot(X,labels):
#     fig1=plt.figure(figsize=(4,8))
#     plt.imshow(X[:,:])
#     plt.hot()
#     clbar=plt.colorbar()
#     clbar.set_label('Normalized Feature Value')
#     plt.grid()
#     plt.xlabel('Restaurant #')
#     plt.ylabel('Restaurant #')
#     plt.title('Figure 1: Restaurant similarity')
#     set_xticklabels(labels, minor=False)
#     set_yticklabels(labels, minor=False)
#
#     plt.show()
#
# data = np.random.rand(4,4)
# labels = list('ABCD')
# heatplot(data,labels)
