import matplotlib.pyplot as plt
def draw_plot(precision, recall):
    plt.figure()
    plt.step(recall, precision, color='r', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='#F59B00')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()
    return
