import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)

    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.legend()

    display.display(plt.gcf())

    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    
    # Display current figure after all elements are drawn
    display.display(plt.gcf())
    plt.pause(0.1)