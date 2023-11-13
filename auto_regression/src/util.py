import matplotlib.pyplot as plt
def seq2tensor(s):
    return s.reshape(-1, 1, 28, 28)
def tensor2seq(s):
    return s.reshape(-1, 28 * 28)



def plot_inference(sequences, nrow, ncol):
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True, figsize=(ncol, nrow ))
    for i in range(nrow*ncol):
        ax = axs[i // ncol, i % ncol]
        ax.imshow(sequences[i, :].cpu().reshape(28,28), cmap=plt.cm.gray)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_example(img):
    img = img.reshape(28,28)
    fig = plt.figure(figsize=(3, 3))
    # And displaying the image
    plt.imshow(img, cmap="gray")

def plot_prior(img):
    _, nrow, _ = img.shape
    img = img.reshape(nrow,28)
    fig = plt.figure(figsize=(3, 3))
    # And displaying the image
    plt.imshow(img, cmap="gray")