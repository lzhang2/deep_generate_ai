def seq2tensor(s):
    return s.reshape(-1, 1, 28, 28)
def tensor2seq(s):
    return s.reshape(-1, 28 * 28)