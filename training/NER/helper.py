import re
import numpy

def sanitise_text(string):
    # english regexp pre-processing for TF
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", string) # weird string removal
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"-", " - ", string)
    string = re.sub(r"'", " \" ", string)
    string = re.sub(r"\"", " \" ", string)
    return string.strip().lower()


def sanitise_vocabulary_string(string):
    # weird
    if string not in ["--", "``", "`"]:
        # , "'s", "n't", ".", ",", "(", ")", "[", "]", "'", "\"", ??
        return True
    else:
        return False


def sanitise_en_negation(string):
    # replace negation with reasonable words
    string.replace("can\'t", "can not")
    string.replace("couldn\'t", "could not")
    string.replace("won\'t", "will not")
    string.replace("wouldn\'t", "would not")
    string.replace("isn\'t", "is not")
    string.replace("wasn\'t", "was not")
    string.replace("shan\'t", "shall not")
    string.replace("shouldn\'t", "should not")
    # ...
    # https://en.wiktionary.org/wiki/Category:English_words_suffixed_with_-n%27t
    # isn't useful or is?
    return string

def plot_graph_from_hist(histories, filepath='plots', filename='graph'):
    import matplotlib.pyplot as plt
    import datetime, time
    # summarize keras history for accuracy
    for h in histories:
        plt.plot(h.history['acc'], color='red', linestyle='solid', label="Train accuracy")
        plt.plot(h.history['val_acc'], color='blue', linestyle='solid', label="Validation accuracy")
        plt.plot(h.history['loss'], color='red', linestyle='dashed', label="Test loss")
        plt.plot(h.history['val_loss'], color='blue', linestyle='dashed', label="Test loss")
    plt.title('model accuracy/loss')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'test_acc', 'train_loss', 'test_loss'], loc='upper left')
    # plt.show()
    plt.savefig(filepath+'/'+filename+str(time.mktime(datetime.datetime.today().timetuple()))+'.png')

def gimme_rand_wv():
    new_random_wv = numpy.array((numpy.random.rand(300) * 2) - 1, dtype=numpy.float32)
    norm_const = numpy.linalg.norm(new_random_wv)
    new_random_wv /= norm_const
    return new_random_wv