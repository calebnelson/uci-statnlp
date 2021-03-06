#!/bin/python

import numpy as np
import scipy as sp


def read_files(tarfname):
    """Read the training and development data from the speech tar file.
    The returned object contains various fields that store the data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")

    class Data:
        pass
    speech = Data()
    print("-- train data")
    speech.train_data, speech.train_fnames, speech.train_labels = read_tsv(
        tar, "train.tsv")
    print(len(speech.train_data))
    print("-- dev data")
    speech.dev_data, speech.dev_fnames, speech.dev_labels = read_tsv(
        tar, "dev.tsv")
    print(len(speech.dev_data))

    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    speech.count_vect = CountVectorizer()
    speech.tfidf_transformer = TfidfTransformer()
    speech.trainX = speech.count_vect.fit_transform(speech.train_data)
    speech.trainX_tfidf = speech.tfidf_transformer.fit_transform(speech.trainX)
    speech.devX = speech.count_vect.transform(speech.dev_data)
    speech.devX_tfidf = speech.tfidf_transformer.transform(speech.devX)
    print(speech.trainX_tfidf.shape)
    print(speech.devX_tfidf.shape)
    from sklearn import preprocessing
    speech.le = preprocessing.LabelEncoder()
    speech.le.fit(speech.train_labels)
    speech.target_labels = speech.le.classes_
    speech.trainy = speech.le.transform(speech.train_labels)
    speech.devy = speech.le.transform(speech.dev_labels)
    print(speech.trainy.shape)
    print(speech.devy.shape)
    tar.close()
    return speech


def read_unlabeled(tarfname, speech):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the speech.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")

    class Data:
        pass
    unlabeled = Data()
    unlabeled.data = []
    unlabeled.fnames = []
    for m in tar.getmembers():
        if "unlabeled" in m.name and ".txt" in m.name:
            unlabeled.fnames.append(m.name)
            unlabeled.data.append(read_instance(tar, m.name))
    unlabeled.X = speech.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled


def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    fnames = []
    for line in tf:
        line = line.decode("utf-8")
        (ifname, label) = line.strip().split("\t")
        # print ifname, ":", label
        content = read_instance(tar, ifname)
        labels.append(label)
        fnames.append(ifname)
        data.append(content)
    return data, fnames, labels


def write_pred_kaggle_file(unlabeled, cls, outfname, speech):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the speech object,
    this function write the predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The speech object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = speech.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("FileIndex,Category\n")
    for i in range(len(unlabeled.fnames)):
        fname = unlabeled.fnames[i]
        # iid = file_to_id(fname)
        f.write(str(i+1))
        f.write(",")
        # f.write(fname)
        # f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def file_to_id(fname):
    return str(int(fname.replace("unlabeled/", "").replace("labeled/", "").replace(".txt", "")))


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("FileIndex,Category\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (ifname, label) = line.strip().split("\t")
            # iid = file_to_id(ifname)
            i += 1
            f.write(str(i))
            f.write(",")
            # f.write(ifname)
            # f.write(",")
            f.write(label)
            f.write("\n")
    f.close()


def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts OBAMA_PRIMARY2008 for all the instances.
    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("FileIndex,Category\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (ifname, label) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("OBAMA_PRIMARY2008")
            f.write("\n")
    f.close()


def read_instance(tar, ifname):
    inst = tar.getmember(ifname)
    ifile = tar.extractfile(inst)
    content = ifile.read().strip()
    return content


def merge_sparse_matrices(m1, m2):
    merged_data = np.concatenate((m1.data, m2.data))
    merged_indicies = np.concatenate((m1.indices, m2.indices))
    merged_index_pointers = np.concatenate(
        (m1.indptr, (m2.indptr + len(m1.data))[1:]))
    return sp.sparse.csr_matrix((merged_data, merged_indicies, merged_index_pointers))


if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/speech.tar.gz"
    speech = read_files(tarfname)

    print("Training classifier")
    import classify
    C_range = [1, 10, 100, 1000]
    solvers = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    for solver in ["saga"]:
        print("Using " + solver)
        for c in [10]:
            print("Evaluating at C=" + str(c))
            for tfidf in [True]:
                print("With tfidf" if tfidf else "Without tfidf")
                cls = classify.train_classifier(
                    speech.trainX_tfidf if tfidf else speech.trainX, speech.trainy, c=c, solver=solver)
                print("Acc on Training Data")
                classify.evaluate(
                    speech.trainX_tfidf if tfidf else speech.trainX, speech.trainy, cls)
                print("Acc on Dev Data")
                classify.evaluate(
                    speech.devX_tfidf if tfidf else speech.devX, speech.devy, cls)
                print("\n")

    print("Reading unlabeled data")
    unlabeled = read_unlabeled(tarfname, speech)
    # numBatches = 10
    # labeledXBatches = np.split(speech.trainX_tfidf.toarray(), numBatches)
    # labeledYBatches = np.split(speech.trainy, numBatches)
    # unlabeledXBatches = np.split(
    #     unlabeled.X.toarray()[:-2], numBatches)
    # trainXBatches = [None] * numBatches
    # trainYBatches = [None] * numBatches
    # print(labeledXBatches[0].shape)
    # print(unlabeledXBatches[0].shape)
    # for i in range(numBatches):
    #     trainXBatches[i] = np.concatenate((
    #         labeledXBatches[i], unlabeledXBatches[i]))
    #     trainYBatches[i] = np.concatenate((labeledYBatches[i], np.full(
    #         unlabeledXBatches[i].shape[0], -1.)), axis=None)
    # from sklearn.semi_supervised import label_propagation
    # label_spread = label_propagation.LabelPropagation(
    #     kernel='knn', alpha=1)
    # print("batches prepared, propagating labels")
    # for i in range(numBatches):
    #     label_spread.fit(trainXBatches[i], trainYBatches[i])
    # print("labels propagated, training model")

    # newTrainX = trainXBatches[0]
    # newTrainy = trainYBatches[0]
    # for i in range(1, numBatches):
    #     newTrainX = np.concatenate((newTrainX, trainXBatches[i]))
    #     newTrainy = np.concatenate((newTrainy, trainYBatches[i]), axis=None)
    # print(newTrainX.shape)
    # print(newTrainy.shape)

    # from scipy import sparse
    # trainXsparse = sparse.csr_matrix(newTrainX)
    # cls = classify.train_classifier(
    #     trainXsparse, newTrainy, c=10, solver="saga")
    # print("model trained, evaluating")
    # classify.evaluate(trainXsparse, newTrainy, cls)
    # classify.evaluate(speech.devX_tfidf, speech.devy, cls)

    print("Writing pred file")
    write_pred_kaggle_file(unlabeled, cls, "data/speech-pred.csv", speech)

    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/speech-unlabeled.tsv", "data/speech-gold.csv")
    # write_basic_kaggle_file("data/speech-unlabeled.tsv", "data/speech-basic.csv")
