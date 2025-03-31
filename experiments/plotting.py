import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import math
import random


def triangular_correlation_heatmap(vec1, label1, vec2, label2):
    corr = np.inner(vec1, vec2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=label2, yticklabels=label1)


def correlation_heatmap(vec1, label1, vec2, label2):
    corr = np.inner(vec1, vec2)
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=label2, yticklabels=label1)


def plot_bias_dir_correlations(bias_space, feature_labels):
    triangular_correlation_heatmap(bias_space, feature_labels, bias_space, feature_labels)


def plot_hist_to_axes(X, y, ax, feature, xlabel=None, ylabel=None, labels=None):
    miny = int(np.min(y))
    maxy = int(np.max(y))
    if miny < 0 or maxy > 1:
        # multi-class -> adjust labels
        # we either get one label (pos class) or ['other', pos class]; keep pos class label for class 1; others indicated by indices
        # if labels is None; just name by class integer
        if labels is None:
            labels = [str(c) for c in range(miny, maxy + 1)]
        if len(labels) == 1:
            labels = [str(c) if c != 1 else labels[0] for c in range(miny, maxy + 1)]
        elif len(labels) == 2:
            labels = [str(c) if c != 1 else labels[1] for c in range(miny, maxy + 1)]

    else:
        if labels is None:
            labels = ['group0', 'group1']
        else:
            if len(labels) == 1:
                labels = labels[0].split('/')
            if len(labels) == 1 and maxy != miny:
                labels.insert(0, 'other')

    label_idx = 0
    for c in range(miny, maxy + 1):
        lbl = labels[label_idx]
        label_idx += 1

        xc = [X[i] for i in range(X.shape[0]) if y[i] == c]
        if len(xc) == 0:
            continue

        counts_c, bins_c = np.histogram(xc, bins=100)
        if len(xc) < len(y) / 10:
            fac = len(y) / len(xc) / 2
            counts_c *= int(fac)
            lbl = "%s (scaled by %.2f)" % (lbl, fac)
        ax.hist(bins_c[:-1], bins_c, weights=counts_c, label=lbl)

    ax.legend()
    if xlabel is not None:
        ax.set_xlabel('%s (%s)' % (feature, xlabel))
    else:
        ax.set_xlabel(feature)
    ax.set_ylabel('test data: %s' % ylabel)


def plot_feature_histogram(X, y, labels=None, features=None, xlabel=None, ylabel=None, savefile=None):

    if features is not None:
        assert len(features) == 1 or X.shape[1] == len(features), ("got %i features, X.shape[1]=%i" % (len(features), X.shape[1]))
    else:
        if len(X.shape) == 1:
            n_features = 1
        else:
            n_features = X.shape[1]
        features = ['feature'+str(i) for i in range(n_features)]
    
    if type(y) is list:
        y = np.asarray(y)

    if len(y.shape) == 1 or y.shape[1] == 1:
        if len(X.shape) == 1 or X.shape[1] == 1:
            fig, ax = plt.subplots()
            plot_hist_to_axes(X, y, ax, features[0], xlabel, ylabel, labels)
        else:
            fig, axes = plt.subplots(1,X.shape[1], figsize=(4*X.shape[1],6))       
            for feature_id in range(X.shape[1]):
                plot_hist_to_axes(X[:, feature_id], y, axes[feature_id], features[feature_id], xlabel, ylabel, labels)
    else:
        if y.shape[1] == 2 and len(labels) == 1:
            fig, axes = plt.subplots(1, X.shape[1], figsize=(4*X.shape[1], 4*len(labels)))
            for feature_id in range(X.shape[1]):
                plot_hist_to_axes(X[:, feature_id], y[:, 1], axes[feature_id], features[feature_id], xlabel, ylabel,
                                  [labels[0]])
        else:
            # multiple binary identity labels
            fig, axes = plt.subplots(y.shape[1], X.shape[1], figsize=(4*X.shape[1], 4*len(labels)))
            for i, group in enumerate(labels):
                for feature_id in range(X.shape[1]):
                    plot_hist_to_axes(X[:, feature_id], y[:, i], axes[i][feature_id], features[feature_id], xlabel,
                                      ylabel, [group])

    if savefile is not None:
        print("save plot at ", savefile)
        plt.savefig(savefile, bbox_inches='tight')
    else:
        plt.show()


def tsne_group_viz(X: np.ndarray, y: list, groups: list):
    assert len(X.shape) == 2
    if type(y) == list:
        y = np.asarray(y)
    
    if len(groups) > 2 or not len(y.shape) == 1:
        print("tsne only implemented for binary single label")
        return
        
    X_pca = PCA(n_components=50).fit_transform(X)
    X_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)

    fig, ax = plt.subplots()
    x0 = np.asarray([X[i,:] for i in range(len(X)) if y[i] == 0])
    x1 = np.asarray([X[i,:] for i in range(len(X)) if y[i] == 1])
    ax.scatter(x0[:,0], x0[:,1], label=groups[0])
    ax.scatter(x1[:,0], x1[:,1], label=groups[1])
    ax.legend()
    plt.show()


def plot_bias_dir_corr(b_pairwise, B, savefile=None, figsize=(10, 6)):
    n_bias_dir = len(b_pairwise)
    corr = np.eye(n_bias_dir+1, n_bias_dir+1)
    for i in range(n_bias_dir):
        for j in range(n_bias_dir-1):
            if j < i:
                corr[i, j] = cossim(b_pairwise[i], b_pairwise[j])
        if B.shape[0] > 1:
            corr[-1, i] = np.sum([cossim(b_pairwise[i], B[j,:].flatten()) for j in range(B.shape[0])])
        else:
            corr[-1, i] = cossim(b_pairwise[i], B.flatten())
        
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(np.abs(corr), ax=ax)
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()


def plot_bias_space_robustness(n_terms, mean_angles, std_angles, bias_type=None):    
    fig, ax = plt.subplots(1)
    ax.plot(n_terms, mean_angles, lw=2, label='bias space similarity', color='blue')
    ax.fill_between(n_terms, mean_angles+std_angles, mean_angles-std_angles, facecolor='blue', alpha=0.3)
    if bias_type is not None:
        ax.set_title('Robustness of bias space per number of defining terms (%s)' % bias_type)
    else:
        ax.set_title('Robustness of bias space per number of defining terms')
    #ax.legend(loc='lower left')
    ax.set_xlabel('# defining terms')
    ax.set_ylabel('similarity with bias space defined on all %i terms' % n_terms[-1])
    ax.grid()
    #plt.savefig('plots/attribute_robustness_'+bt+'.png', bbox_inches="tight")
    plt.show()


def compute_bias_space_robustness(emb, B, repeat = 1000):
    mean_angles = []
    std_angles = []
    ns = range(2,emb.shape[1])
    for n_terms in ns:
        angles = []
        for i in range(repeat):
            ids = random.sample(range(emb.shape[1]), n_terms)
            B_ = get_bias_space(emb[:,ids,:])

            if B.shape[0] > 1:
                angles.append(np.sum([[cossim(B[i,:].flatten(), B_[j,:].flatten()) for i in range(B.shape[0])] for j in range(B_.shape[0])]))
            else:
                angles.append(cossim(B.flatten(),B_.flatten()))

        mean_angles.append(np.mean(angles))
        std_angles.append(np.std(angles))

    return np.asarray(ns), np.asarray(mean_angles), np.asarray(std_angles)


def plot_in_bias_space(B, emb, groups, texts=None, label=None, classes=None, xlabel=None, ylabel=None, lims=None):
    assert (ylabel is not None and B.shape[0]==1 or xlabel is not None) or (groups is not None and len(groups) == B.shape[0]+1)
    fig, ax = plt.subplots(figsize=(10,10))
    
    if B.shape[0] > 1:
        proj0 = [cossim(e, B[0,:].flatten()) for e in emb]
        proj1 = [cossim(e, B[1,:].flatten()) for e in emb]
        if xlabel is None:
            xlabel = groups[0]+" <---------> "+groups[1]
        if ylabel is None:
            ylabel = groups[0]+" <---------> "+groups[2]
        ax.plot((0,0), (-0.8,0.8), color='black')
        
        # todo higher dim
    else:
        proj1 = [cossim(e, B[0,:].flatten()) for e in emb] # y-axis better for annotation
        if label is not None:
            proj0 = label
        else:
            proj0 = [0 for e in emb]
        if ylabel is None:
            ylabel = groups[0]+" <---------> "+groups[1]
        xlabel = ""

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_ylim([-1,1])

    if lims is None:
        lims = 0.8
    ax.plot((-lims,lims),(0,0), color='black')

    if label is not None and classes is not None:
        ax.scatter(proj0, proj1, c=label, label=classes)
        ax.legend()
    else:
        ax.scatter(proj0, proj1, c=label)

    if texts is not None:
        for i, text in enumerate(texts):
            ax.annotate(text, (proj0[i],proj1[i]))
