from sklearn.decomposition import TruncatedSVD
import pickle

def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def load_pc(path_to_pc):
    with open(path_to_pc,'rb') as f_in:
        pc = pickle.load(f_in)
    return pc

def save_pc(pc, path_to_pc):
    with open(path_to_pc,'wb') as f_out:
        pickle.dump(pc, f_out)

def remove_pc(X, npc, pc_mode, path_to_pc):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    if pc_mode == 'load':
        pc = load_pc(path_to_pc)
    elif pc_mode == 'self' or pc_mode == 'save':
        pc = compute_pc(X, npc)
        if pc_mode == 'save':
            save_pc(pc, path_to_pc)
    
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX
