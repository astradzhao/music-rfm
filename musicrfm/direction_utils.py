import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

from xrfm import xRFM, RFM
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score

from copy import deepcopy
from tqdm import tqdm

from .utils import preds_to_proba

import torch
import torch.nn as nn
import torch.nn.functional as F

def batch_transpose_multiply(A, B, mb_size=5000):
    n = len(A)
    assert(len(A) == len(B))
    batches = torch.split(torch.arange(n), mb_size)
    sum = 0.
    for b in batches:
        Ab = A[b].cuda()
        Bb = B[b].cuda()
        sum += Ab.T @ Bb

        del Ab, Bb
    return sum

def accuracy_fn(preds, truth, multiclass_labeled=False):
    assert(len(preds)==len(truth))
    true_shape = truth.shape
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(truth, torch.Tensor):
        truth = truth.cpu().numpy()
        
    if multiclass_labeled:
        acc = np.sum(preds==truth)/len(preds) * 100
        return acc
        
    preds = preds.reshape(true_shape)
    
    if preds.shape[1] == 1:
        preds = np.where(preds >= 0.5, 1, 0)
        truth = np.where(truth >= 0.5, 1, 0)
    else:
        preds = np.argmax(preds, axis=1)
        truth = np.argmax(truth, axis=1)
        
    acc = np.sum(preds==truth)/len(preds) * 100
    return acc

def pearson_corr(x, y):     
    assert(x.shape == y.shape)
    
    x = x.float() + 0.0
    y = y.float() + 0.0

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))

    return numerator / denominator

def split_data(data, labels):
    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, test_size=0.2, random_state=0, shuffle=True
    ) 
    return data_train, data_test, labels_train, labels_test

def precision_score(preds, labels):
    true_positives = np.sum((preds == 1) & (labels == 1))
    predicted_positives = np.sum(preds == 1)
    return true_positives / (predicted_positives + 1e-8)  # add small epsilon to prevent division by zero

def recall_score(preds, labels):
    true_positives = np.sum((preds == 1) & (labels == 1))
    actual_positives = np.sum(labels == 1)
    return true_positives / (actual_positives + 1e-8)  # add small epsilon to prevent division by zero

def f1_score(preds, labels):
    precision = precision_score(preds, labels)
    recall = recall_score(preds, labels)
    return 2 * (precision * recall) / (precision + recall + 1e-8)  # add small epsilon to prevent division by zero


def compute_prediction_metrics(preds, labels, classification_threshold=0.5, regression=False):
    #print("Regression", regression)
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
    num_classes = labels.shape[1]
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    if regression:
        mse = np.mean((preds-labels)**2)
        r2 = r2_score(labels, preds)
        metrics = {'mse': mse, 'r2': r2}
        return metrics

    if not regression:
        # For multiclass, need to specify multi_class parameter
        if num_classes > 1:
            print("Labels shape", labels.shape, "Preds shape", preds.shape)
            print("Labels", labels, "Preds", preds)
            auc = roc_auc_score(labels, preds, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(labels, preds)
   
    mse = np.mean((preds-labels)**2)
    if num_classes == 1:  # Binary classification
        preds = np.where(preds >= classification_threshold, 1, 0)
        labels = np.where(labels >= classification_threshold, 1, 0)
        acc = accuracy_fn(preds, labels)
        precision = precision_score(preds, labels)
        recall = recall_score(preds, labels)
        f1 = f1_score(preds, labels)
    else:  # Multiclass classification
        preds_classes = np.argmax(preds, axis=1)
        label_classes = np.argmax(labels, axis=1)
        
        # Compute accuracy
        acc = np.sum(preds_classes == label_classes)/ len(preds) * 100
        
        # Initialize metrics for averaging
        precision, recall, f1 = 0.0, 0.0, 0.0
        
        # Compute metrics for each class
        for class_idx in range(num_classes):
            class_preds = (preds_classes == class_idx).astype(np.float32)
            class_labels = (label_classes == class_idx).astype(np.float32)
            
            precision += precision_score(class_preds, class_labels)
            recall += recall_score(class_preds, class_labels)
            f1 += f1_score(class_preds, class_labels)
        
        # Average metrics across classes
        precision /= num_classes
        recall /= num_classes
        f1 /= num_classes

    if not regression:
        metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mse': mse, 'accuracy': acc} # uncomment for classification
    else:
        metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'mse': mse, 'accuracy': acc} #uncomment for regression
    return metrics

def project_hidden_states(hidden_states, directions, n_components):
    """
    directions:
        {-1 : [beta_{1}, .., beta_{m}],
        ...,
        -32 : [beta_{1}, ..., beta_{m}]
        }
    hidden_states:
        {-1 : [h_{1}, .., h_{d}],
        ...,
        -32 : [h_{1}, ..., h_{d}]
        }
    """
    print("n_components", n_components)
    assert(hidden_states.keys()==directions.keys())
    layers = hidden_states.keys()
    
    projections = {}
    for layer in layers:
        vecs = directions[layer][:n_components].T
        projections[layer] = hidden_states[layer].cuda()@vecs.cuda()
    return projections

def aggregate_projections_on_coefs(projections, detector_coef):
    """
    detector_coefs:
        {-1 : [beta_{1}, bias_{1}],
        ...,
        -32 : [beta_32_{32}, bias_{32},
        'agg_sol': [beta_{agg}, bias_{agg}]]
    projections:
        {-1 : tensor (n, n_components),
        ...,
        -32 : tensor (n, n_components),
        }
    """
        
    layers = projections.keys()
    agg_projections = []
    for layer in layers:
        X = projections[layer].cuda()
        agg_projections.append(X.squeeze(0))
    
    agg_projections = torch.concat(agg_projections, dim=1).squeeze()
    agg_beta = detector_coef[0]
    agg_bias = detector_coef[1]
    agg_preds = agg_projections@agg_beta + agg_bias
    return agg_preds

def project_onto_direction(tensors, direction, device='cuda'):
    """
    tensors : (n, d)
    direction : (d, )
    output : (n, )
    """
    assert(len(tensors.shape)==2)
    assert(tensors.shape[1] == direction.shape[0])
    
    return tensors.to(device=device) @ direction.to(device=device, dtype=tensors.dtype)

def fit_pca_model(train_X, train_y, n_components=1, mean_center=True):
    """
    Assumes the data are in ordered pairs of pos/neg versions of the same prompts:
    
    e.g. the first four elements of train_X correspond to 
    
    Dishonestly say something about {object x}
    Honestly say something about {object x}
    
    Honestly say something about {object y}
    Dishonestly say something about {object y}
    
    """
    pos_indices = torch.isclose(train_y, torch.ones_like(train_y)).squeeze(1)
    neg_indices = torch.isclose(train_y, torch.zeros_like(train_y)).squeeze(1)
    
    pos_examples = train_X[pos_indices]
    neg_examples = train_X[neg_indices]
    
    dif_vectors = pos_examples - neg_examples
    
    # randomly flip the sign of the vectors
    random_signs = torch.randint(0, 2, (len(dif_vectors),)).float().to(dif_vectors.device) * 2 - 1
    dif_vectors = dif_vectors * random_signs.reshape(-1,1)
    if mean_center:
        dif_vectors -= torch.mean(dif_vectors, dim=0, keepdim=True)

    # dif_vectors : (n//2, d)
    XtX = dif_vectors.T@dif_vectors
    # _, U = torch.linalg.eigh(XtX)
    # return torch.flip(U[:,-n_components:].T, dims=(0,))

    _, U = torch.lobpcg(XtX, k=n_components)
    return U.T

def append_one(X):
    Xb = torch.concat([X, torch.ones_like(X[:,0]).unsqueeze(1)], dim=1)
    new_shape = X.shape[:1] + (X.shape[1]+1,) 
    #print("Xb.shape", Xb.shape, "new_shape", new_shape)
    assert(Xb.shape == new_shape)
    return Xb

def linear_solve(X, y, use_bias=True, reg=0):
    """
    projected_inputs : (n, d)
    labels : (n, c) or (n, )
    """
    print("X.shape", X.shape, "y.shape", y.shape)
    if use_bias:
        inputs = append_one(X)
    else:
        inputs = X
    
    if len(y.shape) == 1:
        y = y.unsqueeze(1)

    num_classes = y.shape[1]
    n, d = inputs.shape
    
    if n>d:   
        XtX = inputs.T@inputs
        XtY = inputs.T@y
        beta = torch.linalg.pinv(XtX + reg*torch.eye(d).to(inputs.device))@XtY # (d, c)
    else:
        XXt = inputs@inputs.T
        alpha = torch.linalg.pinv(XXt + reg*torch.eye(n).to(inputs.device))@y # (n, c)
        beta = inputs.T @ alpha
    
    if use_bias:
        sol = beta[:-1]
        bias = beta[-1]
        if num_classes == 1:
            bias = bias.item()
        return sol, bias
    else:
        return beta
    
def logistic_solve(X, y, C=1):
    """
    projected_inputs : (n, d)
    labels : (n, c)
    """

    num_classes = y.shape[1]
    if num_classes == 1:
        y = y.flatten()
    else:
        y = y.argmax(dim=1)
    model = LogisticRegression(fit_intercept=True, max_iter=10000, C=C) # use bias
    model.fit(X.cpu(), y.cpu())
    
    beta = torch.from_numpy(model.coef_).to(X.dtype).to(X.device)
    bias = torch.from_numpy(model.intercept_).to(X.dtype).to(X.device)
    
    return beta.T, bias

def aggregate_layers_on_test(test_layer_outputs, test_y, agg_beta, agg_bias, regression=False):
    """
    Apply pre-trained aggregation weights to test layer outputs.
    
    Args:
        test_layer_outputs: List of layer outputs for test data
        test_y: Test labels
        agg_beta: Pre-trained aggregation weights
        agg_bias: Pre-trained aggregation bias
        regression: Whether this is a regression task
    
    Returns:
        metrics: Test performance metrics
        test_preds: Test predictions
    """
    # Concatenate test layer outputs
    test_X = torch.concat(test_layer_outputs, dim=1)  # (n, num_layers*n_components)
    
    print("test_X", test_X.shape)
    
    # Apply pre-trained aggregation weights
    test_preds = test_X @ agg_beta + agg_bias
    
    # Apply sigmoid for classification (unless it's regression)
    if not regression:
        test_preds = torch.sigmoid(test_preds)
    
    # Compute metrics
    metrics = compute_prediction_metrics(test_preds, test_y, regression=regression)
    
    return metrics, test_preds


def aggregate_layers(layer_outputs, train_y, val_y, test_y, agg_model='rfm', tuning_metric='accuracy', top_k_layers=None, regression=False, n_random_samples=100):
    """
    Aggregate layer outputs with optional layer selection.
    
    Args:
        layer_outputs: Dictionary with 'train', 'val', 'test' keys containing lists of layer outputs
        train_y, val_y, test_y: Labels for each split
        agg_model: Aggregation model ('rfm', 'logistic', 'linear')
        tuning_metric: Metric to optimize ('auc', 'accuracy', 'f1', etc.)
        top_k_layers: If provided, select top k layers based on validation performance before aggregation
    """
    print("AGGREGATION training with tuning metric:", tuning_metric)
    
    # If top_k_layers is specified, select the best layers based on validation performance
    if top_k_layers is not None and top_k_layers < len(layer_outputs['train']):
        print(f"Selecting top {top_k_layers} layers from {len(layer_outputs['train'])} total layers")
        
        # Evaluate each layer individually on validation set
        layer_scores = []
        for i, val_output in enumerate(layer_outputs['val']):
            # Train a simple linear model on this layer's validation output
            beta, bias = linear_solve(val_output, val_y)
            val_preds = val_output @ beta + bias
            metrics = compute_prediction_metrics(val_preds, val_y, regression=regression)
            layer_scores.append((i, metrics[tuning_metric]))
        
        # Sort by validation performance and select top k
        layer_scores.sort(key=lambda x: x[1], reverse=(tuning_metric in ['f1', 'auc', 'r2', 'accuracy']))
        selected_indices = [idx for idx, score in layer_scores[:top_k_layers]]
        
        print(f"Selected layers (validation {tuning_metric}): {[(idx, score) for idx, score in layer_scores[:top_k_layers]]}")
        
        # Filter layer outputs to only include selected layers
        filtered_outputs = {
            'train': [layer_outputs['train'][i] for i in selected_indices],
            'val': [layer_outputs['val'][i] for i in selected_indices],
            'test': [layer_outputs['test'][i] for i in selected_indices]
        }
        
        layer_outputs = filtered_outputs
    
    # solve aggregator on validation set
    train_X = torch.concat(layer_outputs['train'], dim=1) # (n, num_layers*n_components)    
    val_X = torch.concat(layer_outputs['val'], dim=1) # (n, num_layers*n_components)    
    test_X = torch.concat(layer_outputs['test'], dim=1) # (n, num_layers*n_components)    

    maximize_metric = (tuning_metric in ['f1', 'auc', 'r2', 'accuracy'])

    if agg_model=='rfm':
        search_space = {
            'regs': ["log", [1e-5, 10]],
            'bws': ["log", [1, 100]],
            'center_grads': [True, False],
            'exponents': ["uniform", [0.7, 1.4]],
            'p_interp': ["uniform", [0, 1]]
        }
        print(f"Running {n_random_samples} random hyperparameter samples for RFM aggregation...")

        # Generate random samples for each hyperparameter
        def sample_hyperparams(search_space, n_samples):
            samples = []
            for _ in range(n_samples):
                sample = {}
                
                # Sample reg (log uniform)
                if search_space['regs'][0] == "log":
                    log_min, log_max = np.log(search_space['regs'][1])
                    reg = np.exp(np.random.uniform(log_min, log_max))
                else:
                    reg = np.random.uniform(*search_space['regs'][1])
                sample['reg'] = reg
                
                # Sample bw (log uniform)
                if search_space['bws'][0] == "log":
                    log_min, log_max = np.log(search_space['bws'][1])
                    bw = np.exp(np.random.uniform(log_min, log_max))
                else:
                    bw = np.random.uniform(*search_space['bws'][1])
                sample['bw'] = bw
                
                # Sample center_grads (boolean)
                sample['center_grads'] = np.random.choice(search_space['center_grads'])
                
                # Sample exponent (uniform)
                if search_space['exponents'][0] == "uniform":
                    sample['exponent'] = np.random.uniform(*search_space['exponents'][1])
                else:
                    sample['exponent'] = np.random.choice(search_space['exponents'][1])
                
                # Sample p_interp (uniform)
                if search_space['p_interp'][0] == "uniform":
                    sample['p_interp'] = np.random.uniform(*search_space['p_interp'][1])
                else:
                    sample['p_interp'] = np.random.choice(search_space['p_interp'][1])
                
                samples.append(sample)
            return samples

        # Generate random hyperparameter samples
        hyperparam_samples = sample_hyperparams(search_space, n_random_samples)

        best_rfm_params = None
        best_rfm_score = float('-inf') if maximize_metric else float('inf')
        best_acc = float('-inf')
        
        for i, sample in enumerate(hyperparam_samples):
            reg = sample['reg']
            bw = sample['bw']
            center_grads = sample['center_grads']
            exponent = sample['exponent']
            p_interp = sample['p_interp']
            print("Testing with reg:", reg, "bw:", bw, "center_grads:", center_grads, "exponent:", exponent, "p_interp:", p_interp)
            
            try:
                rfm_params = {
                    'model': {
                        'kernel': 'l2_high_dim',
                        'bandwidth': bw,
                        'exponent': exponent,
                    },
                    'fit': {
                        'reg': reg,
                        'iters': 10,
                        'center_grads': center_grads,
                        'early_stop_rfm': True,
                        'early_stop_multiplier': 1.1, 
                        'get_agop_best_model': True,
                        'top_k': 16  # can change
                    }
                }
                model = xRFM(rfm_params, device='cuda', tuning_metric=tuning_metric)
                model.fit(train_X, train_y, val_X, val_y, verbose=False)
                val_preds = model.predict(val_X)
                metrics = compute_prediction_metrics(val_preds, val_y, regression=regression)
                
                val_score = metrics[tuning_metric]
                acc = metrics.get('accuracy', float('-inf'))

                if (maximize_metric and val_score > best_rfm_score) or (not maximize_metric and val_score < best_rfm_score):
                    print(f'New best RFM aggregation score: {val_score}, acc: {acc}, reg: {reg:.6f}, bw: {bw:.6f}, center_grads: {center_grads}, sample {i+1}/{n_random_samples}')
                    best_rfm_score = val_score
                    best_acc = acc
                    best_rfm_params = deepcopy(rfm_params)

            except Exception as e:
                import traceback
                print(f'Error fitting RFM aggregation (sample {i+1}/{n_random_samples}): {traceback.format_exc()}')
                continue

        print(f'Best RFM aggregation {tuning_metric}: {best_rfm_score}, acc: {best_acc}')
        
        model = xRFM(best_rfm_params, verbose=False, device='cuda', tuning_metric=tuning_metric)
        model.fit(train_X, train_y, val_X, val_y, verbose=False)
        vy = test_y.detach().cpu() if isinstance(test_y, torch.Tensor) else torch.tensor(test_y)
        num_classes = vy.shape[1] if vy.ndim == 2 else 1

        if num_classes > 1 and not regression:
            test_proba = model.predict_proba(test_X)
        else:
            test_proba = model.predict(test_X)

        metrics = compute_prediction_metrics(test_proba, test_y, regression=regression)

        if num_classes > 1 and not regression:
            test_preds = np.argmax(test_proba, axis=1)
        else:
            test_preds = test_proba

        return metrics, None, None, test_preds
    
    elif agg_model=='logistic':
        C_search_space = [1000, 100, 10, 1, 1e-1, 1e-2]
        best_logistic_params = None
        best_logistic_score = float('-inf') if maximize_metric else float('inf')
        for C in C_search_space:
            agg_beta, agg_bias = logistic_solve(train_X, train_y, C=C) # (num_layers*n_components, num_classes)
            val_preds = val_X@agg_beta + agg_bias
            metrics = compute_prediction_metrics(val_preds, val_y)
            if (maximize_metric and metrics[tuning_metric] > best_logistic_score)\
                or (not maximize_metric and metrics[tuning_metric] < best_logistic_score):

                best_logistic_score = metrics[tuning_metric]
                best_logistic_params = (agg_beta, agg_bias)

        agg_beta, agg_bias = best_logistic_params

    elif agg_model=='linear':
        reg_search_space = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        best_linear_params = None
        best_linear_score = float('-inf') if maximize_metric else float('inf')
        for reg in reg_search_space:
            agg_beta, agg_bias = linear_solve(train_X, train_y, reg=reg)
            val_preds = val_X@agg_beta + agg_bias
            val_preds = torch.sigmoid(val_preds)
            metrics = compute_prediction_metrics(val_preds, val_y)
            if (maximize_metric and metrics[tuning_metric] > best_linear_score)\
                or (not maximize_metric and metrics[tuning_metric] < best_linear_score):
                
                best_linear_score = metrics[tuning_metric]
                best_linear_params = (agg_beta, agg_bias)

        agg_beta, agg_bias = best_linear_params

    else:
        raise ValueError(f"Invalid aggregation model: {agg_model}")

    # evaluate aggregated predictor on test set
    test_preds = test_X@agg_beta + agg_bias
    test_preds = torch.sigmoid(test_preds)
    metrics = compute_prediction_metrics(test_preds, test_y)
    return metrics, agg_beta, agg_bias, test_preds

def train_rfm_probe_on_concept(train_X, train_y, val_X, val_y, 
                               hyperparams, search_space=None, 
                               tuning_metric='auc', regression=False, n_random_samples=100):
    """
    Trains an RFM probe with hyperparam search using random sampling and (for multiclass) applies
    temperature scaling on validation probabilities BEFORE computing metrics.
    The learned temperature for the best model is returned in train_results.
    """
    print("INDIVUDAL PROBES training with tuning metric:", tuning_metric)
    if search_space is None:
        search_space = {
            'regs': ["log", [1e-5, 10]],
            'bws': ["log", [1, 100]],
            'center_grads': [True, False],
            'exponents': ["uniform", [0.7, 1.4]],
        }

    best_model = None
    maximize_metric = (tuning_metric in ['f1', 'auc', 'accuracy', 'top_agop_vectors_ols_auc', 'r2'])
    best_score = float('-inf') if maximize_metric else float('inf')
    best_acc = float('-inf')
    best_reg = None
    best_bw = None
    best_center_grads = None
    best_temperature = 1.0  # default if calibration not used
    best_r2 = float('-inf') if regression else None

    # convenience
    is_tensor = lambda x: isinstance(x, torch.Tensor)

    # Generate random samples for each hyperparameter
    def sample_hyperparams(search_space, n_samples):
        samples = []
        for _ in range(n_samples):
            sample = {}
            
            # Sample reg (log uniform)
            if search_space['regs'][0] == "log":
                log_min, log_max = np.log(search_space['regs'][1])
                reg = np.exp(np.random.uniform(log_min, log_max))
            else:
                reg = np.random.uniform(*search_space['regs'][1])
            sample['reg'] = reg
            
            # Sample bw (log uniform)
            if search_space['bws'][0] == "log":
                log_min, log_max = np.log(search_space['bws'][1])
                bw = np.exp(np.random.uniform(log_min, log_max))
            else:
                bw = np.random.uniform(*search_space['bws'][1])
            sample['bw'] = bw
            
            # Sample center_grads (boolean)
            sample['center_grads'] = np.random.choice(search_space['center_grads'])
            
            # Sample exponent (uniform)
            if search_space['exponents'][0] == "uniform":
                sample['exponent'] = np.random.uniform(*search_space['exponents'][1])
            else:
                sample['exponent'] = np.random.choice(search_space['exponents'][1])
            
            samples.append(sample)
        return samples

    # Generate random hyperparameter samples
    hyperparam_samples = sample_hyperparams(search_space, n_random_samples)
    
    print(f"Running {n_random_samples} random hyperparameter samples...")
    
    for i, sample in enumerate(hyperparam_samples):
        reg = sample['reg']
        bw = sample['bw']
        center_grads = sample['center_grads']
        exponent = sample['exponent']
        
       #print("Testing single rfm probe with reg:", reg, "bw:", bw, "center_grads:", center_grads, "exponent:", exponent)
        
        try:
            rfm_params = {
                'model': {
                    'kernel': 'l2_high_dim',
                    'bandwidth': bw,
                    "exponent": exponent,
                },
                'fit': {
                    'reg': reg,
                    'iters': hyperparams['rfm_iters'],
                    'center_grads': center_grads,
                    'early_stop_rfm': True,
                    'early_stop_multiplier': 1.1, 
                    'get_agop_best_model': True,
                    'top_k': hyperparams['n_components']
                }
            }
            model = xRFM(rfm_params, verbose=False, device='cuda', tuning_metric=tuning_metric)

            # squeeze spurious batch dimension if present
            if train_X.shape[0] == 1:
                train_X = train_X.squeeze(0); train_y = train_y.squeeze(0)
            if val_X.shape[0] == 1:
                val_X = val_X.squeeze(0); val_y = val_y.squeeze(0)

            # fit RFM
            model.fit(train_X, train_y, val_X, val_y, verbose=False)
            acc = float('-inf')
            this_temperature = 1.0  # default

            if tuning_metric == 'top_agop_vectors_ols_auc':
                # Special path that evaluates AUC from top AGOP vectors + OLS
                top_k = hyperparams['n_components']
                targets = val_y

                _, U = torch.lobpcg(model.agop_best_model, k=top_k)
                top_eigenvectors = U[:, :top_k]                # [d, k]
                projections = val_X @ top_eigenvectors         # [N, k]
                projections = projections.reshape(-1, top_k)

                XtX = projections.T @ projections
                Xty = projections.T @ targets
                betas = torch.linalg.pinv(XtX) @ Xty           # [k, C] or [k, 1]
                preds = torch.sigmoid(projections @ betas).reshape(targets.shape)

                val_score = roc_auc_score(targets.detach().cpu().numpy(),
                                        preds.detach().cpu().numpy())
                # Calculate accuracy
                acc = accuracy_fn(preds, targets)
            else:
                vy = val_y.detach().cpu() if is_tensor(val_y) else torch.tensor(val_y)
                num_classes = vy.shape[1] if vy.ndim == 2 else 1
                
                if num_classes > 1 and not regression:
                    pred_proba = model.predict_proba(val_X)
                    # Compute labels from probabilities
                else:
                    pred_proba = model.predict(val_X)

                # ensure numpy for compute_prediction_metrics
                if is_tensor(pred_proba):
                    pred_proba_np = pred_proba.detach().cpu().numpy()
                else:
                    pred_proba_np = pred_proba

                prediction_metrics = compute_prediction_metrics(pred_proba_np, val_y, regression=regression)

                val_score = prediction_metrics[tuning_metric]
                if regression:
                    r2 = prediction_metrics['r2']
                acc = prediction_metrics.get('accuracy', float('-inf'))

            # hyperparam selection
            is_better = (val_score > best_score) if maximize_metric else (val_score < best_score)
            if is_better:
                print(f'New best score: {val_score}, acc: {acc}, reg: {reg:.6f}, bw: {bw:.6f}, center_grads: {center_grads}, sample {i+1}/{n_random_samples}')
                best_score = val_score
                best_acc = acc
                best_reg = reg
                best_bw = bw
                best_center_grads = center_grads
                best_model = deepcopy(model)
                best_temperature = this_temperature
                if regression:
                    best_r2 = r2

        except Exception as e:
            import traceback
            print(f'Error fitting RFM (sample {i+1}/{n_random_samples}): {traceback.format_exc()}')
            continue

    print(f'Best RFM {tuning_metric}: {best_score}, acc: {best_acc}, reg: {best_reg:.6f}, bw: {best_bw:.6f}, center_grads: {best_center_grads}, T={best_temperature:.3f}')

    train_results = {
        'score': float(best_score),
        'accuracy': float(best_acc),
        'reg': best_reg,
        'bw': best_bw,
        'center_grads': best_center_grads,
        'temperature': float(best_temperature)   # <-- store learned T for later use on test
    }
    
    if regression:
        train_results['r2'] = best_r2

    return best_model, train_results


def train_linear_probe_on_concept(train_X, train_y, val_X, val_y, use_bias=False, tuning_metric='auc', device='cuda', regression=False):

    reg_search_space = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
    
    if use_bias:
        X = append_one(train_X)
        Xval = append_one(val_X)
    else:
        X = train_X
        Xval = val_X
    
    n, d = X.shape
    num_classes = train_y.shape[1]

    best_beta = None
    maximize_metric = (tuning_metric in ['f1', 'auc', 'accuracy', 'r2'])
    best_score = float('-inf') if maximize_metric else float('inf')
    for reg in reg_search_space:
        try:
            if n>d:
                XtX = batch_transpose_multiply(X, X)
                XtY = batch_transpose_multiply(X, train_y)
                beta = torch.linalg.solve(XtX + reg*torch.eye(X.shape[1]).to(device), XtY)
            else:
                X = X.to(device)
                train_y = train_y.to(device)
                Xval = Xval.to(device)

                XXt = X@X.T
                alpha = torch.linalg.lstsq(XXt + reg*torch.eye(X.shape[0]).to(device), train_y).solution
                beta = X.T@alpha

            preds = Xval.to(device) @ beta
            preds_proba = preds_to_proba(preds)
            val_score = compute_prediction_metrics(preds_proba, val_y, regression=regression)[tuning_metric]

            if maximize_metric and val_score > best_score or not maximize_metric and val_score < best_score:
                best_score = val_score
                best_reg = reg
                best_beta = deepcopy(beta)

        except Exception as e:
            import traceback
            print(f'Error fitting linear probe: {traceback.format_exc()}')
            continue
    
    print(f'Linear probe {tuning_metric}: {best_score}, reg: {best_reg}')

    if use_bias:
        line = best_beta[:-1].to(train_X.device)
        if num_classes == 1:
            bias = best_beta[-1].item()
        else:
            bias = best_beta[-1]
    else:
        line = best_beta.to(train_X.device)
        bias = 0
        
    return line, bias

def train_logistic_probe_on_concept(train_X, train_y, val_X, val_y, use_bias=False, num_classes=1, tuning_metric='auc'):
    
    C_search_space = [1000, 100, 10, 1, 1e-1, 1e-2]

    val_y = val_y.cpu()
    if num_classes == 1:
        train_y_flat = train_y.squeeze(1).cpu()
    else:
        train_y_flat = train_y.argmax(dim=1).cpu()   

    best_beta = None
    best_bias = None
    maximize_metric = (tuning_metric in ['f1', 'auc', 'accuracy', 'r2'])
    best_score = float('-inf') if maximize_metric else float('inf')
    for C in C_search_space:
        model = LogisticRegression(fit_intercept=False, max_iter=1000, C=C)
        model.fit(train_X.cpu(), train_y_flat.cpu())
        
        # Get probability predictions
        val_probs = torch.tensor(model.predict_proba(val_X.cpu()))
        if num_classes == 1:
            val_probs = val_probs[:,1].reshape(val_y.shape)
        val_score = compute_prediction_metrics(val_probs, val_y)[tuning_metric]

        if maximize_metric and val_score > best_score or not maximize_metric and val_score < best_score:
            best_score = val_score
            best_beta = torch.from_numpy(model.coef_).T
            if use_bias:
                best_bias = torch.from_numpy(model.intercept_)
            best_C = C

    print(f'Logistic probe {tuning_metric}: {best_score}, C: {best_C}')

    if use_bias:
        line = best_beta.to(train_X.device)
        if num_classes == 1:
            bias = best_bias.item()
        else:
            bias = best_bias
    else:
        line = best_beta.to(train_X.device)
        bias = 0
        
    return line, bias


def get_hidden_states_music(prompts, model, processor, hidden_layers, forward_batch_size, pooling=None, rep_token=-1, all_positions=False):
    # convert prompts to tensor
    if isinstance(prompts, torch.Tensor):
        input_tensor = prompts
    else:
        input_tensor = torch.cat(prompts, dim=0)

    decoder_input_ids = TensorDataset(input_tensor)
    dataloader = DataLoader(decoder_input_ids, batch_size=forward_batch_size)

    all_hidden_states = {}
    for layer_idx in hidden_layers:
        all_hidden_states[layer_idx] = []
    # Loop over batches and accumulate outputs
    print("Getting activations from forward passes")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch[0]
            #print("input id shape", input_ids.shape)
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            num_layers = 48
            out_hidden_states = outputs.hidden_states
            #print("out_hidden_states", out_hidden_states)
            #print("out_hidden_states shape", len(out_hidden_states), out_hidden_states[0].shape)
            
            hidden_states_all_layers = []
            for layer_idx, hidden_state in zip(range(-1, -num_layers, -1), reversed(out_hidden_states)):
                
                if pooling=="mean":
                    #print("Using mean pooling")
                    all_hidden_states[layer_idx].append(hidden_state.mean(dim=1).detach().cpu())
                elif pooling=="max":
                    #print("Using max pooling")
                    all_hidden_states[layer_idx].append(hidden_state.max(dim=1)[0].detach().cpu())
                elif pooling == "meanstd":
                    #print("Using meanstd pooling")
                    mu = hidden_state.mean(dim=1)                 # [B, d]
                    sigma = hidden_state.std(dim=1, unbiased=False)
                    x = torch.cat([mu, sigma], dim=-1)            # [B, 2d]
                    all_hidden_states[layer_idx].append(x.detach().cpu())
                elif all_positions:
                    all_hidden_states[layer_idx].append(hidden_state.detach().cpu())
                else:
                    # print("hidden_state", hidden_state.shape)
                    # print("rep_token", rep_token)
                    # print("layer_idx", layer_idx)
                    # # which one is the index error
                    # print("all hidden states", all_hidden_states)
                    # print("all hidden states layer idx", all_hidden_states[layer_idx])
                    # print("hidden_state[:,rep_token,:].shape", hidden_state[:,rep_token,:].shape)
                    # print("hidden_state[:,0,:].shape", hidden_state[:,0,:].shape)
                    all_hidden_states[layer_idx].append(hidden_state[:,rep_token,:].detach().cpu())
                      
    # Concatenate results from all batches
    final_hidden_states = {}
    for layer_idx, hidden_state_list in all_hidden_states.items():
        final_hidden_states[layer_idx] = torch.cat(hidden_state_list, dim=0)
        
    return final_hidden_states
