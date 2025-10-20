import torch
from sklearn.linear_model import LogisticRegression
from torch.cuda import tunable
from xrfm import RFM

from . import direction_utils
from .utils import split_indices, split_train_states

import time
from tqdm import tqdm
import numpy as np

class Toolkit:
    def __init__(self):
        pass
        
    def preprocess_data(self, train_data, train_labels, val_data, val_labels, test_data, test_labels, 
                         model, tokenizer, hidden_layers, hyperparams, device='cuda'):
        """
        Handles preprocessing of train/val/test data and extracts hidden states.
        
        Returns:
            train_hidden_states: Dictionary mapping layer names to hidden states
            val_hidden_states: Dictionary mapping layer names to hidden states
            test_hidden_states: Dictionary mapping layer names to hidden states
            train_y: Training labels
            val_y: Validation labels
            test_y: Test labels
            test_data_provided: Boolean indicating if test data was provided
            num_classes: Number of classes in the labels
        """
        train_passed_as_hidden_states = isinstance(train_data, dict)
        val_passed_as_hidden_states = isinstance(val_data, dict)
        test_passed_as_hidden_states = isinstance(test_data, dict)

        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels).reshape(-1,1)
        if val_data is not None and not isinstance(val_labels, torch.Tensor):
            val_labels = torch.tensor(val_labels).reshape(-1,1)

        if len(train_labels.shape) == 1:
            train_labels = train_labels.unsqueeze(-1)
        if val_labels is not None and len(val_labels.shape) == 1:
            val_labels = val_labels.unsqueeze(-1)

        if val_data is None:
            # val data not provided, split train data into train and val
            if train_passed_as_hidden_states:
                assert -1 in train_data.keys(), "train_data must have a key -1 for the last layer"
                train_indices, val_indices = split_indices(len(train_data[-1]))
                train_data, val_data = split_train_states(train_data, train_indices, val_indices)
                val_passed_as_hidden_states = True
            else:
                train_indices, val_indices = split_indices(len(train_data))
                val_data = [train_data[i] for i in val_indices]
                train_data = [train_data[i] for i in train_indices]
                
            all_y = train_labels.float().to(device)
            train_y = all_y[train_indices]
            val_y = all_y[val_indices]
        else:
            train_y = train_labels.float().to(device)
            val_y = val_labels.float().to(device)

        test_data_provided = test_data is not None 
        num_classes = train_y.shape[1]
        
        # Extract hidden states
        if train_passed_as_hidden_states:
            print("Assuming train_data is already a dictionary of hidden states")
            train_hidden_states = train_data
        else:
            train_hidden_states = direction_utils.get_hidden_states(train_data, model, tokenizer, hidden_layers, hyperparams['forward_batch_size'])

        if val_passed_as_hidden_states:
            print("Assuming val_data is already a dictionary of hidden states")
            val_hidden_states = val_data
        else:
            val_hidden_states = direction_utils.get_hidden_states(val_data, model, tokenizer, hidden_layers, hyperparams['forward_batch_size'])

        test_hidden_states = None
        test_y = None
        if test_data_provided:
            if test_passed_as_hidden_states:
                print("Assuming test_data is already a dictionary of hidden states")
                test_hidden_states = test_data
            else:
                test_hidden_states = direction_utils.get_hidden_states(test_data, model, tokenizer, hidden_layers, hyperparams['forward_batch_size'])
            test_y = torch.tensor(test_labels).reshape(-1, num_classes).float().to(device)
        
        return (train_hidden_states, val_hidden_states, test_hidden_states, 
                train_y, val_y, test_y, test_data_provided, num_classes)
    
    def get_layer_data(self, layer_to_eval, train_hidden_states, val_hidden_states, train_y, val_y, device='cuda'):
        """
        Extracts data for a specific layer.
        
        Returns:
            train_X: Training data for the layer
            val_X: Validation data for the layer
        """

        train_X = train_hidden_states[layer_to_eval].float().to(device)
        val_X = val_hidden_states[layer_to_eval].float().to(device)
            
        # print("train X shape:", train_X.shape, "train y shape:", train_y.shape, 
        #       "val X shape:", val_X.shape, "val y shape:", val_y.shape)
        assert(len(train_X) == len(train_y))
        assert(len(val_X) == len(val_y))
        
        return train_X, val_X
    
    def _compute_directions(self, train_data, train_labels, val_data, val_labels, model, tokenizer, hidden_layers, hyperparams,
                            test_data=None, test_labels=None, device='cuda', **kwargs):
        """
        Base implementation to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _compute_directions")


class MusicRFMToolkit(Toolkit):
    def __init__(self):
        super().__init__()

    def preprocess_data(self, train_data, train_labels, val_data, val_labels, test_data, test_labels, 
                         model, processor, hidden_layers, hyperparams, pooling=None, device='cuda'):
        """
        Handles preprocessing of train/val/test data and extracts hidden states.
        
        Returns:
            train_hidden_states: Dictionary mapping layer names to hidden states
            val_hidden_states: Dictionary mapping layer names to hidden states
            test_hidden_states: Dictionary mapping layer names to hidden states
            train_y: Training labels
            val_y: Validation labels
            test_y: Test labels
            test_data_provided: Boolean indicating if test data was provided
            num_classes: Number of classes in the labels
        """
        train_passed_as_hidden_states = isinstance(train_data, dict)
        val_passed_as_hidden_states = isinstance(val_data, dict)
        test_passed_as_hidden_states = isinstance(test_data, dict)

        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels).reshape(-1,1)
        if val_data is not None and not isinstance(val_labels, torch.Tensor):
            val_labels = torch.tensor(val_labels).reshape(-1,1)

        if len(train_labels.shape) == 1:
            train_labels = train_labels.unsqueeze(-1)
        if val_labels is not None and len(val_labels.shape) == 1:
            val_labels = val_labels.unsqueeze(-1)

        if val_data is None:
            # val data not provided, split train data into train and val
            if train_passed_as_hidden_states:
                assert -1 in train_data.keys(), "train_data must have a key -1 for the last layer"
                train_indices, val_indices = split_indices(len(train_data[-1]))
                train_data, val_data = split_train_states(train_data, train_indices, val_indices)
                val_passed_as_hidden_states = True
            else:
                train_indices, val_indices = split_indices(len(train_data))
                val_data = [train_data[i] for i in val_indices]
                train_data = [train_data[i] for i in train_indices]
                
            all_y = train_labels.float().to(device)
            train_y = all_y[train_indices]
            val_y = all_y[val_indices]
        else:
            train_y = train_labels.float().to(device)
            val_y = val_labels.float().to(device)

        test_data_provided = test_data is not None 
        num_classes = train_y.shape[1]
        
        # Extract hidden states
        if train_passed_as_hidden_states:
            print("Assuming train_data is already a dictionary of hidden states")
            train_hidden_states = train_data
        else:
            train_hidden_states = direction_utils.get_hidden_states_music(train_data, model, processor, hidden_layers, hyperparams['forward_batch_size'], pooling=pooling)

        if val_passed_as_hidden_states:
            print("Assuming val_data is already a dictionary of hidden states")
            val_hidden_states = val_data
        else:
            val_hidden_states = direction_utils.get_hidden_states_music(val_data, model, processor, hidden_layers, hyperparams['forward_batch_size'], pooling=pooling)

        test_hidden_states = None
        test_y = None
        if test_data_provided:
            if test_passed_as_hidden_states:
                print("Assuming test_data is already a dictionary of hidden states")
                test_hidden_states = test_data
            else:
                test_hidden_states = direction_utils.get_hidden_states_music(test_data, model, processor, hidden_layers, hyperparams['forward_batch_size'], pooling=pooling)
            test_y = torch.tensor(test_labels).reshape(-1, num_classes).float().to(device)
        
        return (train_hidden_states, val_hidden_states, test_hidden_states, 
                train_y, val_y, test_y, test_data_provided, num_classes)

    def _compute_directions(self, train_data, train_labels, val_data, val_labels, model, processor, hidden_layers, hyperparams,
                            test_data=None, test_labels=None, device='cuda', regression=False, pooling=None, tuning_metric='auc', hyperparam_samples=100, **kwargs):
        
        compare_to_linear = kwargs.get('compare_to_linear', False)
        log_spectrum = kwargs.get('log_spectrum', False)
        log_path = kwargs.get('log_path', None)
        #tuning_metric = kwargs.get('tuning_metric', 'auc')
        classification = kwargs.get('classification', True)

        
        # Process data and extract hidden states
        (train_hidden_states, val_hidden_states, test_hidden_states, 
         train_y, val_y, test_y, test_data_provided, num_classes) = self.preprocess_data(
            train_data, train_labels, val_data, val_labels, test_data, test_labels, 
            model, processor, hidden_layers, hyperparams, pooling, device
        )

        # IMPORTANT: Free up GPU memory from models after hidden state extraction
        print("Clearing model from GPU memory...")
        del model, processor
        torch.cuda.empty_cache()
        
        direction_outputs = {
            'train': [],
            'val': [],
            'test': []
        }
        
        if test_data_provided:
            test_direction_accs = {}
            test_predictor_accs = {}            
            test_predictor_accs = {}            
            #test_y = torch.tensor(test_labels).reshape(-1,1).float().cuda()
            test_y = test_labels.float().cuda()
            test_predictor_accs = {}
            #test_y = torch.tensor(test_labels).reshape(-1,1).float().cuda()
            test_y = test_labels.float().cuda()
        
        n_components = hyperparams['n_components']
        directions = {}
        detector_coefs = {}

        results = {
            'layer_metrics': {},
            'aggregated_metrics': {}
        }

        for layer_to_eval in tqdm(hidden_layers):
            train_X, val_X = self.get_layer_data(layer_to_eval, train_hidden_states, val_hidden_states, train_y, val_y, device)

            start_time = time.time()
            probe_rfm, train_results = direction_utils.train_rfm_probe_on_concept(train_X, train_y, val_X, val_y, hyperparams, tuning_metric=tuning_metric, regression=regression, n_random_samples=hyperparam_samples)
            end_time = time.time()
            print(f"Time taken to train rfm probe: {end_time - start_time} seconds")
            
            if hasattr(probe_rfm, 'M'):
                concept_features = probe_rfm.M
            else:
                concept_features = probe_rfm.collect_best_agops()[0]

            if compare_to_linear:
                _ = direction_utils.train_linear_probe_on_concept(train_X, train_y, val_X, val_y, regression=regression)
    
            start_time = time.time()
            S, U = torch.lobpcg(concept_features, k=n_components)
            end_time = time.time()
            print(f"Time taken to compute eigenvectors: {end_time - start_time} seconds")

            if log_spectrum:
                spectrum_filename = log_path + f'_layer_{layer_to_eval}.pt'
                print("spectrum_filename", spectrum_filename)
                torch.save(S.cpu(), spectrum_filename)

            components = U.T
            directions[layer_to_eval] = components
            
            ### Generate direction accuracy
            # solve for slope, intercept on training data
            vec = directions[layer_to_eval].T
            projected_train = train_X@vec
            
            # Loop over regularization coefficients to find best validation accuracy
            reg_search_space = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
            best_reg_params = None
            best_val_score = float('-inf')

            projected_val = val_X@vec

            for reg in reg_search_space:
                # Fit linear model with current regularization
                beta, b = direction_utils.linear_solve(projected_train, train_y, reg=reg)
                
                # Evaluate on validation set
                val_preds = projected_val@beta + b
                val_preds = val_preds.reshape(-1, num_classes)
                
                val_acc = direction_utils.accuracy_fn(val_preds, val_y)
                
                if val_acc > best_val_score:
                    best_val_score = val_acc
                    best_reg_params = (beta, b, reg)

            # Use best regularization parameters
            beta, b, best_reg = best_reg_params
            print(f"Best regularization for layer {layer_to_eval}: {best_reg}, val_acc: {best_val_score}")

            detector_coefs[layer_to_eval] = [beta, b]
            
            if test_data_provided:
                test_X = test_hidden_states[layer_to_eval].to(device).float()
                test_preds = probe_rfm.predict(test_X) # for multiclass, this generates labels, not OHE
                
                ### Generate predictor accuracy
                if num_classes > 1:
                    test_y_labels = torch.argmax(test_y, axis=1)
                    pred_acc = direction_utils.accuracy_fn(test_preds, test_y_labels, multiclass_labeled=True)
                else:
                    pred_acc = direction_utils.accuracy_fn(test_preds, test_y)
                    
                test_predictor_accs[layer_to_eval] = pred_acc
                 
                ### Generate direction outputs    
                projected_train = train_X@vec         
                projected_val = val_X@vec
                projected_test = test_X@vec

                direction_outputs['train'].append(projected_train.reshape(-1,n_components))
                direction_outputs['val'].append(projected_val.reshape(-1,n_components))
                direction_outputs['test'].append(projected_test.reshape(-1,n_components))
                
                # evaluate slope, intercept on test data using best regularization
                projected_preds = projected_test@beta + b
                projected_preds = projected_preds.reshape(-1,num_classes)
                
                assert(projected_preds.shape==test_y.shape)
                
                dir_acc = direction_utils.accuracy_fn(projected_preds, test_y)
                test_direction_accs[layer_to_eval] = dir_acc

            if 'accuracy' in train_results and (train_results['accuracy'] == -np.inf or train_results['accuracy'] == float('-inf')):
                train_results['accuracy'] = None

            results['layer_metrics'][layer_to_eval] = {
                'train_results': train_results,
            }
                
        signs = {}
        if num_classes == 1: # only if binary do you compute signs
            signs = self._compute_signs(train_hidden_states, train_y, directions, n_components)
            for layer_to_eval in tqdm(hidden_layers):
                for c_idx in range(n_components):
                    directions[layer_to_eval][c_idx] *= signs[layer_to_eval][c_idx]
                
        
        if test_data_provided:
            tuning_metric = 'mse' if regression else 'accuracy'
            metrics, _, _, _ = direction_utils.aggregate_layers(direction_outputs, train_y, val_y, test_y, tuning_metric=tuning_metric, regression=regression)
            test_direction_accs['aggregated'] = metrics

            for key, value in metrics.items():
                metrics[key] = float(value)

            results['aggregated_metrics'] = metrics
        
            return directions, signs, detector_coefs, test_predictor_accs, test_direction_accs, results
        else: 
            return directions, signs, detector_coefs, None, None, results

    def _compute_signs(self, hidden_states, all_y, directions, n_components):
        
        signs = {}
        for layer in hidden_states.keys():
            xs = hidden_states[layer]
            signs[layer] = {}
            for c_idx in range(n_components):
                direction = directions[layer][c_idx]
                hidden_state_projections = direction_utils.project_onto_direction(xs, direction).to(all_y.device)
                sign = 2*(direction_utils.pearson_corr(all_y.squeeze(1), hidden_state_projections) > 0) - 1
                signs[layer][c_idx] = sign.item()

        return signs


class MusicLinearProbeToolkit(Toolkit):
    def __init__(self):
        super().__init__()

    def preprocess_data(self, train_data, train_labels, val_data, val_labels, test_data, test_labels, 
                         model, processor, hidden_layers, hyperparams, pooling=None, device='cuda'):
        """
        Handles preprocessing of train/val/test data and extracts hidden states.
        
        Returns:
            train_hidden_states: Dictionary mapping layer names to hidden states
            val_hidden_states: Dictionary mapping layer names to hidden states
            test_hidden_states: Dictionary mapping layer names to hidden states
            train_y: Training labels
            val_y: Validation labels
            test_y: Test labels
            test_data_provided: Boolean indicating if test data was provided
            num_classes: Number of classes in the labels
        """
        train_passed_as_hidden_states = isinstance(train_data, dict)
        val_passed_as_hidden_states = isinstance(val_data, dict)
        test_passed_as_hidden_states = isinstance(test_data, dict)

        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels).reshape(-1,1)
        if val_data is not None and not isinstance(val_labels, torch.Tensor):
            val_labels = torch.tensor(val_labels).reshape(-1,1)

        if len(train_labels.shape) == 1:
            train_labels = train_labels.unsqueeze(-1)
        if val_labels is not None and len(val_labels.shape) == 1:
            val_labels = val_labels.unsqueeze(-1)

        if val_data is None:
            # val data not provided, split train data into train and val
            if train_passed_as_hidden_states:
                assert -1 in train_data.keys(), "train_data must have a key -1 for the last layer"
                train_indices, val_indices = split_indices(len(train_data[-1]))
                train_data, val_data = split_train_states(train_data, train_indices, val_indices)
                val_passed_as_hidden_states = True
            else:
                train_indices, val_indices = split_indices(len(train_data))
                val_data = [train_data[i] for i in val_indices]
                train_data = [train_data[i] for i in train_indices]
                
            all_y = train_labels.float().to(device)
            train_y = all_y[train_indices]
            val_y = all_y[val_indices]
        else:
            train_y = train_labels.float().to(device)
            val_y = val_labels.float().to(device)

        test_data_provided = test_data is not None 
        num_classes = train_y.shape[1]
        
        # Extract hidden states
        if train_passed_as_hidden_states:
            print("Assuming train_data is already a dictionary of hidden states")
            train_hidden_states = train_data
        else:
            train_hidden_states = direction_utils.get_hidden_states_music(train_data, model, processor, hidden_layers, hyperparams['forward_batch_size'], pooling=pooling)

        if val_passed_as_hidden_states:
            print("Assuming val_data is already a dictionary of hidden states")
            val_hidden_states = val_data
        else:
            val_hidden_states = direction_utils.get_hidden_states_music(val_data, model, processor, hidden_layers, hyperparams['forward_batch_size'], pooling=pooling)

        test_hidden_states = None
        test_y = None
        if test_data_provided:
            if test_passed_as_hidden_states:
                print("Assuming test_data is already a dictionary of hidden states")
                test_hidden_states = test_data
            else:
                test_hidden_states = direction_utils.get_hidden_states_music(test_data, model, processor, hidden_layers, hyperparams['forward_batch_size'], pooling=pooling)
            test_y = torch.tensor(test_labels).reshape(-1, num_classes).float().to(device)
        
        return (train_hidden_states, val_hidden_states, test_hidden_states, 
                train_y, val_y, test_y, test_data_provided, num_classes)

    def _compute_directions(self, train_data, train_labels, val_data, val_labels, model, processor, hidden_layers, hyperparams,
                            test_data=None, test_labels=None, device='cuda', regression=False, pooling=None, **kwargs):
        
        tuning_metric = kwargs.get('tuning_metric', 'auc')
        classification = kwargs.get('classification', True)

        print("Tuning metric:", tuning_metric)
        
        # Process data and extract hidden states
        (train_hidden_states, val_hidden_states, test_hidden_states, 
         train_y, val_y, test_y, test_data_provided, num_classes) = self.preprocess_data(
            train_data, train_labels, val_data, val_labels, test_data, test_labels, 
            model, processor, hidden_layers, hyperparams, pooling, device
        )

        # IMPORTANT: Free up GPU memory from models after hidden state extraction
        print("Clearing model from GPU memory...")
        del model, processor
        torch.cuda.empty_cache()
        
        direction_outputs = {
            'train': [],
            'val': [],
            'test': []
        }

        if test_data_provided:
            test_direction_accs = {}
            test_predictor_accs = {}
            test_y = test_labels.float().cuda()

        directions = {}
        detector_coefs = {}

        results = {
            'layer_metrics': {},
            'aggregated_metrics': {}
        }

        for layer_to_eval in tqdm(hidden_layers):
            # Get data for this layer
            train_X, val_X = self.get_layer_data(layer_to_eval, train_hidden_states, val_hidden_states, train_y, val_y, device)
            
            start_time = time.time()
            beta, bias = direction_utils.train_linear_probe_on_concept(train_X, train_y, val_X, val_y, tuning_metric=tuning_metric, regression=regression)
            end_time = time.time()
            print(f"Time taken to train linear probe: {end_time - start_time} seconds")
            
            assert(len(beta)==train_X.shape[1])
            if num_classes == 1: # assure beta is (num_classes, num_features)
                beta = beta.reshape(1,-1) 
            else:
                beta = beta.T
            beta /= beta.norm(dim=1, keepdim=True)
            directions[layer_to_eval] = beta
            
            ### Generate direction accuracy
            # solve for slope, intercept on training data
            vec = beta.T
            projected_train = train_X@vec
            
            # Loop over regularization coefficients to find best validation accuracy
            reg_search_space = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
            best_reg_params = None
            best_val_score = float('-inf')

            projected_val = val_X@vec

            for reg in reg_search_space:
                # Fit linear model with current regularization
                m, b = direction_utils.linear_solve(projected_train, train_y, reg=reg)
                
                # Evaluate on validation set
                val_preds = projected_val@m + b
                val_preds = val_preds.reshape(-1, num_classes)
                
                val_acc = direction_utils.accuracy_fn(val_preds, val_y)
                
                if val_acc > best_val_score:
                    best_val_score = val_acc
                    best_reg_params = (m, b, reg)

            # Use best regularization parameters
            m, b, best_reg = best_reg_params
            print(f"Best regularization for layer {layer_to_eval}: {best_reg}, val_acc: {best_val_score}")
            
            detector_coefs[layer_to_eval] = [m, b]
                
            if test_data_provided:
                test_X = test_hidden_states[layer_to_eval].to(device).float()
                
                ### Generate predictor outputs
                test_preds = test_X@vec + bias
                
                ### Generate predictor accuracy
                pred_acc = direction_utils.accuracy_fn(test_preds, test_y)
                test_predictor_accs[layer_to_eval] = pred_acc
                
                ### Generate direction outputs                
                projected_train = train_X@vec         
                projected_val = val_X@vec
                projected_test = test_X@vec

                direction_outputs['train'].append(projected_train.reshape(-1,num_classes))
                direction_outputs['val'].append(projected_val.reshape(-1,num_classes))
                direction_outputs['test'].append(projected_test.reshape(-1,num_classes))
                
                ### Generate direction accuracy
                # evaluate slope, intercept on test data using best regularization
                projected_preds = projected_test@m + b
                projected_preds = projected_preds.reshape(-1, num_classes)
                
                assert(projected_preds.shape==test_y.shape)
                
                dir_acc = direction_utils.accuracy_fn(projected_preds, test_y)
                test_direction_accs[layer_to_eval] = dir_acc

            results['layer_metrics'][layer_to_eval] = {
                'train_results': {'val_acc': best_val_score, 'best_reg': best_reg},
            }
        
        signs = {}
        if num_classes == 1: # only if binary do you compute signs
            signs = self._compute_signs(train_hidden_states, train_y, directions)
            for layer_to_eval in tqdm(hidden_layers):
                directions[layer_to_eval][0] *= signs[layer_to_eval][0] # only one direction, index 0
            
        
        if test_data_provided:
            print("Aggregating predictions over layers using linear stacking")
            direction_agg_acc = direction_utils.aggregate_layers(direction_outputs, train_y, val_y, test_y, tuning_metric=tuning_metric, regression=regression)
            test_direction_accs['linear_agg'] = direction_agg_acc

            return directions, signs, detector_coefs, test_predictor_accs, test_direction_accs, results
        else: 
            return directions, signs, detector_coefs, None, None, results

    def _compute_signs(self, hidden_states, all_y, directions):
        
        signs = {}
        for layer in hidden_states.keys():
            xs = hidden_states[layer]
            signs[layer] = {}
            c_idx = 0
            direction = directions[layer][c_idx]
            hidden_state_projections = direction_utils.project_onto_direction(xs, direction).to(all_y.device)
            sign = 2*(direction_utils.pearson_corr(all_y.squeeze(1), hidden_state_projections) > 0) - 1
            signs[layer][c_idx] = sign.item()

        return signs



class MusicLogisticRegressionToolkit(Toolkit):
    def __init__(self):
        super().__init__()

    def preprocess_data(self, train_data, train_labels, val_data, val_labels, test_data, test_labels, 
                         model, processor, hidden_layers, hyperparams, pooling=None, device='cuda'):
        """
        Handles preprocessing of train/val/test data and extracts hidden states.
        
        Returns:
            train_hidden_states: Dictionary mapping layer names to hidden states
            val_hidden_states: Dictionary mapping layer names to hidden states
            test_hidden_states: Dictionary mapping layer names to hidden states
            train_y: Training labels
            val_y: Validation labels
            test_y: Test labels
            test_data_provided: Boolean indicating if test data was provided
            num_classes: Number of classes in the labels
        """
        train_passed_as_hidden_states = isinstance(train_data, dict)
        val_passed_as_hidden_states = isinstance(val_data, dict)
        test_passed_as_hidden_states = isinstance(test_data, dict)

        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels).reshape(-1,1)
        if val_data is not None and not isinstance(val_labels, torch.Tensor):
            val_labels = torch.tensor(val_labels).reshape(-1,1)

        if len(train_labels.shape) == 1:
            train_labels = train_labels.unsqueeze(-1)
        if val_labels is not None and len(val_labels.shape) == 1:
            val_labels = val_labels.unsqueeze(-1)

        if val_data is None:
            # val data not provided, split train data into train and val
            if train_passed_as_hidden_states:
                assert -1 in train_data.keys(), "train_data must have a key -1 for the last layer"
                train_indices, val_indices = split_indices(len(train_data[-1]))
                train_data, val_data = split_train_states(train_data, train_indices, val_indices)
                val_passed_as_hidden_states = True
            else:
                train_indices, val_indices = split_indices(len(train_data))
                val_data = [train_data[i] for i in val_indices]
                train_data = [train_data[i] for i in train_indices]
                
            all_y = train_labels.float().to(device)
            train_y = all_y[train_indices]
            val_y = all_y[val_indices]
        else:
            train_y = train_labels.float().to(device)
            val_y = val_labels.float().to(device)

        test_data_provided = test_data is not None 
        num_classes = train_y.shape[1]
        
        # Extract hidden states
        if train_passed_as_hidden_states:
            print("Assuming train_data is already a dictionary of hidden states")
            train_hidden_states = train_data
        else:
            train_hidden_states = direction_utils.get_hidden_states_music(train_data, model, processor, hidden_layers, hyperparams['forward_batch_size'], pooling=pooling)

        if val_passed_as_hidden_states:
            print("Assuming val_data is already a dictionary of hidden states")
            val_hidden_states = val_data
        else:
            val_hidden_states = direction_utils.get_hidden_states_music(val_data, model, processor, hidden_layers, hyperparams['forward_batch_size'], pooling=pooling)

        test_hidden_states = None
        test_y = None
        if test_data_provided:
            if test_passed_as_hidden_states:
                print("Assuming test_data is already a dictionary of hidden states")
                test_hidden_states = test_data
            else:
                test_hidden_states = direction_utils.get_hidden_states_music(test_data, model, processor, hidden_layers, hyperparams['forward_batch_size'], pooling=pooling)
            test_y = torch.tensor(test_labels).reshape(-1, num_classes).float().to(device)
        
        return (train_hidden_states, val_hidden_states, test_hidden_states, 
                train_y, val_y, test_y, test_data_provided, num_classes)

    def _compute_directions(self, train_data, train_labels, val_data, val_labels, model, processor, hidden_layers, hyperparams,
                            test_data=None, test_labels=None, device='cuda', regression=False, pooling=None, **kwargs):
        
        tuning_metric = kwargs.get('tuning_metric', 'auc')
        classification = kwargs.get('classification', True)

        print("Tuning metric:", tuning_metric)
                
        # Process data and extract hidden states
        (train_hidden_states, val_hidden_states, test_hidden_states, 
         train_y, val_y, test_y, test_data_provided, num_classes) = self.preprocess_data(
            train_data, train_labels, val_data, val_labels, test_data, test_labels, 
            model, processor, hidden_layers, hyperparams, pooling, device
        )

        # IMPORTANT: Free up GPU memory from models after hidden state extraction
        print("Clearing model from GPU memory...")
        del model, processor
        torch.cuda.empty_cache()
        
        direction_outputs = {
            'train': [],
            'val': [],
            'test': []
        }

        if test_data_provided:
            test_direction_accs = {}
            test_predictor_accs = {}
            test_y = test_labels.float().cuda()

        directions = {}
        detector_coefs = {}

        results = {
            'layer_metrics': {},
            'aggregated_metrics': {}
        }

        for layer_to_eval in tqdm(hidden_layers):
            # Get data for this layer
            train_X, val_X = self.get_layer_data(layer_to_eval, train_hidden_states, val_hidden_states, train_y, val_y, device)
            
            start_time = time.time()
            print("Training logistic regression")
            beta, bias = direction_utils.train_logistic_probe_on_concept(train_X, train_y, val_X, val_y, num_classes=num_classes, tuning_metric=tuning_metric)
            end_time = time.time()
            print(f"Time taken to train logistic probe: {end_time - start_time} seconds")
            
            concept_features = beta.to(train_X.dtype).T
            if num_classes == 1:
                concept_features = concept_features.reshape(1,-1)

            assert(concept_features.shape == (num_classes, train_X.size(1)))
            concept_features /= concept_features.norm(dim=1, keepdim=True)

            directions[layer_to_eval] = concept_features
            
             # solve for slope, intercept on training data
            vec = concept_features.T.to(device=train_X.device)
            projected_train = train_X@vec
            
            # Loop over regularization coefficients to find best validation accuracy
            reg_search_space = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
            best_reg_params = None
            best_val_score = float('-inf')

            projected_val = val_X@vec

            for reg in reg_search_space:
                # Fit linear model with current regularization
                m, b = direction_utils.linear_solve(projected_train, train_y, reg=reg)
                
                # Evaluate on validation set
                val_preds = projected_val@m + b
                val_preds = val_preds.reshape(-1, num_classes)
                
                val_acc = direction_utils.accuracy_fn(val_preds, val_y)
                
                if val_acc > best_val_score:
                    best_val_score = val_acc
                    best_reg_params = (m, b, reg)

            # Use best regularization parameters
            m, b, best_reg = best_reg_params
            print(f"Best regularization for layer {layer_to_eval}: {best_reg}, val_acc: {best_val_score}")
            
            detector_coefs[layer_to_eval] = [m, b]
                
            if test_data_provided:
                test_X = test_hidden_states[layer_to_eval].to(device).float()
                
                ### Generate predictor outputs using logistic regression
                # Create a simple sklearn logistic regression model for predictions
                from sklearn.linear_model import LogisticRegression
                
                if num_classes == 1:
                    train_y_flat = train_y.squeeze(1).cpu()
                else:
                    train_y_flat = train_y.argmax(dim=1).cpu()
                
                logistic_model = LogisticRegression(fit_intercept=True, max_iter=1000)
                logistic_model.fit(train_X.cpu(), train_y_flat.cpu())
                test_preds = torch.from_numpy(logistic_model.predict_proba(test_X.cpu())).to(test_y.device)
                
                if num_classes == 1:
                    test_preds = test_preds[:,1].reshape(test_y.shape)
                
                ### Generate predictor accuracy
                pred_acc = direction_utils.accuracy_fn(test_preds, test_y)
                test_predictor_accs[layer_to_eval] = pred_acc
                
                ### Generate direction outputs                
                projected_train = train_X@vec         
                projected_val = val_X@vec
                projected_test = test_X@vec

                direction_outputs['train'].append(projected_train.reshape(-1, num_classes))
                direction_outputs['val'].append(projected_val.reshape(-1, num_classes))
                direction_outputs['test'].append(projected_test.reshape(-1, num_classes))
                
                ### Generate direction accuracy
                # evaluate slope, intercept on test data using best regularization
                projected_preds = projected_test@m + b
                projected_preds = projected_preds.reshape(-1, num_classes)
                
                assert(projected_preds.shape==test_y.shape)
                
                dir_acc = direction_utils.accuracy_fn(projected_preds, test_y)
                test_direction_accs[layer_to_eval] = dir_acc
        
            results['layer_metrics'][layer_to_eval] = {
                'train_results': {'val_acc': best_val_score, 'best_reg': best_reg},
            }
        
        signs = {}
        if num_classes == 1: # only if binary do you compute signs
            signs = self._compute_signs(train_hidden_states, train_y, directions)
            for layer_to_eval in tqdm(hidden_layers):
                directions[layer_to_eval][0] *= signs[layer_to_eval][0] # only one direction, index 0
            
        
        if test_data_provided:
            metrics, _, _, _ = direction_utils.aggregate_layers(direction_outputs, train_y, val_y, test_y, tuning_metric=tuning_metric, regression=regression)
            test_direction_accs['aggregated'] = metrics

            for key, value in metrics.items():
                metrics[key] = float(value)

            results['aggregated_metrics'] = metrics

            return directions, signs, detector_coefs, test_predictor_accs, test_direction_accs, results
        else: 
            return directions, signs, detector_coefs, None, None, results

    def _compute_signs(self, hidden_states, all_y, directions):
        signs = {}
        for layer in hidden_states.keys():
            xs = hidden_states[layer]
            signs[layer] = {}
            c_idx = 0
            direction = directions[layer][c_idx]
            hidden_state_projections = direction_utils.project_onto_direction(xs, direction).to(all_y.device)
            sign = 2*(direction_utils.pearson_corr(all_y.squeeze(1), hidden_state_projections) > 0) - 1
            signs[layer][c_idx] = sign.item()

        return signs
