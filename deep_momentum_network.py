# Based on the papers:
# 
# @article{wood2021trading,
#   title={Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture},
#   author={Wood, Kieran and Giegerich, Sven and Roberts, Stephen and Zohren, Stefan},
#   journal={arXiv preprint arXiv:2112.08534},
#   year={2021}
# }
# 
# @article {Wood111,
# 	author = {Wood, Kieran and Roberts, Stephen and Zohren, Stefan},
# 	title = {Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection},
# 	volume = {4},
# 	number = {1},
# 	pages = {111--129},
# 	year = {2022},
# 	doi = {10.3905/jfds.2021.1.081},
# 	publisher = {Institutional Investor Journals Umbrella},
# 	issn = {2640-3943},
# 	URL = {https://jfds.pm-research.com/content/4/1/111},
# 	eprint = {https://jfds.pm-research.com/content/4/1/111.full.pdf},
# 	journal = {The Journal of Financial Data Science}
# }


import os

os.environ['CUDA_HOME'] = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2" 
os.environ['PATH'] += r";C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin" 
os.environ['PATH'] += r";C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/extras/CUPTI/libx64" 
os.environ['PATH'] += r";C:/tools/cuda/bin"

from keras_tuner.tuners import RandomSearch
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import keras_tuner as kt
from empyrical import sharpe_ratio
import collections
import copy
from abc import ABC, abstractmethod

from model_inputs import ModelFeatures

class SharpeLoss(tf.keras.losses.Loss):
    def __init__(self, output_size: int = 1):
        self.output_size = output_size
        super().__init__()
    
    def call(self, y_true, weights):
        captured_returns = weights * y_true
        mean_returns = tf.reduce_mean(captured_returns)
        return - (mean_returns / tf.sqrt(tf.reduce_mean(tf.square(captured_returns)) - tf.square(mean_returns) + 1e-9) * tf.sqrt(252.0))


class SharpeValidationLoss(keras.callbacks.Callback):
    def __init__(
        self,
        inputs,
        returns,
        time_indices,
        num_time,  # including a count for nulls which will be indexed as 0
        early_stopping_patience,
        n_multiprocessing_workers,
        weights_save_location="tmp/checkpoint",
        min_delta=1e-4
    ):
        super(keras.callbacks.Callback, self).__init__()
        self.inputs = inputs
        self.returns = returns
        self.time_indices = time_indices
        self.n_multiprocessing_workers = n_multiprocessing_workers
        self.early_stopping_patience = early_stopping_patience
        self.num_time = num_time
        self.min_delta = min_delta

        self.best_sharpe = np.NINF 
        self.weights_save_location = weights_save_location

    def set_weights_save_loc(self, weights_save_location):
        self.weights_save_location = weights_save_location

    def on_train_begin(self, logs=None):
        self.patience_counter = 0
        self.stopped_epoch = 0
        self.best_sharpe = np.NINF

    def on_epoch_end(self, epoch, logs=None):
        positions = self.model.predict(
            self.inputs,
            workers=self.n_multiprocessing_workers,
            use_multiprocessing=True
        )

        captured_returns = tf.math.unsorted_segment_mean(
            positions * self.returns, self.time_indices, self.num_time
        )[1:]

        sharpe = (
            tf.reduce_mean(captured_returns)
            / tf.sqrt(
                tf.math.reduce_variance(captured_returns)
                + tf.constant(1e-9, dtype=tf.float64)
            )
            * tf.sqrt(tf.constant(252.0, dtype=tf.float64))
        ).numpy()
        if sharpe > self.best_sharpe + self.min_delta:
            self.best_sharpe = sharpe
            self.patience_counter = 0  # reset the count
            print("better")
            print(os.path.normpath("./" + self.weights_save_location))
            self.model.save_weights(os.path.normpath("./" + self.weights_save_location))
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(os.path.normpath(self.weights_save_location))
                self.model.load_weights(os.path.normpath(self.weights_save_location))
        logs["sharpe"] = sharpe  # for keras tuner
        print(f"\nval_sharpe {logs['sharpe']}")


class TunerValidationLoss(kt.tuners.RandomSearch):
    def __init__(self, hypermodel, objective, max_trials, hp_minibatch_size, seed=None, hyperparameters=None, 
                 tune_new_entries=True, allow_new_entries=True, **kwargs):
        
        self.hp_minibatch_size = hp_minibatch_size
        super().__init__(hypermodel, objective, max_trials, seed, hyperparameters, tune_new_entries, allow_new_entries, **kwargs)
    

    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", values=self.hp_minibatch_size
        )
        super(TunerValidationLoss, self).run_trial(trial, *args, **kwargs)


class TunerDiversifiedSharpe(kt.tuners.RandomSearch):
    def __init__(self, hypermodel, objective, max_trials, hp_minibatch_size, seed=None, hyperparameters=None, 
                 tune_new_entries=True, allow_new_entries=True, **kwargs):
        
        self.hp_minibatch_size = hp_minibatch_size
        super().__init__(
            hypermodel,
            objective,
            max_trials,
            seed,
            hyperparameters,
            tune_new_entries,
            allow_new_entries,
            **kwargs
        )
    

    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", values=self.hp_minibatch_size
        )

        original_callbacks = kwargs.pop("callbacks", [])

        for callback in original_callbacks:
            print("test")
            if isinstance(callback, SharpeValidationLoss):
                print(callback.weights_save_location)
                callback.set_weights_save_loc(
                    self._get_checkpoint_fname(trial.trial_id, self._reported_step)
                )

        metrics = collections.defaultdict(list)
        for execution in range(self.executions_per_trial):
            copied_fit_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(kt.engine.tuner_utils.TunerCallback(self, trial)) 
            copied_fit_kwargs["callbacks"] = callbacks

            history = self._build_and_fit_model(trial, args, copied_fit_kwargs)
            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == "min":
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)
        
        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(trial.trial_id, metrics=averaged_metrics, step=self._reported_step)


class DeepMomentumNetworkModel(ABC):
    def __init__(self, project_name, hp_directory, hp_minibatch_size, **params):
        params = params.copy()

        self.time_steps = int(params["total_time_steps"])
        self.input_size = int(params["input_size"])
        self.output_size = int(params["output_size"])
        self.n_multiprocessing_workers = int(params["multiprocessing_workers"])
        self.num_epochs = int(params["num_epochs"])
        self.early_stopping_patience = int(params["early_stopping_patience"])
        self.random_search_iterations = params["random_search_iterations"]
        self.evaluate_diversified_val_sharpe = params["evaluate_diversified_val_sharpe"]
        self.force_output_sharpe_length = params["force_output_sharpe_length"]


        print("Deep Momentum Network Parameters:")
        for k in params:
            print(f"{k} = {params[k]}")
        

        def model_builder(hp):
            return self.model_builder(hp)
    
        if self.evaluate_diversified_val_sharpe:
            self.tuner = TunerDiversifiedSharpe(model_builder, objective=kt.Objective("sharpe", "max"), hp_minibatch_size=hp_minibatch_size,
                                                max_trials=self.random_search_iterations, directory=hp_directory, project_name=project_name)
        else:
            self.tuner = TunerValidationLoss(model_builder, objective="val_loss", hp_minibatch_size=hp_minibatch_size,
                                             max_trials=self.random_search_iterations, directory=hp_directory, project_name=project_name)
        

    @abstractmethod
    def model_builder(self, hp):
        return
    
    @staticmethod
    def _index_times(val_time):
        val_time_unique = np.sort(np.unique(val_time))
        if val_time_unique[0]:
            val_time_unique = np.insert(val_time_unique, 0, "")
        mapping = dict(zip(val_time_unique, range(len(val_time_unique))))

        @np.vectorize
        def get_indices(t):
            return mapping[t]
        
        return get_indices(val_time), len(mapping)
    


    def hyperparameter_search(self, train_data, valid_data):
        data, labels, active_flags, _, _ = ModelFeatures._unpack(train_data)
        val_data, val_labels, val_flags, _, val_time = ModelFeatures._unpack(valid_data)

        if self.evaluate_diversified_val_sharpe:
            val_time_indices, num_val_time = self._index_times(val_time)
            callbacks = [
                SharpeValidationLoss(
                    val_data,
                    val_labels,
                    val_time_indices,
                    num_val_time,
                    self.early_stopping_patience,
                    self.n_multiprocessing_workers
                ),
                tf.keras.callbacks.TerminateOnNaN()
            ]

            self.tuner.search(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True,
                workers=self.n_multiprocessing_workers
            )
        else:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    min_delta=1e-4
                ),
            ]
            self.tuner.search(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                # batch_size=minibatch_size,
                # covered by Tuner class
                validation_data=(
                    val_data,
                    val_labels,
                    val_flags
                ),
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True,
                workers=self.n_multiprocessing_workers
            )

        best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0].values
        best_model = self.tuner.get_best_models(num_models=1)[0]
        return best_hp, best_model


    def load_model(self, hyperparameters):
        hyp = kt.engine.hyperparameters.HyperParamters()
        hyp.values = hyperparameters
        return self.tuner.hypermodel.build(hyp)
    

    def fit(self, train_data, valid_data, hyperparameters, temp_folder):
        data, labels, active_flags, _, _ = ModelFeatures._unpack(train_data)
        val_data, val_labels, val_flags, _, val_time = ModelFeatures._unpack(valid_data)

        model = self.load_model(hyperparameters)

        if self.evaluate_diversified_val_sharpe:
            val_time_indices, num_val_time = self._index_times(val_time)
            callbacks = [
                SharpeValidationLoss(
                    val_data,
                    val_labels,
                    val_time_indices,
                    num_val_time,
                    self.early_stopping_patience,
                    self.n_multiprocessing_workers,
                    weights_save_location=temp_folder
                ),
                tf.keras.callbacks.TerminateOnNaN()
            ]

            model.fit(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                batch_size=hyperparameters["batch_size"],
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True,
                workers=self.n_multiprocessing_workers
            )
            model.load_weights(temp_folder)
        else:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    min_delta=1e-4,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.TerminateOnNaN()
            ]

            model.fit(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                batch_size=hyperparameters["batch_size"],
                validation_data=(
                    val_data,
                    val_labels,
                    val_flags
                ),
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True,
                workers=self.n_multiprocessing_workers
            )

        return model
    
    def evaluate(self, data, model):
        inputs, outputs, active_entries, _, _ = ModelFeatures._unpack(data)

        if self.evaluate_diversified_val_sharpe:
            _, performance = self.get_positions(data, model, False)
            return performance
        
        else:
            metric_values = model.evaluate(
                x=inputs,
                y=outputs,
                sample_weight=active_entries,
                workers=32,
                use_multiprocessing=True
            )
            metrics = pd.Series(metric_values, model.metrics_name)
            return metrics["loss"]
    
    def get_positions(self, data, model, sliding_window=True, years_geq=np.iinfo(np.int32).min, years_lt=np.iinfo(np.int32).max):
        inputs, outputs, _, identifier, time = ModelFeatures._unpack(data)
        if sliding_window:
            time = pd.to_datetime(
                time[:, -1, 0].flatten()
            )
            years = time.map(lambda t: t.year)
            identifier = identifier[:, -1, 0].flatten()
            returns = outputs[:, -1, 0].flatten()
        else:
            time = pd.to_datetime(time.flatten())
            years = time.map(lambda t: t.year)
            identifier = identifier.flatten()
            returns = outputs.flatten()
        
        mask = (years >= years_geq) & (years < years_lt)

        positions = model.predict(inputs, workers=self.n_multiprocessing_workers, use_multiprocessing=True)
        
        if sliding_window:
            positions = positions[:, -1, 0].flatten()
        else:
            positions = positions.flatten()

        captured_returns = returns * positions
        results = pd.DataFrame(
            {
                "identifier": identifier[mask],
                "time": time[mask],
                "returns": returns[mask],
                "position": positions[mask],
                "captured_returns": captured_returns[mask]
            }
        )

        performance = sharpe_ratio(results.groupby("time")["captured_returns"].sum())

        return results, performance


class LstmDeepMomentumNetworkModel(DeepMomentumNetworkModel):
    def __init__(self, project_name, hp_directory, hp_minibatch_size=[64, 128, 256], **params):
        super().__init__(project_name, hp_directory, hp_minibatch_size, **params)
    

    def model_builder(self, hp):
        hidden_layer_size = hp.Choice("hidden_layer_size", values=[5, 10, 20, 40, 80, 160])
        dropout_rate = hp.Choice("dropout_rate", values=[0.1, 0.2, 0.3, 0.4, 0.5])
        max_gradient_norm = hp.Choice("max_gradient_norm", values=[0.01, 1.0, 100.0])
        learning_rate = hp.Choice("learning_rate", values=[1e-4, 1e-3, 1e-2, 1e-1])

        input = keras.Input((self.time_steps, self.input_size))
        lstm = tf.keras.layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            dropout=dropout_rate,
            stateful=False,
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0,
            unroll=False,
            use_bias=True
        )(input)

        dropout = keras.layers.Dropout(dropout_rate)(lstm)

        output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                self.output_size,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3)
            )
        )(dropout[..., :, :])

        model = keras.Model(inputs=input, outputs=output)

        adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)

        sharpe_loss = SharpeLoss(self.output_size).call

        model.compile(
            loss=sharpe_loss,
            optimizer=adam,
            sample_weight_mode="temporal"
        )
        return model


