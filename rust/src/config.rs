use crate::loss::{HalfMSE, Loss};
use crate::optimizer::OptimizerConfig;
use crate::Float;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EarlyStoppingMetric {
    ValidationLoss,
    ValidationAccuracy,
}

impl EarlyStoppingMetric {
    pub(crate) fn score<F: Float>(&self, val_accuracy: f64, val_loss: F) -> F {
        match self {
            Self::ValidationLoss => val_loss,
            Self::ValidationAccuracy => F::from_f64(val_accuracy).unwrap(),
        }
    }

    pub(crate) fn is_improvement<F: Float>(&self, score: F, best_score: F) -> bool {
        match self {
            Self::ValidationLoss => score < best_score,
            Self::ValidationAccuracy => score > best_score,
        }
    }

    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::ValidationLoss => "Val Loss",
            Self::ValidationAccuracy => "Val Accuracy",
        }
    }

    pub(crate) fn format_score<F: Float>(&self, score: F) -> String {
        match self {
            Self::ValidationLoss => format!("{:.8}", score),
            Self::ValidationAccuracy => format!("{:.4}", score),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EarlyStoppingConfig {
    pub metric: EarlyStoppingMetric,
    pub patience: usize,
    pub restore_best_weights: bool,
}

impl EarlyStoppingConfig {
    pub fn new(metric: EarlyStoppingMetric, patience: usize, restore_best_weights: bool) -> Self {
        Self {
            metric,
            patience,
            restore_best_weights,
        }
    }

    pub fn validation_loss(patience: usize) -> Self {
        Self::new(EarlyStoppingMetric::ValidationLoss, patience, true)
    }

    pub fn validation_accuracy(patience: usize) -> Self {
        Self::new(EarlyStoppingMetric::ValidationAccuracy, patience, true)
    }
}

pub struct NNConfig<F: Float> {
    pub epochs: usize,
    pub learning_rate: F,
    pub batch_size: usize,
    pub evaluate_step: usize,
    pub loss_function: Box<dyn Loss<F>>,
    pub optimizer: OptimizerConfig<F>,
    pub seed: Option<u64>,
    pub early_stopping: Option<EarlyStoppingConfig>,
}

impl<F: Float> NNConfig<F> {
    pub fn new(
        epochs: usize,
        learning_rate: F,
        batch_size: usize,
        evaluate_step: usize,
        loss_function: Box<dyn Loss<F>>,
        optimizer: OptimizerConfig<F>,
    ) -> Self {
        NNConfig {
            epochs,
            learning_rate,
            batch_size,
            evaluate_step,
            loss_function,
            optimizer,
            seed: None,
            early_stopping: None,
        }
    }
}

impl<F: Float> Default for NNConfig<F> {
    fn default() -> Self {
        NNConfig {
            epochs: 1,
            learning_rate: F::from_f32(0.5).unwrap(),
            batch_size: 1,
            evaluate_step: 10,
            loss_function: HalfMSE::new().into(),
            optimizer: OptimizerConfig::SGD,
            seed: None,
            early_stopping: None,
        }
    }
}
