use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, IxDyn, Zip};

use super::Float;

/// Per-parameter optimizer instance.
pub trait Optimizer<F: Float>: Send {
    fn step(&mut self, params: ArrayViewMutD<F>, grads: ArrayViewD<F>, lr: F);
}

/// Network-level optimizer config.
#[derive(Clone, Copy, Default)]
pub enum OptimizerConfig<F: Float> {
    #[default]
    SGD,
    Adam {
        beta1: F,
        beta2: F,
        eps: F,
    },
}

impl<F: Float> OptimizerConfig<F> {
    pub fn adam_default() -> Self {
        Self::Adam {
            beta1: F::from_f32(0.9).unwrap(),
            beta2: F::from_f32(0.999).unwrap(),
            eps: F::from_f32(1e-8).unwrap(),
        }
    }

    pub fn build(&self) -> Box<dyn Optimizer<F>> {
        match *self {
            Self::SGD => Box::new(SGD),
            Self::Adam { beta1, beta2, eps } => Box::new(Adam::new(beta1, beta2, eps)),
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct SGD;

impl<F: Float> Optimizer<F> for SGD {
    fn step(&mut self, mut params: ArrayViewMutD<F>, grads: ArrayViewD<F>, lr: F) {
        Zip::from(&mut params)
            .and(&grads)
            .for_each(|p, &g| *p -= g * lr);
    }
}

pub struct Adam<F: Float> {
    beta1: F,
    beta2: F,
    eps: F,
    m: Option<ArrayD<F>>,
    v: Option<ArrayD<F>>,
    t: i32,
}

impl<F: Float> Adam<F> {
    pub fn new(beta1: F, beta2: F, eps: F) -> Self {
        Self {
            beta1,
            beta2,
            eps,
            m: None,
            v: None,
            t: 0,
        }
    }
}

impl<F: Float> Optimizer<F> for Adam<F> {
    fn step(&mut self, mut params: ArrayViewMutD<F>, grads: ArrayViewD<F>, lr: F) {
        if self.m.is_none() {
            self.m = Some(ArrayD::zeros(IxDyn(params.shape())));
            self.v = Some(ArrayD::zeros(IxDyn(params.shape())));
        }
        self.t += 1;

        let one = F::from_f32(1.0).unwrap();
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let one_minus_beta1 = one - beta1;
        let one_minus_beta2 = one - beta2;
        let bias_correction1 = one - beta1.powi(self.t);
        let bias_correction2 = one - beta2.powi(self.t);

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        Zip::from(&mut params)
            .and(&grads)
            .and(m)
            .and(v)
            .for_each(|p, &g, m_i, v_i| {
                *m_i = beta1 * *m_i + one_minus_beta1 * g;
                *v_i = beta2 * *v_i + one_minus_beta2 * g * g;
                let m_hat = *m_i / bias_correction1;
                let v_hat = *v_i / bias_correction2;
                *p -= lr * m_hat / (v_hat.sqrt() + eps);
            });
    }
}
