use crate::matrix::Matrix;

pub struct SingleHeadAttention {
    pub query: Matrix,
    pub key: Matrix,
    pub value: Matrix,
    pub output: Matrix,
    pub d_k: usize,
}

impl SingleHeadAttention {
    pub fn new(d_model: usize, d_k: usize, d_v: usize) -> Self {
        Self {
            query: Matrix::new(d_model, d_k),
            key: Matrix::new(d_model, d_k),
            value: Matrix::new(d_model, d_v),
            output: Matrix::new(d_v, d_model),
            d_k,
        }
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        let q = x.matmul(&self.query);
        let k = x.matmul(&self.key);
        let v = x.matmul(&self.value);

        let k_t = k.transpose();
        let mut scores = q.matmul(&k_t);

        let scale_factor = 1.0 / (self.d_k as f32).sqrt();
        scores.scale(scale_factor);

        let weights = scores.softmax_rows();

        let context = weights.matmul(&v);
        let final_output = context.matmul(&self.output);

        final_output
    }
}
