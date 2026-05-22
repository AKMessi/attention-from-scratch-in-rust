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

        scores.apply_causal_mask();

        let weights = scores.softmax_rows();

        let context = weights.matmul(&v);
        let final_output = context.matmul(&self.output);

        final_output
    }
}

pub struct MultiHeadAttention {
    pub heads: Vec<SingleHeadAttention>,
    pub output: Matrix,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads:usize) -> Self {
        assert_eq!(d_model % num_heads, 0, "d_model must be perfectly divisible by num_heads");
        let d_k = d_model / num_heads;
        let d_v = d_model / num_heads;

        let mut heads = Vec::new();
        for _ in 0..num_heads {
            heads.push(SingleHeadAttention::new(d_model, d_k, d_v));
        }

        Self {
            heads,
            output: Matrix::new(d_model, d_model),
        }
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        let seq_len = x.rows();
        let d_model = x.cols();

        let mut head_outputs = Vec::new();
        for head in &self.heads {
            head_outputs.push(head.forward(x));
        }

        let mut concatenated = Matrix::new(seq_len, d_model);
        let d_v = d_model / self.heads.len();

        for row in 0..seq_len {
            for head_idx in 0..self.heads.len() {
                for col in 0..d_v {
                    let val = head_outputs[head_idx].read(row, col);
                    let target_col = head_idx * d_v + col;
                    concatenated.modify(row, target_col, val);
                }
            }
        }
        concatenated.matmul(&self.output)
    }
}
