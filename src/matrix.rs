pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn read(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    #[inline]
    pub fn modify(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    #[inline]
    pub fn print(&self) {
        for row in 0..self.rows {
            for col in 0..self.cols{
                print!("{}", self.read(row, col));
                print!(" ");
            }
            println!();
        }
    }

    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Matrix dimensions mismatch for matmul!");
        let mut result = Matrix::new(self.rows, other.cols);
        for row in 0..self.rows {
            for col in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.read(row, k) * other.read(k, col);
                }
                result.modify(row, col, sum);
            }
        }
        result
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                result.modify(col, row, self.read(row, col));
            }
        }
        result
    }

    pub fn softmax_rows_naive(&self) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for row in 0..self.rows {
            let mut sum = 0.0;
            for k in 0..self.cols {
                sum += self.read(row, k).exp();
            }
            for col in 0..self.cols {
                result.modify(row, col, self.read(row, col).exp() / sum);
            }
        }
        result
    }

    pub fn softmax_rows(&self) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for row in 0..self.rows {
            let mut max = self.read(row, 0);
            for col in 1..self.cols {
                max = max.max(self.read(row, col))
            }

            let mut sum = 0.0;
            for k in 0..self.cols {
                sum += (self.read(row, k) - max).exp();
            }

            for col in 0..self.cols {
                result.modify(row, col, (self.read(row, col) - max).exp() / sum);
            }
        }
        result
    }

    pub fn scale(&mut self, scalar: f32) {
        for i in 0..self.data.len() {
            self.data[i] *= scalar;
        }
    }

    pub fn apply_causal_mask(&mut self) {
        for row in 0..self.rows {
            for col in (row + 1)..self.cols {
                self.modify(row, col, -1e9);
            }
        }
    }
}
