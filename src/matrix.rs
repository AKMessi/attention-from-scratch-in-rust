pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
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
}
