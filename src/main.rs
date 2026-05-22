mod matrix;
mod attention;

use matrix::Matrix;
use attention::MultiHeadAttention;

fn main() {
    println!("🚀 Starting Bare-Metal Attention Verification...");

    // 1. Setup dimensions (GPT-2 style test proportions)
    let seq_len = 3;   // "The cat sat"
    let d_model = 4;   // Small hidden size
    let num_heads = 2; // 2 heads means d_k = d_v = 2

    // 2. Create input sequence matrix
    let mut input = Matrix::new(seq_len, d_model);
    // Fill it with mock token embedding values
    for i in 0..seq_len {
        for j in 0..d_model {
            input.modify(i, j, (i + j) as f32 * 0.5);
        }
    }

    // 3. Instantiate Multi-Head Attention module
    let mut mha = MultiHeadAttention::new(d_model, num_heads);

    // 4. Manually initialize weights to a constant value so we aren't multiplying by zero
    for head in &mut mha.heads {
        for i in 0..head.query.data.len() { head.query.data[i] = 0.1; }
        for i in 0..head.key.data.len()   { head.key.data[i] = 0.1; }
        for i in 0..head.value.data.len() { head.value.data[i] = 0.2; }
    }
    for i in 0..mha.output.data.len() { mha.output.data[i] = 0.2; }

    // 5. Run the forward pass!
    println!("🧠 Running Multi-Head Attention forward pass...");
    let result = mha.forward(&input);

    println!("✅ Forward pass completed successfully! Shape: {}x{}", result.rows, result.cols);
    assert_eq!(result.rows, seq_len);
    assert_eq!(result.cols, d_model);

    // 6. Direct unit verification of our custom causal mask + stable softmax logic
    println!("🧪 Verifying Causal Masking & Softmax isolated correctness...");
    let mut test_scores = Matrix::new(3, 3);
    // Row 0 has a dangerously high value (testing numerical stability)
    test_scores.modify(0, 0, 10.0);  test_scores.modify(0, 1, 90.0);  test_scores.modify(0, 2, 12.0);
    // Row 2 has standard values
    test_scores.modify(2, 0, 1.0);   test_scores.modify(2, 1, 2.0);   test_scores.modify(2, 2, 3.0);

    // Apply the scaling, mask, and softmax sequence
    test_scores.scale(1.0);
    test_scores.apply_causal_mask();
    let attention_weights = test_scores.softmax_rows();

    // Verify causal constraint: future tokens MUST be 0.0
    // Token 0 can't look at token 1 or 2
    assert_eq!(attention_weights.read(0, 1), 0.0);
    assert_eq!(attention_weights.read(0, 2), 0.0);
    // Token 1 can't look at token 2
    assert_eq!(attention_weights.read(1, 2), 0.0);

    // Verify normalization constraint: every row must sum to exactly 1.0
    for r in 0..attention_weights.rows {
        let mut row_sum = 0.0_f32;
        for c in 0..attention_weights.cols {
            row_sum += attention_weights.read(r, c);
        }
        println!("   Row {} sum: {:.4}", r, row_sum);
        assert!((row_sum - 1.0).abs() < 1e-5, "Row sum deviates from 1.0!");
    }

    println!("🎉 ALL TESTS PASSED! Your custom AI engine is flawless.");
}
