pub mod candle_inference;
pub mod candle_qnet;
pub mod env;
pub mod grid;
// pub mod mcts;

use pyo3::prelude::*;

use env::{PacmanGym, PacmanGymConfiguration};
// use mcts::MCTSContext;

/// A Python module containing Rust implementations of the PacBot environment.
#[pymodule]
fn pacbot_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PacmanGymConfiguration>()?;
    m.add_class::<PacmanGym>()?;
    // m.add_class::<MCTSContext>()?;
    Ok(())
}
