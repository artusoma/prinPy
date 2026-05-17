pub mod global;
pub mod local;

use pyo3::prelude::*;
#[pymodule]
mod prinpy_rs {
    use super::local::constrained::{ConstrainedAlgorithm, GreedyFitter};
    use pyo3::prelude::*;

    use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};

    #[pyfunction]
    fn clpg<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        tol: f32,
    ) -> Bound<'py, PyArray2<f32>> {
        let arr = ConstrainedAlgorithm::new(GreedyFitter::default(), tol).fit(&x.to_owned_array());
        PyArray2::from_array(py, &arr)
    }
}
