pub mod global;
pub mod local;
pub mod utilities;

use pyo3::prelude::*;

macro_rules! to_pyerr {
    ($err:ident) => {
        PyRuntimeError::new_err(format!("Rust Runtime Error: {}", $err))
    };
}

#[pymodule]
mod prinpy_rs {
    use super::local::{ConstrainedFitIterator, GreedyFitter, SVDFitter};
    use super::utilities::find_nearest_candidates;
    use pyo3::prelude::*;

    use ndarray::Array1;
    use numpy::{PyArray2, PyReadonlyArray2};
    use pyo3::exceptions::PyRuntimeError;

    #[pyfunction]
    fn clpg<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        tol: f32,
        inner_radius: f32,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let array_view = x.as_array();
        let iterator =
            ConstrainedFitIterator::new(&array_view, tol, GreedyFitter::new(inner_radius));

        // Use ? to propagate errors directly to Python
        let fit_points: Vec<Array1<f32>> = iterator
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| to_pyerr!(e))?;

        let view_points: Vec<_> = fit_points.iter().map(|v| v.view()).collect();

        let result_array =
            ndarray::stack(ndarray::Axis(0), &view_points).map_err(|e| to_pyerr!(e))?;

        Ok(PyArray2::from_array(py, &result_array))
    }

    #[pyfunction]
    fn clppca<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        tol: f32,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let array_view = x.as_array();
        let iterator = ConstrainedFitIterator::new(&array_view, tol, SVDFitter::default());

        // Use ? to propagate errors directly to Python
        let fit_points: Vec<Array1<f32>> =
            iterator.collect::<Result<Vec<_>, _>>().map_err(|e| {
                // This string becomes the exception message in Python
                PyRuntimeError::new_err(format!("Rust ConstrainedFit error: {:?}", e))
            })?;

        let view_points: Vec<_> = fit_points.iter().map(|v| v.view()).collect();

        let result_array = ndarray::stack(ndarray::Axis(0), &view_points)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to stack results: {}", e)))?;

        Ok(PyArray2::from_array(py, &result_array))
    }

    #[pyfunction]
    fn find_nearest_points<'py>(
        data: PyReadonlyArray2<'py, f32>,
        curve: PyReadonlyArray2<'py, f32>,
    ) -> Vec<usize> {
        let data = data.as_array();
        let curve = curve.as_array();
        find_nearest_candidates(&data, &curve)
    }
}
