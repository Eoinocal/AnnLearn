
#pragma once

#include "stdafx.hpp"
#include <annlearn/vex_matrix.hpp>

template<typename T>
inline
std::vector<T> to_std_vector(const py::object& iterable)
{
	return std::vector<T>(py::stl_input_iterator<T>(iterable), py::stl_input_iterator<T>());
}

template<typename T>
inline
vex::vector<T> to_vex_vector(const np::ndarray& a)
{
	assert(a.get_nd() == 1);

	const Py_intptr_t* dim = a.get_shape();

	std::cout << a.get_nd() << " -> " << dim[0] << std::endl;

//	auto total = static_cast<size_t>(std::accumulate(&dim[0], &dim[a.get_nd()], (Py_intptr_t)1, std::multiplies<Py_intptr_t>()));

//	std::cout << total << std::endl;

	return vex::vector<T>{context(), static_cast<size_t>(a.get_shape()[0]), reinterpret_cast<T*>(a.get_data())};
}

template<typename T>
inline
annlearn::matrix<T> to_ann_matrix(const np::ndarray& a_t)
{

	if (a_t.get_nd() == 1)
	{
		const Py_intptr_t* dim = a_t.get_shape();

		std::cout << a_t.get_nd() << " -> " << dim[0] << std::endl;

		return annlearn::matrix<T>{1, static_cast<size_t>(dim[0]), vex::vector<T>{context(), static_cast<size_t>(dim[0]), reinterpret_cast<T*>(a_t.get_data())}};
	}
	else if (a_t.get_nd() == 2) 
	{
		auto a = a_t.transpose();
		const Py_intptr_t* dim = a.get_shape();

		std::cout << a.get_nd() << " -> ";
		for (int d = 0; d < a.get_nd(); ++d)
			std::cout << dim[d] << " ";

		auto total = static_cast<size_t>(std::accumulate(&dim[0], &dim[a.get_nd()], (Py_intptr_t)1, std::multiplies<Py_intptr_t>()));

		std::cout << "= " << total << std::endl;

		return annlearn::matrix<T>{static_cast<size_t>(dim[0]), static_cast<size_t>(dim[1]), vex::vector<T>{context(), total, reinterpret_cast<T*>(a.get_data())}};
	}
	else
		throw std::runtime_error("Unsupport ndarray dimension > 2");
}

template<typename T>
inline
np::ndarray to_ndarray(const annlearn::matrix<T>& m)
{
	std::vector<double> o(m.data.size());
	vex::copy(m.data.begin(), m.data.end(), o.begin());

	Py_intptr_t shape[2] = {(Py_intptr_t)m.nrow(), (Py_intptr_t)m.ncol()};
	np::ndarray result = np::zeros(2, shape, np::dtype::get_builtin<double>());
	std::copy(o.begin(), o.end(), reinterpret_cast<double*>(result.get_data()));

	return result;
}

