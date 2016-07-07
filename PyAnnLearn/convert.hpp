
#pragma once

#include "stdafx.hpp"
#include <annlearn/vex_matrix.hpp>

#include <numpy/ndarraytypes.h>
#include <numpy/__multiarray_api.h>

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

//	std::cout << a.get_nd() << " -> " << dim[0] << std::endl;
//	auto total = static_cast<size_t>(std::accumulate(&dim[0], &dim[a.get_nd()], (Py_intptr_t)1, std::multiplies<Py_intptr_t>()));
//	std::cout << total << std::endl;

	return vex::vector<T>{context(), static_cast<size_t>(a.get_shape()[0]), reinterpret_cast<T*>(a.get_data())};
}

template<typename T>
inline
const np::ndarray* correct_array(const np::ndarray& in, np::ndarray& out)
{
//	std::cout << "Correcting: with flags: " << in.get_flags() << std::endl;
	auto type = in.get_dtype();
	const np::ndarray* current = &in;

	if (np::dtype::get_builtin<T>() != type)
	{
//		std::cout << " >> converting type\n";
		out = current->astype(np::dtype::get_builtin<T>());
		current = &out;
	}

	if (!(in.get_flags() & np::ndarray::bitflag::C_CONTIGUOUS))
	{
//		std::cout << " >> re-ordering\n";
		out = np::ndarray(boost::python::detail::new_reference(PyObject_CallMethod(current->ptr(), const_cast<char*>("copy"), "(i)", NPY_CORDER)));
		current = &out;
	}

	return current;
}

template<typename T>
inline
annlearn::matrix<T> to_ann_matrix(const np::ndarray& m)
{
	Py_intptr_t shape[1] = {(Py_intptr_t)1};
	np::ndarray possibly_converted = np::empty(1, shape, np::dtype::get_builtin<T>());
	const np::ndarray* active = correct_array<T>(m, possibly_converted);

	if (active->get_nd() == 1)
	{
		const Py_intptr_t* dim = active->get_shape();
//		std::cout << active->get_nd() << " -> " << dim[0] << std::endl;

		return annlearn::matrix<T>{1, static_cast<size_t>(dim[0]), vex::vector<T>{static_cast<size_t>(dim[0]), reinterpret_cast<T*>(active->get_data())}};
	}
	else if (active->get_nd() == 2) 
	{
		const Py_intptr_t* dim = active->get_shape();
/*		std::cout << m.get_nd() << " -> ";
		for (int d = 0; d < active->get_nd(); ++d)
			std::cout << dim[d] << " ";
*/
		auto total = static_cast<size_t>(std::accumulate(&dim[0], &dim[active->get_nd()], (Py_intptr_t)1, std::multiplies<Py_intptr_t>()));
//		std::cout << "= " << total << std::endl;

		return annlearn::matrix<T>{static_cast<size_t>(dim[1]), static_cast<size_t>(dim[0]), vex::vector<T>{total, reinterpret_cast<T*>(active->get_data())}};
	}
	else
		throw std::runtime_error("Unsupported ndarray dimension > 2");
}

template<typename T>
inline
np::ndarray to_ndarray(const annlearn::matrix<T>& m)
{
	std::vector<double> o(m.size());
	vex::copy(m.data.begin(), m.data.end(), o.begin());

	Py_intptr_t shape[2] = {(Py_intptr_t)m.nrow(), (Py_intptr_t)m.ncol()};
	np::ndarray result = np::zeros(2, shape, np::dtype::get_builtin<double>());
	std::copy(o.begin(), o.end(), reinterpret_cast<double*>(result.get_data()));

	return result;
}
template<typename T>
inline
void save_nd_matrix(const np::ndarray& nd_m, const std::string& filename)
{
	ann::matrix<T> m{to_ann_matrix<T>(nd_m)};

	{	std::ofstream ofs(filename);
		boost::archive::xml_oarchive oa(ofs);
		oa << ann::make_nvp("m", m);
	}
}

template<typename T>
inline
np::ndarray load_nd_matrix(const std::string& filename)
{
	ann::matrix<T> m;
	
	{	std::ifstream ifs(filename);
		boost::archive::xml_iarchive ia(ifs);
		ia >> ann::make_nvp("m", m);
	}

	return to_ndarray(m);
}
