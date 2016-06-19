
#include "stdafx.hpp"
#include "classifier/base.hpp"

void inspect(np::ndarray x)
{
	int ndim = x.get_nd();
	const Py_intptr_t* dim = x.get_shape();

	std::cout << "Dim: " << ndim << std::endl;

	for (int i = 0; i < ndim; ++i)
		std::cout << dim[i] << ", ";
	std::cout << std::endl;

	if (x.get_dtype() != np::dtype::get_builtin<double>())
	{
		std::cout << " !! wrong type" << std::endl;
		return;
	}
	
	double* data = reinterpret_cast<double*>(x.get_data());



	std::cout << data[1] << std::endl;
}

BOOST_PYTHON_MODULE(annlearn)
{
	np::initialize();

	bp::def("inspect", &inspect);

	bp::class_<annlearn::base>("base")
		.def("reset", &annlearn::base::reset);
}
