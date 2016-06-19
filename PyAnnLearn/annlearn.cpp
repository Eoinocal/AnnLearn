
#include "stdafx.hpp"
#include "classifier/base.hpp"

BOOST_PYTHON_MODULE(annlearn)
{
	np::initialize();

	bp::class_<annlearn::base>("base")
		.def("reset", &annlearn::base::reset);
}
