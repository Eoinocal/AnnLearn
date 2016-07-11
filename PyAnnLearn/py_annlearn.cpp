
#include "py_pch.hpp"

#include <annlearn/print.hpp>

#include "convert.hpp"
#include "classifier/base.hpp"
#include "classifier/backprop_net.hpp"

namespace ann = annlearn;

enum filter { cpu = 1, gpu = 2 };

std::unique_ptr<vex::Context> ctx_;

void deinitialise_context()
{
	ctx_.reset();
}

void print_context()
{
	std::cout << "Using context(s):\n" << vex::current_context() << std::endl;
}

void initialise_context(filter f, int p)
{
	vex::Filter::Type filter =
		f == cpu ? vex::Filter::CPU :
		f == gpu ? vex::Filter::GPU : vex::Filter::CPU;

	ctx_ = std::make_unique<vex::Context>(filter && vex::Filter::Position{p});
	print_context();
}

np::ndarray inspect(np::ndarray x)
{
	std::cout << x.get_nd() << std::endl;
	return x;
}


np::ndarray prod(np::ndarray n, np::ndarray m)
{
	ann::matrix<double> o = ann::prod(to_ann_matrix<double>(n), to_ann_matrix<double>(m));
	ann::print(o);

	//m.astype()

	return to_ndarray(o);
}

void inspect2(const py::object& iterable)
{
//	std::vector<size_t> a(annlearn::to_std_vector<size_t>(iterable));

//	std::cout << a.size() << std::endl;
}

np::ndarray test_array_double(const np::ndarray m)
{
/*	vex::Reductor<double, vex::SUM> sum(ctx);

	ann::matrix<double> test{context(), 2, 3,
		std::vector<double>
		{	1., 2.,
			3., 4.,
			5., 6.  }};

	auto n = to_ann_matrix<double>(m);

//	annlearn::print(test);
	ann::print(n);

	return to_ndarray(n);
	*/


	Py_intptr_t shape[1] = {(Py_intptr_t)1};
	np::ndarray possibly_converted = np::zeros(1, shape, np::dtype::get_builtin<double>());
	return possibly_converted;
}


#if defined(NDEBUG)
#	define MODULE_NAME annlearn
#else
#	define MODULE_NAME annlearn_d
#endif

BOOST_PYTHON_MODULE(MODULE_NAME)
{
	np::initialize();

	py::enum_<filter>("filter")
		.value("cpu", cpu)
		.value("gpu", gpu);

	py::def("initialise_context", &initialise_context);
	py::def("deinitialise_context", &deinitialise_context);
	py::def("print_context", &print_context);

//	py::def("inspect", &inspect);
	py::def("inspect", &inspect);
//	py::def("prod", &prod);
	py::def("inspect2", &inspect2);

	py::def("save_nd_matrix", &save_nd_matrix<double>);
	py::def("load_nd_matrix", &load_nd_matrix<double>);

	py::class_<ann::py_backprop_net>("back_prop")
		.def("reset", &ann::py_backprop_net::reset)
		.def("fit", &ann::py_backprop_net::fit)
		.def("print_profile", &ann::py_backprop_net::print_profile)
		.def("predict", &ann::py_backprop_net::predict);
}
