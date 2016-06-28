
#include "stdafx.hpp"
#include "convert.hpp"
#include "classifier/base.hpp"
#include "classifier/back_prop.hpp"


template<typename V>
void print(const V& vs)
{
	for (typename V::value_type v : vs)
		std::cout << (boost::format("%8.5f ") % v);

	std::cout << std::endl << std::endl;
}

template<typename T>
void print(vex::vector<T>& vs)
{
	std::vector<T> us(vs.size());
	vex::copy(vs.begin(), vs.end(), us.begin());
	print(us);
}

template<typename T>
void print(const annlearn::matrix<T>& vs)
{
	std::vector<T> us(vs.data.size());
	vex::copy(vs.data.begin(), vs.data.end(), us.begin());

	for (int r = 0; r < vs.nrow(); ++r)
	{
		for (size_t c = 0; c < vs.ncol(); ++c)
			std::cout << (boost::format("%8.5f ") % us[r* vs.ncol() + c]);

		std::cout << std::endl;
	}

	std::cout << std::endl;
}



void inspect(np::ndarray x)
{
	std::cout << x.get_nd() << std::endl;

	if (x.get_nd() == 1)
	{
		vex::vector<double> v = to_vex_vector<double>(x);
		annlearn::print(v);
	}
	else if (x.get_nd() == 2)
	{
		annlearn::matrix<double> m = to_ann_matrix<double>(x);
		annlearn::print(m);
	}
	else
		std::cout << "Invalid dimension" << std::endl;
}

np::ndarray prod(np::ndarray n, np::ndarray m)
{
	annlearn::matrix<double> o = annlearn::prod(to_ann_matrix<double>(n), to_ann_matrix<double>(m));
	annlearn::print(o);

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
	auto ctx = context();
	vex::Reductor<double, vex::SUM> sum(ctx);

	annlearn::matrix<double> test{context(), 2, 3,
		std::vector<double>
		{	1., 2.,
			3., 4.,
			5., 6.  }};

	auto n = to_ann_matrix<double>(m);

//	annlearn::print(test);
	annlearn::print(n);

	return to_ndarray(n);
}


#if defined(NDEBUG)
#	define MODULE_NAME annlearn
#else
#	define MODULE_NAME annlearn_d
#endif

BOOST_PYTHON_MODULE(MODULE_NAME)
{
	np::initialize();

//	ctx_.reset((std::vector<vex::backend::command_queue>)vex::Context(vex::Filter::GPU && vex::Filter::Position{1}));
//	if (!*ctx_) throw std::runtime_error("No devices available.");
//	std::cout << *ctx_ << std::endl;

	py::def("test_array_double", &test_array_double);
//	py::def("inspect", &inspect);
	py::def("inspect", &inspect);
	py::def("prod", &prod);

	py::def("inspect2", &inspect2);

	py::class_<annlearn::back_prop>("back_prop")
		.def("reset", &annlearn::back_prop::reset)
		.def("fit", &annlearn::back_prop::fit)
		.def("print_profile", &annlearn::back_prop::print_profile)
		.def("predict", &annlearn::back_prop::predict);
}
