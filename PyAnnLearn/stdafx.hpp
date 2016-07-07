
#pragma once


#include <numeric>
#include <random>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <boost/progress.hpp>

#include <boost/optional.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/detail/vector_assign.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/operations.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#pragma warning (disable : 4003)

#pragma warning (push)
//#define VEXCL_SHOW_KERNELS
#pragma warning (disable : 4996)
#	include <vexcl/vexcl.hpp>
#	include <vexcl/reductor.hpp>
#	include <vexcl/vector.hpp>
#	include <vexcl/tensordot.hpp>
#pragma warning (pop)

#pragma warning (push)
#pragma warning (disable : 4244)
#	include <boost/python.hpp>
#	define BOOST_LIB_NAME boost_numpy
#	include <boost/config/auto_link.hpp>
#	include <boost/numpy.hpp>
#pragma warning (push)

namespace py = boost::python;
namespace np = boost::numpy;

//const std::vector<vex::backend::command_queue>& context();
