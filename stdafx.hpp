
#pragma once

#include <stdio.h>
#include <tchar.h>

#include <numeric>
#include <random>

#include <boost/progress.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#pragma warning (disable : 4003)

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/detail/vector_assign.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/operations.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#pragma warning (push)
#define VEXCL_SHOW_KERNELS
#pragma warning (disable : 4996)
#include <vexcl/vexcl.hpp>
#include <vexcl/reductor.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/tensordot.hpp>
#pragma warning (pop)

namespace ub = boost::numeric::ublas;
