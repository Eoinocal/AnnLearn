
#include "stdafx.hpp"

#define BOOST_TEST_MODULE Matrix
#include <boost/test/unit_test.hpp>

#include <annlearn/print.hpp>
#include <annlearn/prod.hpp>
#include <annlearn/vex_matrix.hpp>

using annlearn::matrix;

BOOST_AUTO_TEST_CASE(matrix_row)
{
	const vex::Context& ctx = vex::current_context();
	vex::Reductor<double, vex::SUM> sum(ctx);

	matrix<double> X(2, 3, std::vector<double>{1., 2., 3., 4., 5., 6.});
	vex::vector<double> v{std::vector<double>{7., 8.}};

	X.row(1) = v;

	annlearn::matrix<double> Y(2, 3, std::vector<double>{1., 2., 7., 8., 5., 6.});

	BOOST_CHECK_SMALL(
		sum(fabs(
			X.data - Y.data
		)
		), 1e-6);
}

BOOST_AUTO_TEST_CASE(matrix_col)
{
	const vex::Context& ctx = vex::current_context();
	vex::Reductor<double, vex::SUM> sum(ctx);

	matrix<double> X(2, 3, std::vector<double>{1., 2., 3., 4., 5., 6.});
	vex::vector<double> u{std::vector<double>{7., 8., 9.}};

	X.column(1) = u;

	annlearn::matrix<double> Y(2, 3, std::vector<double>{1., 7., 3., 8., 5., 9.});

	BOOST_CHECK_SMALL(
		sum(fabs(
			X.data - Y.data
		)
		), 1e-6);
}
