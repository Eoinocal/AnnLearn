
#include "stdafx.hpp"

#define BOOST_TEST_MODULE Prod
#include <boost/test/unit_test.hpp>

#include <annlearn/print.hpp>
#include <annlearn/prod.hpp>
#include <annlearn/vex_matrix.hpp>

BOOST_AUTO_TEST_CASE(mat_mat_prod)
{
	const vex::Context& ctx = vex::current_context();

	vex::Reductor<double, vex::SUM> sum(ctx);

	vex::vector<double> x{std::vector<double>{1., 2., 3., 4., 5.}};

	annlearn::matrix<double> A(ctx, 3, 5, std::vector<double>{
		0.f, 0.f, 0.f,
		-0.2f, -0.2f, 1.f,
		0.7f, 0.2f, 1.f,
		-0.3f, -0.5f, 1.f,
		-0.7f, 0.4f, 1.f});

	vex::vector<double> b = {std::vector<double>{-3., 0.2, 14.}};

	BOOST_CHECK_SMALL(
		sum(fabs(
				annlearn::prod(x, A) - b
			)
		), 1e-6);
}
