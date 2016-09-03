
#include "stdafx.hpp"

#define BOOST_TEST_MODULE Prod
#include <boost/test/unit_test.hpp>

#include <annlearn/print.hpp>
#include <annlearn/prod.hpp>
#include <annlearn/vex_matrix.hpp>

BOOST_AUTO_TEST_CASE(vec_mat_prod)
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


VEX_FUNCTION(float, vec_mat_prod2, (size_t, n)(size_t, m)(size_t, j)(double*, v)(double*, w),

	double sum = 0.0;
	
	for (size_t i = 0; i < m; ++i)
	{
		sum += v[i] * w[i*n + j];
	}

	return sum;
);

BOOST_AUTO_TEST_CASE(vec_mat_prod_manual)
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

	vex::vector<double> y{ctx, 3};

	y = vec_mat_prod2(3, 5, vex::element_index(), vex::raw_pointer(x), vex::raw_pointer(A.data));

	vex::vector<double> b = {std::vector<double>{-3., 0.2, 14.}};

	BOOST_CHECK_SMALL(
		sum(fabs(
			y - b
		)
		), 1e-6);
}

BOOST_AUTO_TEST_CASE(vex_vector_prod_vex_vector_as_matrix)
{
	const vex::Context& ctx = vex::current_context();

	vex::Reductor<double, vex::SUM> sum(ctx);

	vex::vector<double> x{std::vector<double>{1., 2., 3., 4., 5.}};

	vex::vector<double> A(std::vector<double>{
		0.f, 0.f, 0.f,
			-0.2f, -0.2f, 1.f,
			0.7f, 0.2f, 1.f,
			-0.3f, -0.5f, 1.f,
			-0.7f, 0.4f, 1.f});

	vex::vector<double> y{ctx, 3};

	y = annlearn::vec_mat_prod(x, A);

	vex::vector<double> b = {std::vector<double>{-3., 0.2, 14.}};

	BOOST_CHECK_SMALL(
		sum(fabs(
			y - b
		)
		), 1e-6);
}

BOOST_AUTO_TEST_CASE(vex_vector_prod_ann_matrix)
{
	const vex::Context& ctx = vex::current_context();

	vex::Reductor<double, vex::SUM> sum(ctx);

	vex::vector<double> x{std::vector<double>{1., 2., 3., 4., 5.}};

	vex::vector<double> A(std::vector<double>{
		0.f, 0.f, 0.f,
		-0.2f, -0.2f, 1.f,
		0.7f, 0.2f, 1.f,
		-0.3f, -0.5f, 1.f,
		-0.7f, 0.4f, 1.f});

	annlearn::matrix<double> B{3, 5, A};

	vex::vector<double> y{ctx, 3};

	y = annlearn::prod(x, B);

	vex::vector<double> b = {std::vector<double>{-3., 0.2, 14.}};

	BOOST_CHECK_SMALL(
		sum(fabs(
			y - b
		)
		), 1e-6);
}
