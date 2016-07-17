
#include "stdafx.hpp"

#define BOOST_TEST_MODULE BackProp
#include <boost/test/unit_test.hpp>

#include <annlearn/print.hpp>
#include <annlearn/vex_vector_io.hpp>
#include <annlearn/backprop_net.hpp>

using annlearn::matrix;

BOOST_AUTO_TEST_CASE(layer_forward)
{
	vex::Reductor<double, vex::SUM> sum(vex::current_context());

	auto net = annlearn::backprop_net<double>::load(std::ifstream{"hyper_tan_backprop_net.xml"});
	auto X{annlearn::load_matrix<double>(std::ifstream{"blobs_X.xml"})};
	auto target = annlearn::load_vector<double>(std::ifstream{"hyper_tan_layer_target.xml"});

	net.get_layer(0).activate(X.row(0));

	BOOST_CHECK_SMALL(
		sum(fabs(
			net.get_layer(0).activation() - target
		)
		), 1e-6);
}

BOOST_AUTO_TEST_CASE(net_forward)
{
	vex::Reductor<double, vex::SUM> sum(vex::current_context());

	auto net = annlearn::backprop_net<double>::load(std::ifstream{"hyper_tan_backprop_net.xml"});
	auto X{annlearn::load_matrix<double>(std::ifstream{"blobs_X.xml"})};
	auto target = annlearn::load_vector<double>(std::ifstream{"hyper_tan_net_target.xml"});

	vex::vector<double> out = net.forward_pass(X.row(0));

	BOOST_CHECK_SMALL(
		sum(fabs(
			out - target
		)
		), 1e-6);
}
