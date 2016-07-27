
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

	auto net = annlearn::backprop_net<double, annlearn::fc_layer<double>>::load(std::ifstream{"hyper_tan_backprop_net.xml"});
	auto X{annlearn::load_matrix<double>(std::ifstream{"blobs_X.xml"})};
	auto target = annlearn::load_vector<double>(std::ifstream{"hyper_tan_layer1_target.xml"});

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

	auto net = annlearn::backprop_net<double, annlearn::fc_layer<double>>::load(std::ifstream{"hyper_tan_backprop_net.xml"});
	auto X{annlearn::load_matrix<double>(std::ifstream{"blobs_X.xml"})};
	auto target = annlearn::load_vector<double>(std::ifstream{"hyper_tan_layer2_target.xml"});

	vex::vector<double> out = net.forward_pass(X.row(0));

	BOOST_CHECK_SMALL(
		sum(fabs(
			out - target
		)
		), 1e-6);
}

BOOST_AUTO_TEST_CASE(layer_delta)
{
	vex::Reductor<double, vex::SUM> sum(vex::current_context());

	auto net = annlearn::backprop_net<double, annlearn::fc_layer<double>>::load(std::ifstream{"hyper_tan_backprop_net.xml"});
	auto X{annlearn::load_matrix<double>(std::ifstream{"blobs_X.xml"})};
	auto y{annlearn::load_matrix<double>(std::ifstream{"blobs_y_hot.xml"})};
	auto target = annlearn::load_vector<double>(std::ifstream{"hyper_tan_deltas2_target.xml"});

	vex::vector<double> out = net.forward_pass(X.row(0));

	net.get_layer(1).compute_deltas(y.row(0));

	BOOST_CHECK_SMALL(
		sum(fabs(
			net.get_layer(1).deltas - target
		)
		), 1e-6);
}

BOOST_AUTO_TEST_CASE(net_delta)
{
	vex::Reductor<double, vex::SUM> sum(vex::current_context());

	auto net = annlearn::backprop_net<double, annlearn::fc_layer<double>>::load(std::ifstream{"hyper_tan_backprop_net.xml"});
	auto X{annlearn::load_matrix<double>(std::ifstream{"blobs_X.xml"})};
	auto y{annlearn::load_matrix<double>(std::ifstream{"blobs_y_hot.xml"})};
	auto target = annlearn::load_vector<double>(std::ifstream{"hyper_tan_deltas1_target.xml"});

	vex::vector<double> out = net.forward_pass(X.row(0));

	net.backward_pass(1.0, X.row(0), y.row(0));

	BOOST_CHECK_SMALL(
		sum(fabs(
			net.get_layer(0).deltas - target
		)
		), 1e-6);
}

BOOST_AUTO_TEST_CASE(layer_weights)
{
	vex::Reductor<double, vex::SUM> sum(vex::current_context());

	auto net = annlearn::backprop_net<double, annlearn::fc_layer<double>>::load(std::ifstream{"hyper_tan_backprop_net.xml"});
	auto X{annlearn::load_matrix<double>(std::ifstream{"blobs_X.xml"})};
	auto y{annlearn::load_matrix<double>(std::ifstream{"blobs_y_hot.xml"})};
	auto target = annlearn::load_vector<double>(std::ifstream{"hyper_tan_weights1_target.xml"});

	vex::vector<double> out = net.forward_pass(X.row(0));

	net.get_layer(1).compute_deltas(y.row(0));
	net.get_layer(0).compute_deltas(net.get_layer(1));
	net.get_layer(0).update_weights(1.0, X.row(0));

	BOOST_CHECK_SMALL(
		sum(fabs(
			net.get_layer(0).weights - target
		)
		), 1e-6);
}

BOOST_AUTO_TEST_CASE(net_weights)
{
	vex::Reductor<double, vex::SUM> sum(vex::current_context());

	auto net = annlearn::backprop_net<double, annlearn::fc_layer<double>>::load(std::ifstream{"hyper_tan_backprop_net.xml"});
	auto X{annlearn::load_matrix<double>(std::ifstream{"blobs_X.xml"})};
	auto y{annlearn::load_matrix<double>(std::ifstream{"blobs_y_hot.xml"})};
	auto target = annlearn::load_vector<double>(std::ifstream{"hyper_tan_weights2_target.xml"});

	vex::vector<double> out = net.forward_pass(X.row(0));
	net.backward_pass(1.0, X.row(0), y.row(0));


	BOOST_CHECK_SMALL(
		sum(fabs(
			net.get_layer(1).weights - target
		)
		), 1e-6);
}
