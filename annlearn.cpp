
#include "stdafx.hpp"

#include "annlearn/neuron.hpp"
#include "annlearn/layer.hpp"
#include "annlearn/backprop_net.hpp"
#include "annlearn/vex_matrix.hpp"
#include "annlearn/print.hpp"
#include "annlearn/vex_vector_io.hpp"

namespace ann = annlearn;


template<typename Expr>
inline
double l2_norm(const Expr& v)
{
	typedef vex::traits::value_type<Expr>::type T;
	vex::Reductor<double, vex::SUM> sum(vex::current_context());

	return round(sum(v*v));
}

int main()
{
	vex::Context ctx(vex::Filter::CPU && vex::Filter::Position{0});
	vex::profiler<> prof(ctx);

	using vex::_;

	if (!ctx) throw std::runtime_error("No devices available.");
	std::cout << ctx << std::endl;

	vex::Random<double, vex::random::threefry> rnd;

#if 1

	ann::matrix<double> x_train;
	ann::matrix<double> y_train;

	{	std::ifstream ifs("iris.txt");
		boost::archive::xml_iarchive ia(ifs);
		ia >> ann::make_nvp("x_train", x_train);
		ia >> ann::make_nvp("y_train", y_train);
	}

	std::cout << "Inputs" << std::endl;

//	ann::print(x_train);
	ann::print(y_train);

	ann::backprop_net<double> net{ctx, {x_train.ncol(), 200, y_train.ncol()}};
	net.random_initialise();

	std::cout << "Pre training prediction error: " << l2_norm(net.predict(x_train).data - y_train.data) << std::endl;

//	ann::print(net.predict(x_train));
//	std::cout << "Error: " << l2_norm(net.predict(x_train).data - y_train.data) << std::endl;

	net.fit(x_train, y_train, 0.1, 2000);

	vex::vector<double> predictions = vex::round(net.predict(x_train).data);
//	ann::print(predictions);
	std::cout << "Post training prediction error: " << l2_norm(predictions - y_train.data) << std::endl;

//	std::cout << "Error: " << l2_norm(predictions.data - y_train.data) << std::endl;

#elif 0
	size_t in = 1 << 10;
	size_t out = 1 << 12;

	ann::backprop_net<float> net(ctx, std::vector<size_t>{in, 1 << 14, out});


	vex::vector<float> input(ctx, in);
	input = 2 * rnd(vex::element_index(), std::rand()) - 1;
	vex::vector<float> target(ctx, out);
	target = 2 * rnd(vex::element_index(), std::rand()) - 1;

	prof.tic_cl("fwd warmup");

	for (int i = 0; i < 10; ++i)
		net.forward_pass(input);

	prof.toc("fwd warmup");

	prof.tic_cl("fwd");

	for (int i = 0; i < 100; ++i)
		net.forward_pass(input);

	prof.toc("fwd");

	prof.tic_cl("train warmup");

	for (int i = 0; i < 10; ++i)
	{
		net.forward_pass(input);
		net.backward_pass(0.01f, input, target);
	}

	prof.toc("train warmup");

	prof.tic_cl("train");

	for (int i = 0; i < 100; ++i)
	{
		net.forward_pass(input);
		net.backward_pass(0.01f, input, target);
	}

	prof.toc("train");
#elif 0


	size_t layer1 = 3; // 1 << 8;
	size_t layer2 = 5; // 1 << 10;
	size_t layer3 = 2; // 1 << 6;

	prof.tic_cl("initialise");

	ann::backprop_net<double> net;

	net.reset(ctx, std::vector<size_t>{layer1, layer2, layer3});
	net.random_initialise();

	prof.toc("initialise");

	vex::vector<double> vv{ctx, std::vector<double>{1., 2., 3., 4., 5.}};

	{	std::ofstream ofs("vex_vec.txt");
		boost::archive::text_oarchive oa(ofs);
		oa << vv;
	}

	vex::vector<double> v;

	{	std::ifstream ifs("vex_vec.txt");
		boost::archive::text_iarchive ia(ifs);
		ia >> v;
	}
	
	ann::matrix<double> input(ctx, layer1, 5, std::vector<double>{
		0.f, 0.f, 0.f,
		-0.2f, -0.2f, 1.f,
		0.7f, 0.2f, 1.f,
		-0.3f, -0.5f, 1.f,
		-0.7f, 0.4f, 1.f});

	ann::print(net.predict(input));

	ann::print(input);

	vex::vector<double> ans = prod(v, input);

	ann::print(ans);

	ann::matrix<double> V(5, 1, v);

	ann::matrix<double> ans3 = prod(V, input);
	ann::print(ans3);


	vex::vector<double> u{ctx, std::vector<double>{1., 2., 3.}};

	vex::vector<double> ans2 = prod(input, u);
	ann::print(ans2);

	ann::matrix<double> U(1, 3, u);

	ann::matrix<double> ans4 = prod(input, U);
	ann::print(ans4);

	ann::matrix<double> ans5 = prod(U, V);
	ann::print(ans5);

	ann::matrix<double> ans6 = ann::outer_product(u, v);
	ann::print(ans6);


	ann::matrix<double> target(ctx, layer3, 5, std::vector<double>{
		-0.2f, 0.6f,
		-0.7f, -0.6f,
		0.3f, -0.9f,
		-0.1f, -0.2f,
		0.4f, 0.2f});
	
	std::vector<size_t> indices(input.size() / layer1);
	std::iota(indices.begin(), indices.end(), 0);

	ann::matrix<double> empty(ctx, 3, 5);
	empty.row(2) = u;
	empty.column(1) = v;
	ann::print(empty);
	
	prof.tic_cl("train");

	for (int i = 0; i < 4000; ++i)
	{
		std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

		for (size_t j : indices)
		{
			net.forward_pass(input.row(j));
			net.backward_pass(0.05f, input.row(j), target.row(j));
		}
	}

	prof.toc("train");

	ann::print(net.predict(input));

	{	std::ofstream ofs("net.txt");
		boost::archive::xml_oarchive oa(ofs);
		oa << ann::make_nvp("net", net);
	}

	ann::backprop_net<double> net_loaded;

	{	std::ifstream ifs("net.txt");
		boost::archive::xml_iarchive ia(ifs);
		ia >> ann::make_nvp("net", net_loaded);
	}

	ann::print(net_loaded.predict(input));

#endif
	std::cout << prof << std::endl;

	system("pause");
	return 0; 
}
