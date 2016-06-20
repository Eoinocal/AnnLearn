
#include "stdafx.hpp"

#include "annlearn/neuron.hpp"
#include "annlearn/layer.hpp"
#include "annlearn/backprop_net.hpp"
#include "annlearn/vex_matrix.hpp"
#include "annlearn/print.hpp"

namespace ann = annlearn;


int main()
{
	vex::Context ctx(vex::Filter::GPU && vex::Filter::Position{1});
	vex::profiler<> prof(ctx);

	using vex::_;

	if (!ctx) throw std::runtime_error("No devices available.");
	std::cout << ctx << std::endl;

	vex::Random<double, vex::random::threefry> rnd;

#if 0
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
#else
	size_t layer1 = 3; // 1 << 8;
	size_t layer2 = 5; // 1 << 10;
	size_t layer3 = 2; // 1 << 6;

	prof.tic_cl("initialise");

	ann::backprop_net<double> net;

	net.reset(ctx, std::vector<size_t>{layer1, layer2, layer3});
	net.random_initialise();

	prof.toc("initialise");

	vex::vector<double> v{ctx, std::vector<double>{1., 2., 3., 4., 5.}};

	ann::matrix<double> input(ctx, layer1, 5, std::vector<double>{
		0.f, 0.f, 0.f,
		-0.2f, -0.2f, 1.f,
		0.7f, 0.2f, 1.f,
		-0.3f, -0.5f, 1.f,
		-0.7f, 0.4f, 1.f});

	ann::print(input);

	vex::vector<double> ans = prod(v, input);

	ann::print(ans);

	ans = prod(v, input);

	ann::print(ans);

	vex::vector<double> u{ctx, std::vector<double>{1., 2., 3.}};

	vex::vector<double> ans2 = prod(input, u);

	ann::print(ans2);

	ann::matrix<double> target(ctx, layer3, 5, std::vector<double>{
		-0.2f, 0.6f,
		-0.7f, -0.6f,
		0.3f, -0.9f,
		-0.1f, -0.2f,
		0.4f, 0.2f});
	
	std::vector<size_t> indices(input.size() / layer1);
	std::iota(indices.begin(), indices.end(), 0);
	
	prof.tic_cl("train");

	for (int i = 0; i < 400; ++i)
	{
		std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

		for (size_t j : indices)
		{
			net.forward_pass(input.row(j));
			net.backward_pass(0.05f, input.row(j), target.row(j));
		}
	}

	prof.toc("train");
	
	for (size_t j : indices)
		ann::print(net.forward_pass(input.row(j)));

#endif
	std::cout << prof << std::endl;

	system("pause");
	return 0; 
}
