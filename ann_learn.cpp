
#include "stdafx.hpp"

#include "annlearn/neuron.hpp"
#include "annlearn/layer.hpp"
#include "annlearn/backprop_net.hpp"

namespace ann = annlearn;

template<typename V>
void print(V& vs)
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

int main()
{
	vex::Context ctx(vex::Filter::GPU && vex::Filter::Position{1});
	vex::profiler<> prof(ctx);

	using vex::_;

	if (!ctx) throw std::runtime_error("No devices available.");
	std::cout << ctx << std::endl;

	vex::Random<float, vex::random::threefry> rnd;

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

	ann::backprop_net<float> net;
	
	net.reset(ctx, std::vector<size_t>{layer1, layer2, layer3});
	net.random_initialise();
	
	prof.toc("initialise");

	std::vector<float> ip{
		0.f, 0.f, 0.f,
		-0.2f, -0.2f, 1.f,
		0.7f, 0.2f, 1.f,
		-0.3f, -0.5f, 1.f,
		-0.7f, 0.4f, 1.f};

	vex::vector<float> iv{ctx, ip};

	std::vector<float> tp{
		-0.2f, 0.6f,
		-0.7f, -0.6f,
		0.3f, -0.9f,
		-0.1f, -0.2f,
		0.4f, 0.2f};

	vex::vector<float> tv{ctx, tp};

	std::vector<size_t> indices(ip.size() / layer1);
	std::iota(indices.begin(), indices.end(), 0);

	vex::slicer<2> is(vex::extents[ip.size() / layer1][layer1]);
	vex::slicer<2> ts(vex::extents[tp.size() / layer3][layer3]);
	
	prof.tic_cl("train");

	for (int i = 0; i < 4000; ++i)
	{
		std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

		for (size_t j : indices)
		{
			net.forward_pass(is[j](iv));
			net.backward_pass(0.05f, is[j](iv), ts[j](tv));
		}
	}

	prof.toc("train");
	
	for (size_t j : indices)
		print(net.forward_pass(is[j](iv)));

#endif
	std::cout << prof << std::endl;

	system("pause");
	return 0; 

}
