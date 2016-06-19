
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

VEX_CONSTANT(one, 1.f);

VEX_FUNCTION(float, sigmoid, (float, x),
	return 1 / (1 + exp(-x));
);

int main()
{
	vex::Context ctx(vex::Filter::GPU && vex::Filter::Position{0});
	vex::profiler<> prof(ctx);

	if (!ctx) throw std::runtime_error("No devices available.");
	std::cout << ctx << std::endl;
	
	vex::Random<float, vex::random::threefry> rnd;

	size_t layer1 = 3; // 1 << 8;
	size_t layer2 = 50; // 1 << 10;
	size_t layer3 = 2; // 1 << 6;

//	size_t layer1 = 1 << 10;
//	size_t layer2 = 1 << 12;
//	size_t layer3 = 1 << 5;

	prof.tic_cl("initialise");

	ann::backprop_net<float> net;
	
	net.reset(ctx, std::vector<size_t>{layer1, layer2, layer3});
	net.random_initialise();
	
	prof.toc("initialise");

	std::vector<std::vector<float>> ip{
		{0.5f, -0.8f, 1.f},
		{-0.2f, -0.2f, 1.f},
		{0.7f, 0.2f, 1.f},
		{-0.3f, -0.5f, 1.f},
		{-0.7f, 0.4f, 1.f}};

	std::vector<std::vector<float>> tp{
		{-0.2f, 0.6f},
		{-0.7f, -0.6f},
		{0.3f, -0.9f},
		{-0.1f, -0.2f},
		{0.4f, 0.2f}};

	std::vector<size_t> indices(ip.size());
	std::iota(indices.begin(), indices.end(), 0);

	prof.tic_cl("train");

	for (int i = 0; i < 10000; ++i)
	{
		std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

		for (size_t j : indices)
		{
			net.forward_pass(ip[j]);
			net.backward_pass(0.05f, ip[j], tp[j]);
		}
	}

	prof.toc("train");

	for (auto& in : ip)
		print(net.forward_pass(in));
	
	std::cout << prof << std::endl;

	system("pause");
	return 0;



	float eta = 0.4f;

	prof.tic_cl("initialise");

	vex::vector<float> input{ctx, layer1};
	input.resize(ip[0]);
//	input.resize(std::vector<float>{2.f, -3.f, 1.f});
//	input.resize(std::vector<float>{1.f, 0.f, 1.f, 1.f});
//	input = 2 * rnd(vex::element_index(), std::rand()) - 1;

	ann::layer<float> layer_1{ctx, layer2, layer1, true};
//	layer_1.set_weights({0.1f, 0.2f, 0.3f, 0.f, 0.4f, 0.5f, 0.6f, 0.f, 0.7f, 0.8f, 0.9f, 0.f});
//	layer_1.set_weights({0.2f, -0.3f, 0.f, 0.4f, 0.1f, 0.0f, -0.5f, 0.2f, 0.f, -0.4f, 0.2f, 0.f});
	layer_1.random_initialise();

//	print(layer_1.weights);

	ann::layer<float> layer_2{ctx, layer3, layer2, false};
//	layer_2.set_weights({0.15f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f, 0.75f, 0.85f});
//	layer_2.set_weights({-0.3f, -0.2f, 0.1f});
	layer_2.random_initialise();

	//print(layer_2.weights);

	vex::vector<float> target{ctx, layer3};
	target.resize(tp[0]);
//	target.resize(std::vector<float>{-0.5f, 0.8f});
//	target.resize(std::vector<float>{1.f});
//	target = 2 * rnd(vex::element_index(), std::rand()) - 1;

	prof.toc("initialise");

	auto do_pass = [&]()
		{
			auto output =
				layer_2.activate(
				layer_1.activate(
				input));
			//	print(layer_1.activation);
			//	print(layer_2.activation);
			print(output);

			layer_2.compute_deltas(target);
			//	print(layer_2.deltas);
			layer_1.compute_deltas(layer_2);
			//	print(layer_1.deltas);

			layer_1.update_weights(eta, input);
			//	print(layer_1.weights); 
			layer_2.update_weights(eta, layer_1.activation);

			output =
				layer_2.activate(
					layer_1.activate(
						input));
			//	print(layer_1.activation);
			//	print(layer_2.activation);
			print(output);
		};


	net.set_weights(0, layer_1.weights);
	net.set_weights(1, layer_2.weights);

	auto op = net.forward_pass(ip[0]);
	print(op);
	net.backward_pass(eta, ip[0], tp[0]);
	op = net.forward_pass(ip[0]);
	print(op);

	prof.tic_cl("1st pass");

	do_pass();

	prof.toc("1st pass");

/*	prof.tic_cl("train");

	for (int i = 0; i < 1; ++i)
	{
		do_pass();
	}

	prof.toc("train");

//	print(layer_2.activation);
*/
	std::cout << prof << std::endl;

	system("pause");
    return 0;
}
