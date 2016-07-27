
#include "stdafx.hpp"

#include "annlearn/neuron.hpp"
#include "annlearn/fc_layer.hpp"
#include "annlearn/backprop_net.hpp"
#include "annlearn/svm.hpp"
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

#if 0
	auto X{ann::load_matrix<double>(std::ifstream{"blobs_X.xml"})};
	auto y{ann::load_matrix<double>(std::ifstream{"blobs_y.xml"})};
	auto y_hot{ann::load_matrix<double>(std::ifstream{"blobs_y_hot.xml"})};

//	print(y);

	vex::vector<double> tmp1 = y.column(0);
	std::vector<double> tmp2(tmp1.size());
	vex::copy(tmp1, tmp2);

	std::vector<size_t> classes(tmp2.size());
	for (size_t i = 0, e = tmp2.size(); i < e; ++i)
		classes[i] = static_cast<size_t>(tmp2[i]);

	auto svm = ann::svm<double>{};

	svm.fit(X, classes, 3);

	print(svm.weights());

	auto result = svm.predict(X);

	print(y);
	print(result);

/*	size_t idx = 10;

	size_t correct = classes[idx];
	vex::vector<double> row = X.row(idx);
	vex::vector<double> output = svm.map(X.row(idx));
	vex::vector<double> target = y_hot.row(idx);

	ann::print(row);
	ann::print(output);
	ann::print(target);

	std::cout << correct << " -> " << svm.loss(output, correct) << std::endl << std::endl;

	svm.update(row, correct);

	output = svm.map(row);

	ann::print(row);
	ann::print(output);
	ann::print(target);

	std::cout << svm.loss(output, correct) << std::endl << std::endl;

	svm.update(row, correct);
	output = svm.map(row);
	ann::print(output);
	std::cout << svm.loss(output, correct) << std::endl << std::endl;

	svm.update(row, correct);
	output = svm.map(row);
	ann::print(output);
	std::cout << svm.loss(output, correct) << std::endl << std::endl;

	svm.update(row, correct);
	output = svm.map(row);
	ann::print(output);
	std::cout << svm.loss(output, correct) << std::endl << std::endl;
*/
/*	std::vector<size_t> tests{0, 10, 12, 23, 34, 05, 36};

	for (auto idx : tests)
	{
		size_t correct = classes[idx];
		vex::vector<double> row = X.row(idx);
		vex::vector<double> output = svm.map(row);
		vex::vector<double> target = y_hot.row(idx);

		ann::print(row);
		ann::print(output);
		ann::print(target);

		std::cout << correct << " -> " << svm.loss(output, correct) << std::endl << std::endl;

/*		svm.update(output, correct);

		row = X.row(0);
		output = svm.map(X.row(0));

		ann::print(row);
		ann::print(output);
		ann::print(target);

		std::cout << svm.loss(output, correct) << std::endl;*/
//	}

#elif 1

	auto X{ann::load_matrix<double>(std::ifstream{"blobs_X.xml"})};
	auto y{ann::load_matrix<double>(std::ifstream{"blobs_y_hot.xml"})};

	ann::backprop_net<double> net{ctx,{X.ncol(), 10, y.ncol()}};
	net.random_initialise();

	auto& layer1 = net.get_layer(0);

	ann::backprop_net<double, ann::fc_layer<double>>::save(std::ofstream{"hyper_tan_backprop_net.xml"}, net);

	net.forward_pass(X.row(0));

	ann::print(layer1.activation());
	ann::save_vector(std::ofstream{"hyper_tan_layer1_target.xml"}, layer1.activation());

	auto& layer2 = net.get_layer(1);

	ann::save_vector(std::ofstream{"hyper_tan_layer2_target.xml"}, layer2.activation());

	vex::vector<double> r = y.row(0);

	auto& deltas2 = layer2.compute_deltas(r);

	ann::print(deltas2);
	ann::save_vector(std::ofstream{"hyper_tan_deltas2_target.xml"}, deltas2);

	auto& deltas1 = layer1.compute_deltas(layer2);

	ann::print(deltas1);
	ann::save_vector(std::ofstream{"hyper_tan_deltas1_target.xml"}, deltas1);


	net.forward_pass(X.row(0));
	net.backward_pass(1.0, X.row(0), y.row(0));

	ann::print(layer2.weights);
	ann::save_vector(std::ofstream{"hyper_tan_weights2_target.xml"}, layer2.weights);

	ann::print(layer1.weights);
	ann::save_vector(std::ofstream{"hyper_tan_weights1_target.xml"}, layer1.weights);


//	ann::save_vector(std::ofstream{"hyper_tan_net_target.xml"}, out);

#elif 0

	auto X{ann::load_matrix<double>(std::ifstream{"blobs_X.xml"})};
	auto y_hot{ann::load_matrix<double>(std::ifstream{"blobs_y_hot.xml"})};
	
//	std::cout << "Inputs" << std::endl;
//	ann::print(X);
//	ann::print(y_hot);

	prof.tic_cl("Initialise");

	ann::backprop_net<double> net{ctx, {X.ncol(), 200, y_hot.ncol()}};
	net.random_initialise();	

	prof.toc("Initialise");

//	std::cout << "Pre training prediction error: " << l2_norm(net.predict(x_train).data - y_train.data) << std::endl;

//	ann::print(net.predict(x_train));
//	std::cout << "Error: " << l2_norm(net.predict(x_train).data - y_train.data) << std::endl;

	prof.tic_cl("Train");

	net.fit(X, y_hot, 0.1, 200);

	prof.toc("Train");

//	vex::vector<double> predictions = vex::round(net.predict(x_train).data);
//	ann::print(predictions);
//	std::cout << "Post training prediction error: " << l2_norm(predictions - y_train.data) << std::endl;

//	std::cout << "Error: " << l2_norm(predictions.data - y_train.data) << std::endl;

#elif 0
	size_t in = 1 << 13;
	size_t out = 1 << 13;

	ann::backprop_net<double> net(ctx, std::vector<size_t>{in, 1 << 12, out});

	vex::vector<double> input(ctx, in);
	input = 2 * rnd(vex::element_index(), std::rand()) - 1;
	vex::vector<double> target(ctx, out);
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
