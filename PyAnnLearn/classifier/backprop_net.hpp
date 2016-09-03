
#pragma once

#include "py_pch.hpp"
#include <iostream>
#include <vector>

#include <annlearn/backprop_net.hpp>
#include <annlearn/print.hpp>
#include <annlearn/vex_vector_io.hpp>

namespace annlearn
{
	
class py_backprop_net
{
public:
	py_backprop_net() :
		prof_(vex::current_context())
	{}

	void reset(const py::object& a)
	{
		auto ls = to_std_vector<size_t>(a);
		net_.reset(vex::current_context(), ls);
		net_.random_initialise();

		std::cout << "Net layers " << net_.num_layers() << " -> ";

		for (auto& l : ls)
			std::cout << l << " ";

		std::cout << std::endl;
	}

	void fit(const np::ndarray& py_x_train, const np::ndarray& py_y_train, double eta= 0.01, int epochs=1000)
	{
		prof_.tic_cl("train");
		net_.fit(to_ann_matrix<double>(py_x_train), to_ann_matrix<double>(py_y_train), eta, epochs);
		prof_.toc("train");
	}

	np::ndarray predict(const np::ndarray& input)
	{
		prof_.tic_cl("predict");
		auto p = net_.predict(to_ann_matrix<double>(input));
		prof_.toc("predict");

		return to_ndarray(p);
	}

	np::ndarray get_weights(int layer)
	{
		auto& l = net_.get_layer(layer);
		auto w = l.weights;
		auto W = annlearn::matrix<double>(l.output_size(), l.input_size(), w);

		return to_ndarray(W);
	}

	void print_profile()
	{
		std::cout << prof_ << std::endl;
	}

private:
	annlearn::backprop_net<double> net_;
	vex::profiler<> prof_;
};

}
