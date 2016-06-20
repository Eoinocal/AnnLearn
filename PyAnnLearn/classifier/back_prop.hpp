
#pragma once

#include "stdafx.hpp"
#include <iostream>
#include <vector>

#include <annlearn/backprop_net.hpp>
#include <annlearn/print.hpp>

namespace annlearn
{
	
class back_prop
{
public:
	back_prop()
	{
		std::cout << "Using context(s):\n" << context() << std::endl;		
	}

	void reset(const py::object& a)
	{
		net_.reset(context(), to_std_vector<size_t>(a));
		net_.random_initialise();
		std::cout << "Net layers " << net_.num_layers() << std::endl;
	}

	void fit(const np::ndarray& py_x_train, const np::ndarray& py_y_train)
	{
		vex::profiler<> prof(context());

		prof.tic_cl("convert");

		auto x_train = to_ann_matrix<double>(py_x_train);
		auto y_train = to_ann_matrix<double>(py_y_train);

		std::vector<size_t> indices(x_train.nrow());
		std::iota(indices.begin(), indices.end(), 0);

		prof.toc("convert");

		prof.tic_cl("train");
		
		for (int i = 0; i < 1000; ++i)
		{
			std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

			for (size_t j : indices)
			{
				net_.forward_pass(x_train.row(j));
				net_.backward_pass(0.05f, x_train.row(j), y_train.row(j));
			}
		}

		std::cout << "out" << std::endl;
		for (size_t j : indices)
		{
			annlearn::print(vex::vector<double>(net_.forward_pass(x_train.row(j))));
		}
		std::cout << "target" << std::endl;
		for (size_t j : indices)
		{
			annlearn::print(vex::vector<double>(y_train.row(j)));
		}
		std::cout << "diff" << std::endl;
		for (size_t j : indices)
		{
			annlearn::print(vex::vector<double>(net_.forward_pass(x_train.row(j)) - y_train.row(j)));
		}

		prof.toc("train");
		std::cout << prof << std::endl;
	}

private:
	annlearn::backprop_net<double> net_;
};

}
