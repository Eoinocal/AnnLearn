
#pragma once

#include "stdafx.hpp"
#include <iostream>
#include <vector>

#include <annlearn/backprop_net.hpp>
#include <annlearn/print.hpp>
#include <annlearn/vex_vector_io.hpp>

namespace annlearn
{
	
class back_prop
{
public:
	back_prop() :
		prof_(context())
	{
		std::cout << "Using context(s):\n" << context() << std::endl;		
	}

	void reset(const py::object& a)
	{
		auto ls = to_std_vector<size_t>(a);
		net_.reset(context(), ls);
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

	/*	prof.tic_cl("convert");

		auto x_train = to_ann_matrix<double>(py_x_train);
		auto y_train = to_ann_matrix<double>(py_y_train);

		{	std::ofstream ofs("iris.txt");
			boost::archive::xml_oarchive oa(ofs);
			oa << annlearn::make_nvp("x_train", x_train);
			oa << annlearn::make_nvp("y_train", y_train);
		}

		std::vector<size_t> indices(x_train.nrow());
		std::iota(indices.begin(), indices.end(), 0);

		prof.toc("convert");

		prof.tic_cl("train");
		
		for (int i = 0; i < 5000; ++i)
		{
			std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

			for (size_t j : indices)
			{
				net_.forward_pass(x_train.row(j));
				net_.backward_pass(0.1f, x_train.row(j), y_train.row(j));
			}
		}
		*/
	//	std::cout << "out" << std::endl;
		//annlearn::print(net_.predict(x_train));

	//	std::cout << "target" << std::endl;
		//annlearn::print(y_train);

		prof_.toc("train");
	}

	np::ndarray predict(const np::ndarray& input)
	{
		prof_.tic_cl("predict");
	//	print(to_ann_matrix<double>(input));
		auto p = net_.predict(to_ann_matrix<double>(input));
	//	print(p);
	//	p.data = vex::round(p.data);

		prof_.toc("predict");
		return to_ndarray(p);
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
