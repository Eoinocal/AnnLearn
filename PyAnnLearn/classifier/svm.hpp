
#pragma once

#include "py_pch.hpp"
#include <iostream>
#include <vector>

#include <annlearn/svm.hpp>
#include <annlearn/print.hpp>
#include <annlearn/vex_vector_io.hpp>

namespace annlearn
{

class py_svm
{
public:
	py_svm() :
		prof_(vex::current_context())
	{}

	void fit(const np::ndarray& py_x_train, const np::ndarray& py_y_train, size_t num_classes, double eta = 0.01, int epochs = 1000)
	{
		prof_.tic_cl("train");

		auto input = to_ann_matrix<double>(py_y_train);
		vex::vector<double> tmp1 = input.column(0);
		std::vector<double> tmp2(tmp1.size());
		vex::copy(tmp1, tmp2);

		std::vector<size_t> classes(tmp2.size());
		for (size_t i = 0, e = tmp2.size(); i < e; ++i)
			classes[i] = static_cast<size_t>(tmp2[i]);

		svm_.fit(to_ann_matrix<double>(py_x_train), classes, num_classes, epochs);
		prof_.toc("train");
	}

	np::ndarray predict(const np::ndarray& input)
	{
		prof_.tic_cl("predict");
		auto p = svm_.predict(to_ann_matrix<double>(input));
		prof_.toc("predict");
		
		return to_ndarray(p);
	}

	void print_profile()
	{
		std::cout << prof_ << std::endl;
	}

private:
	annlearn::svm<double> svm_;
	vex::profiler<> prof_;
};

}
