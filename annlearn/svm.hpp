
#pragma once

#include "layer.hpp"


namespace annlearn
{

VEX_FUNCTION(double, threshold, (double, x),
	return x > 0.0 ? 1.0 : 0.0;
);

template<typename T>
class svm
{
public:
	svm() :
		delta_{10}
	{}

	void fit(const matrix<T>& x_train, size_t classes)
	{

		weights_.resize(x_train.ncol(), classes);
		bias_weights_.resize(classes);

		random_initialise();

		print(weights_);
		print(bias_weights_);
/*		std::vector<size_t> indices(x_train.nrow());
		std::iota(indices.begin(), indices.end(), 0);

		boost::progress_display show_progress(epochs);

		for (int i = 0; i < epochs; ++i)
		{
			std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

			for (size_t j : indices)
			{
				forward_pass(x_train.row(j));
				backward_pass(eta, x_train.row(j), y_train.row(j));
			}

			++show_progress;
		}
*/	}

	template<typename Expr>
	auto map(Expr input)
	{
		return prod(weights_, input) + bias_weights_;
	}		
	
	T loss(const vex::vector<T>& input, size_t class_idx)
	{
		return loss(input, input[class_idx]);
	}

	template<typename Expr>
	T loss(Expr input, T s_j)
	{
		vex::Reductor<T, vex::SUM> sum{vex::current_context()};
		return sum(threshold(input - s_j + delta_)) - 1.0;
	}

	//T loss_dx(const vex::vector<T>& input, size_t class_idx)


private:
	void random_initialise()
	{
		T scale = 0.1f;

		vex::RandomNormal<T, vex::random::threefry> rnd;
		weights_.data = rnd(vex::element_index(), std::rand()) * scale;
		bias_weights_ = (2 * rnd(vex::element_index(), std::rand()) - 1) * scale;
	}

	T delta_;
	matrix<T> weights_;
	vex::vector<T> bias_weights_;
};

}
