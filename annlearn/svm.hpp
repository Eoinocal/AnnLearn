
#pragma once

#include "layer.hpp"


namespace annlearn
{

VEX_FUNCTION(double, threshold, (double, x),
	return x > 0.0 ? 1.0 : 0.0;
);

VEX_FUNCTION(float, get_grad, (size_t, n)(size_t, i)(double*, grad),
	return grad[i / n];
);

VEX_FUNCTION(float, get_grad_by_input, (size_t, n)(size_t, i)(double*, grad)(double*, input),
	return grad[i / n] * input[i % n];
);

template<typename T>
class svm
{
public:
	svm() :
		delta_{1.1},
		rho_{0.01}
	{}

	void fit(const matrix<T>& x_train, const std::vector<size_t>& y_train, size_t classes, int epochs = 100)
	{
		weights_.resize(x_train.ncol(), classes);
		bias_weights_.resize(classes);

		random_initialise();

		std::vector<size_t> indices(x_train.nrow());
		std::iota(indices.begin(), indices.end(), 0);

		boost::progress_display show_progress(epochs);

		for (int i = 0; i < epochs; ++i)
		{
			std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

			for (size_t j : indices)
			{
				auto t_i = y_train[j];
				update(x_train.row(j), t_i);
			}

			++show_progress;
		}
	}

	template<typename Expr>
	vex::vector<T> map(Expr input)
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
		return sum(max(0.0, input - s_j + delta_)) - delta_;
	}
		
	void update(const vex::vector<T>& input, size_t class_idx)
	{
		vex::vector<T> mapped = map(input);

		vex::Reductor<T, vex::SUM> sum{vex::current_context()};
		T s_j = mapped[class_idx];

		vex::vector<T> grad = threshold(mapped - s_j + delta_);
		grad[class_idx] = 0.0;
		grad[class_idx] = -1.0 * sum(grad);

/*		std::cout << "Class " << class_idx << std::endl;
		print(input);
		print(mapped);
		print(grad);
		print(weights_);
*/		
		weights_.data -= get_grad_by_input(weights_.ncol(), vex::element_index(), vex::raw_pointer(grad), vex::raw_pointer(input)) * rho_;
		bias_weights_ -= get_grad(weights_.ncol(), vex::element_index(), vex::raw_pointer(grad)) * rho_;
	} 

	size_t classify(const vex::vector<T> input)
	{
		auto mapped{map(input)};
		return std::distance(std::begin(mapped), std::max_element(std::begin(mapped), std::end(mapped)));
	}

	template<typename T>
	matrix<T> predict(const matrix<T>& input)
	{
		std::vector<T> output(input.nrow());

		std::vector<size_t> indices(input.nrow());
		std::iota(indices.begin(), indices.end(), 0);

		boost::progress_display show_progress(static_cast<unsigned long>(indices.size()));

		for (size_t j : indices)
		{
			output[j] = static_cast<double>(classify(input.row(j)));
			++show_progress;
		}

		return matrix<T>{1, input.nrow(), output};
	}

	const matrix<T>& weights() const{return weights_; }
	
private:
	void random_initialise()
	{
		T scale = -1.1f;

		vex::RandomNormal<T, vex::random::threefry> rnd;
		weights_.data = rnd(vex::element_index(), std::rand()) * scale;
		bias_weights_ = rnd(vex::element_index(), std::rand()) * scale;
	}

	T delta_;
	T rho_;
	matrix<T> weights_;
	vex::vector<T> bias_weights_;
};

}
