
#pragma once

#include "prod.hpp"

namespace annlearn
{

VEX_FUNCTION(float, sigmoid, (double, x),
	return 1 / (1 + exp(-x));
);

VEX_FUNCTION(float, sigmoid_dx, (double, x),
	return x * (1 - x);
);

VEX_FUNCTION(float, hypertan, (double, x),
	return tanh(x);
);

VEX_FUNCTION(float, hypertan_dx, (double, x),
	return (1 - x) * (1 + x);
);


VEX_FUNCTION(float, get_delta_x_activation, (size_t, n)(size_t, j)(double*, d)(double*, a),
	return d[j % n] * a[j / n];
);

VEX_FUNCTION(float, get_delta, (size_t, n)(size_t, j)(double*, d),
	return d[j % n];
);

VEX_FUNCTION(float, get_activation, (size_t, n)(size_t, j)(double*, a),
	return a[j / n];
);

VEX_FUNCTION(float, summed_delta, (size_t, n)(size_t, j)(double*, d)(double*, w),

	float sum = 0.0;

	for (size_t i = 0; i < n; ++i)
		sum += d[i] * w[j*n + i];

	return sum;
);

template<typename T>
class layer
{
public:
	layer(const std::vector<vex::backend::command_queue> &queue, size_t layer_size, size_t previous_layer) :
		weights{queue, layer_size*previous_layer},
		bias_weights{queue, layer_size},
		activation(queue, layer_size),
		activation_stale_{true}
	{
		deltas.resize(activation.size());
	}

	void random_initialise()
	{
		vex::Random<float, vex::random::threefry> rnd;
		weights = 2 * rnd(vex::element_index(), std::rand()) - 1;
		activation = 2 * rnd(vex::element_index(), std::rand()) - 1;
		deltas.resize(activation.size());
	}
	
	void set_weights(const std::vector<T>& v, const std::vector<T>& b)
	{
		weights.resize(v);
		bias_weights.resize(b);
	}

	void set_weights(const vex::vector<T>& v)
	{
		auto x = activation.size();
		auto y = weights.size() / x;

		assert(v.size() == (x) * (y + 1));

		vex::slicer<2> slice(vex::extents[y + 1][x]);

		weights.resize(slice[vex::range(0, y)][vex::_](v));
		bias_weights.resize(slice[y][vex::_](v));
	}

	template<typename E>
	const auto& activate(const E& input)
	{
		activation_stale_ = false;

		auto net = vec_mat_prod(input, weights) + bias_weights;
		return activation = hypertan(net);
	}

	template<typename TT>
	const auto& compute_deltas(const TT& target)
	{
		assert(!activation_stale_);

		return deltas = (activation - target) * hypertan_dx(activation);
	}

	const auto& compute_deltas(layer<T>& above)
	{
		assert(!activation_stale_);

		return deltas = summed_delta(above.deltas.size(), vex::element_index(), vex::raw_pointer(above.deltas), vex::raw_pointer(above.weights))
			* hypertan_dx(activation);
	}
	
	void update_weights(T eta, const vex::vector<T>& input)
	{
		weights -= eta
			* get_delta_x_activation(deltas.size(), vex::element_index(), vex::raw_pointer(deltas), vex::raw_pointer(input));
//			* get_activation(deltas.size(), vex::element_index(), vex::raw_pointer(input)) // NOT a bug that delta's size is used
//			  get_delta(deltas.size(), vex::element_index(), vex::raw_pointer(deltas));

		bias_weights -= eta * get_delta(deltas.size(), vex::element_index(), vex::raw_pointer(deltas));

		activation_stale_ = true;
	}

	size_t layer_size() const { return activation.size(); }

	vex::vector<T> weights;
	vex::vector<T> bias_weights;
	vex::vector<T> activation; 
	vex::vector<T> deltas;

private:
	bool activation_stale_;
};

}
