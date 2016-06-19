
#pragma once

#include "prod.hpp"

namespace annlearn
{

VEX_FUNCTION(float, sigmoid, (float, x),
	return 1 / (1 + exp(-x));
);

VEX_FUNCTION(float, sigmoid_dx, (float, x),
	return x * (1 - x);
);

VEX_FUNCTION(float, hypertan, (float, x),
	return tanh(x);
);

VEX_FUNCTION(float, hypertan_dx, (float, x),
	return (1 - x) * (1 + x);
);

VEX_FUNCTION(float, update_weight, (size_t, n)(size_t, j)(float*, w)(float*, d)(float*, a),
	return  d[j % n] * a[j / n];
);

VEX_FUNCTION(float, summed_delta, (size_t, n)(size_t, j)(float*, d)(float*, w),

	float sum = 0.0;

	for (size_t i = 0; i < n; ++i)
		sum += d[i] * w[j*n + i];

	return sum;
);

template<typename T>
class layer
{
public:
	layer(const std::vector<vex::backend::command_queue> &queue, size_t layer_size, size_t previous_layer, bool has_bias) :
		weights{queue, layer_size*previous_layer},
		activation(queue, layer_size),
		has_bias_{has_bias}
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

	void set_weights(const std::vector<T>& v)
	{
		weights.resize(v);
	}

	template<typename E>
	auto& activate(E& input)
	{
		auto net = vec_mat_prod(input, weights);
		activation = hypertan(net);

		if (has_bias_)
			activation[activation.size() - 1] = 1.f; // correct bias

		return activation;
	}

//	template<typename TT>
	void compute_deltas(vex::vector<T>& target)
	{
		deltas = (activation - target) * hypertan_dx(activation);
	}

	void compute_deltas(layer<T>& above)
	{
		deltas = summed_delta(above.deltas.size(), vex::element_index(), vex::raw_pointer(above.deltas), vex::raw_pointer(above.weights))
			* hypertan_dx(activation);
	}

	template<typename I>
	void update_weights(T eta, I& input)
	{
		weights = weights - eta * update_weight(deltas.size(), vex::element_index(), vex::raw_pointer(weights), vex::raw_pointer(deltas), vex::raw_pointer(input));
	}

	size_t layer_size() const { return activation.size(); }

	vex::vector<T> weights;
	vex::vector<T> activation; 
	vex::vector<T> deltas;

private:
	bool has_bias_;
};

}
